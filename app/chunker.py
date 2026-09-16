"""中文友好的 Markdown 切块器。

设计思路（这是整个项目最需要理解的一段逻辑，请重点读）：

    原文
      └─ 按 Markdown 标题切成 section        ← 保留章节边界，不跨章节拼
           └─ 每个 section 按空行切成段落段落
                └─ 段落过长时按中文句号切成句子
                     └─ 贪心合并到接近 chunk_size
                          └─ 相邻 chunk 保留 overlap 尾巴

为什么不"每 N 个字符硬切"：硬切会把一句话从中间劈开，chunk 开头变成半句，
检索命中的时候人和模型都读不懂。按标题/段落/句子切，边界落在语义单元上。

为什么不跨 section 合并：跨章节拼出来的 chunk 语义混杂，召回后会污染上下文。
代价是短章节可能产生偏小的 chunk，这是可以接受的取舍。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Markdown 标题：1~6 个 # 加空格
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# 中文句子边界：句号、问号、感叹号、分号（英文也带上），用零宽断言切在标点之后
SENTENCE_RE = re.compile(r"[^。！？；\?\!\n]+[。！？；\?\!\n]*")
# 段落边界：从一个非换行字符起，一直延伸到下一个空行（或正文末尾）之前。
# 用前瞻而不是消费掉空行，这样相邻段落的下标是严格连续的，切片不会漏字。
_PARAGRAPH_RE = re.compile(r"[^\n][\s\S]*?(?=\n\s*\n|\Z)")


@dataclass(frozen=True)
class Segment:
    """一段原文片段。start/end 是它在原文中的字符下标区间（左闭右开）。

    带上标下标的好处：SQLite 里存的 start_position/end_position 是真的能对回原文，
    调试时可以直接定位"这个 chunk 到底来自文档哪一段"。
    """

    text: str
    start: int
    end: int


@dataclass
class Chunk:
    """切块结果。chunk_index 由调用方在整个文档范围内统一编号。"""

    content: str
    heading: str
    start_position: int
    end_position: int


def chunk_markdown(
    content: str,
    heading: str = "",
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """把一篇 Markdown 正文切成 Chunk 列表。heading 是文档标题，用于没有标题时的兜底。"""
    chunks: list[Chunk] = []
    for section_heading, start, end, body_start in _split_sections(content):
        # 只有标题行、没有正文的 section 直接跳过：否则每篇文档都会多出一个
        # 十几个字的垃圾 chunk，标题信息其实已经在 Chunk.heading 和 document.title 里了。
        if not content[body_start:end].strip():
            continue
        effective_heading = section_heading or heading
        stream = _section_segments(content, start, end, chunk_size)
        chunks.extend(_merge_segments(content, stream, effective_heading, chunk_size, overlap))
    return chunks


def _line_offsets(text: str) -> list[int]:
    """算出每一行在原文中的起始下标，供标题定位使用。"""
    offsets: list[int] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        offsets.append(pos)
        pos += len(line)
    return offsets


def _split_sections(text: str) -> list[tuple[str, int, int, int]]:
    """按 Markdown 标题把原文切成 [(heading, start, end, body_start), ...]。

    start 从标题行本身开始，这样 section 的第一个 chunk 天然带着标题，
    即使模型只看到 chunk 内容也知道它在讲哪一节。
    body_start 是标题行之后的正文起始位置：body_start 之后没有内容时，
    说明这个 section 只有标题没有正文，调用方应跳过。
    标题之前如果有正文（前言），单独作为一个 heading 为空的 section。
    """
    lines = text.splitlines(keepends=True)
    offsets = _line_offsets(text)

    sections: list[tuple[str, int, int, int]] = []
    pending_start = 0
    pending_heading = ""
    pending_body_start = 0
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if not m:
            continue
        # 只有"标题之前确实还有内容"时才把前面那段封成一个 section
        if i > 0 and pending_start < offsets[i]:
            sections.append((pending_heading, pending_start, offsets[i], pending_body_start))
        pending_start = offsets[i]
        pending_heading = m.group(2).strip()
        pending_body_start = offsets[i] + len(line)
    sections.append((pending_heading, pending_start, len(text), pending_body_start))
    return sections


def _section_segments(text: str, start: int, end: int, chunk_size: int) -> list[Segment]:
    """把 section 切成段落级片段；单个段落超过 chunk_size 时再细分成句子。

    段落 = 空行分隔的连续行块。按段落切而不是按字符数切，
    保证 chunk 边界落在自然语言单元上，不会把一句话劈成两半。
    """
    body = text[start:end]
    segments: list[Segment] = []
    for m in _PARAGRAPH_RE.finditer(body):
        seg_text = body[m.start() : m.end()]
        rel_start, rel_end = start + m.start(), start + m.end()
        if len(seg_text) > chunk_size:
            segments.extend(_split_sentences(seg_text, rel_start))
        else:
            segments.append(Segment(seg_text, rel_start, rel_end))
    return segments


def _split_sentences(text: str, base: int) -> list[Segment]:
    """按中文标点切句。base 是这段文字在整篇文档中的起始下标，返回的是绝对下标。"""
    segments: list[Segment] = []
    for m in SENTENCE_RE.finditer(text):
        piece = text[m.start() : m.end()]
        if not piece.strip():
            continue
        segments.append(Segment(piece, base + m.start(), base + m.end()))
    return segments


def _merge_segments(
    text: str,
    stream: list[Segment],
    heading: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """贪心合并片段成 chunk，并在相邻 chunk 之间保留 overlap 的尾巴。"""
    chunks: list[Chunk] = []
    current: list[Segment] = []
    current_len = 0

    for seg in stream:
        # 当前 chunk 已经攒够，且再加这一段会超目标长度 → 先封口
        if current and current_len + len(seg.text) > chunk_size:
            _flush(chunks, text, current, heading)
            current, current_len = _overlap_tail(current, overlap)
        current.append(seg)
        current_len += len(seg.text)

    if current:
        _flush(chunks, text, current, heading)
    return chunks


def _overlap_tail(segments: list[Segment], overlap: int) -> tuple[list[Segment], int]:
    """从上一个 chunk 的末尾取回若干片段作为下一个 chunk 的前缀。

    注意：取回的是"整句/整段"，不是从中间截字符，否则位置下标会对不上。
    """
    if overlap <= 0:
        return [], 0
    tail: list[Segment] = []
    total = 0
    for seg in reversed(segments):
        tail.insert(0, seg)
        total += len(seg.text)
        if total >= overlap:
            break
    return tail, total


def _flush(chunks: list[Chunk], text: str, segments: list[Segment], heading: str) -> None:
    """把一组片段封成一个 Chunk。

    片段在同一 section 内相邻（中间只有空行），所以直接按 [start, end] 切片
    拿到的就是原文区域。

    存进去的 content 做了 strip（去掉 chunk 边缘的换行和缩进），所以下标必须
    跟着收紧到 strip 之后的真实起止位置。否则原文[start:end] 会比 content 多出
    两头的空白，引用定位就会指到空白行而不是指到那句话。
    """
    if not segments:
        return
    start, end = segments[0].start, segments[-1].end
    raw = text[start:end]
    lead = len(raw) - len(raw.lstrip())   # 头部要跳过的空白数
    trail = len(raw) - len(raw.rstrip())  # 尾部要跳过的空白数
    content = raw.strip()
    if not content:
        return
    chunks.append(
        Chunk(
            content=content,
            heading=heading,
            start_position=start + lead,
            end_position=end - trail,
        )
    )
