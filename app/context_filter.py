"""retrieval 和 LLM context 之间的轻量清洗层。

单独成模块而不是塞进 retriever，是因为检索层的职责只有一件——按相似度排序。
过滤和文本清理是另一件职责，混在一起就再也拿不回"原始检索结果"，
评估时也就复现不出 baseline。

只处理两种在真实数据里已经确认存在、且能用明确模式识别的情况：

    1. navigation chunk：正文几乎全是 `[[wiki link]]` / markdown 链接 / 目录跳转。
       判断依据是链接字符占比，不看 heading 的名字——一篇叫"相关链接"的文档
       完全可能有实质内容。

    2. JSON 导出尾巴：入库时把导出器的 metadata 一起带进了正文，形如
       `...深刻影响了经济学。", "file_path": "/c/Users/.../x.md"}`。
       只从文本末尾剥离，且字段名必须在白名单里，正文中间的 JSON 一律不动。

SQLite 里的 chunk 原文不改。这里产出的 context_content 只给 LLM 看，
sources 仍然引用原始 chunk。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .retriever import SearchResult

# 启用过滤时先多取这么多条，给过滤留余量。20 = 默认 top_k(5) × 4，
# 导航 chunk 占比再高也够补满 5 条。
RETRIEVE_TOP_K = 20

# 链接字符占非空白正文的比例超过这个值判为导航。
NAVIGATION_LINK_RATIO = 0.80
# 至少要这么多链接才算导航。只有 1 个链接的短正文不可能"几乎全是链接"，
# 这个下限挡掉短正文被误判的情况。
NAVIGATION_MIN_LINKS = 2

# 一次最多剥几对 metadata。病态循环保护，真实残留只有一对。
MAX_METADATA_PAIRS = 4

# 只认这些 key。导出器常见的 metadata 字段；key 不在白名单里就不剥，
# 这样正文里出现的任何 `"xxx": "yyy"` 都碰不到。
_EXPORT_META_KEYS = frozenset(
    {
        "file_path", "filename", "path", "url", "name", "title", "id",
        "source", "heading", "chunk", "offset", "line", "start_line", "end_line",
    }
)

# [[目标]] / [[目标|别名]]
_WIKI_LINK = re.compile(r"\[\[[^\]]+\]\]")
# [文本](目标)，目录锚点 [章节](#标题) 也归这一类
_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\([^\)]+\)")
# 围栏代码块。统计链接占比时排除，避免讲 Markdown / JSON 语法的文档被判成导航。
_CODE_FENCE = re.compile(r"```[\s\S]*?```")
# 标题行，不算正文
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}.*$", re.MULTILINE)
# 末尾一对 metadata："<key>": "<value>"（或数字 / 布尔 / null），允许紧跟 ] }
_TRAILING_META = re.compile(
    r'"([A-Za-z_][A-Za-z0-9_]{0,39})"\s*:\s*'
    r'("[^"]*"|\d+(?:\.\d+)?|true|false|null)\s*[\]}]?\s*$',
)
# 剥完 metadata 后残留的引号 / 逗号 / 括号
_TRAILING_PUNCT = re.compile(r'["\',}\]]+\s*$')


@dataclass(frozen=True)
class ChunkJudgement:
    """一条 chunk 的清洗判定结果。"""

    is_navigation: bool
    link_ratio: float
    link_count: int
    metadata_stripped: bool


@dataclass(frozen=True)
class ContextChunk:
    """清洗后的候选 chunk。hit 保留原始命中，context_content 给 LLM 用。"""

    hit: SearchResult
    context_content: str
    judgement: ChunkJudgement


@dataclass(frozen=True)
class ContextSelection:
    """一轮 prepare_context 的产出。selected 进 context，dropped 只进报告。"""

    selected: list[ContextChunk]
    dropped: list[ContextChunk]


def _measure_links(text: str) -> tuple[int, float]:
    """统计正文里的链接个数和链接字符占比。

    占比的分母是去掉空白后的正文字符数。导航 chunk 通常是一串链接加换行，
    不剔空白的话占比会被换行稀释，阈值就定不准了。
    """
    body = _HEADING_LINE.sub("", _CODE_FENCE.sub("", text))
    matches = _WIKI_LINK.findall(body) + _MARKDOWN_LINK.findall(body)
    plain_chars = len(re.sub(r"\s", "", body))
    if plain_chars == 0:
        return 0, 0.0
    link_chars = sum(len(m) for m in matches)
    return len(matches), min(1.0, link_chars / plain_chars)


def judge_chunk(text: str) -> ChunkJudgement:
    """判断这条 chunk 是否为导航型，以及是否带导出 metadata 尾巴。"""
    link_count, link_ratio = _measure_links(text)
    return ChunkJudgement(
        is_navigation=(
            link_count >= NAVIGATION_MIN_LINKS
            and link_ratio >= NAVIGATION_LINK_RATIO
        ),
        link_ratio=link_ratio,
        link_count=link_count,
        metadata_stripped=strip_export_metadata(text) != text.rstrip(),
    )


def strip_export_metadata(text: str) -> str:
    """剥掉文本末尾的导出 metadata 尾巴。

    三个限制让这一步足够保守：
        - 只从末尾往前剥，正文中间出现的 JSON 完全不受影响
        - key 必须在 _EXPORT_META_KEYS 白名单里
        - 一旦剥到剩余部分以 { 或 [ 开头，说明这段文字本身就是一个 JSON 值，
          直接放弃返回原文——它不是"正文加尾巴"，整段都是导出数据
    """
    cleaned = text.rstrip()
    stripped_any = False
    for _ in range(MAX_METADATA_PAIRS):
        match = _TRAILING_META.search(cleaned)
        if not match or match.group(1).lower() not in _EXPORT_META_KEYS:
            break
        stripped_any = True
        # 连续多对时，剥掉最后一对后前面那对的结尾是逗号，得一并去掉，
        # 否则下一轮的末尾锚定匹配不上。
        remaining = cleaned[: match.start()].rstrip().rstrip(",").rstrip()
        if not remaining or remaining[0] in "{[":
            return text.rstrip()
        cleaned = remaining
    # 一轮都没剥到就不该碰末尾标点——否则会删掉正文里合法的引号或括号。
    if not stripped_any:
        return text.rstrip()
    cleaned = _TRAILING_PUNCT.sub("", cleaned)
    return cleaned if cleaned else text.rstrip()


def prepare_context(hits: list[SearchResult], top_k: int) -> ContextSelection:
    """从候选命中里挑出最终进 context 的 top_k 条。

    按相似度顺序遍历，导航 chunk 跳过，凑满 top_k 即停。
    不足 top_k 时返回剩余的全部——宁少不补垃圾。
    """
    selected: list[ContextChunk] = []
    dropped: list[ContextChunk] = []
    for hit in hits:
        judgement = judge_chunk(hit.content)
        chunk = ContextChunk(
            hit=hit,
            context_content=strip_export_metadata(hit.content),
            judgement=judgement,
        )
        if judgement.is_navigation:
            dropped.append(chunk)
            continue
        selected.append(chunk)
        if len(selected) == top_k:
            break
    return ContextSelection(selected=selected, dropped=dropped)
