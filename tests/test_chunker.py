"""测试三：Chunk 数量正常、位置下标对得上原文、边界落在自然语言单元上。"""
from __future__ import annotations

from app.chunker import _section_segments, chunk_markdown

# 含文档级 H1 + 多个二级标题 + 一个足够长的章节（能切出多块并触发 overlap）
LONG_MD = """# 向量数据库入门

## 什么是向量

向量是一串数字，用来表示一段文本的含义。语义相近的文本，向量之间的距离更近。

Embedding 模型负责把文本变成向量。同一个模型输出的向量才处在同一个空间里，
换模型之后旧向量就作废了，必须重新计算并重建索引。

## 为什么要切块

大模型有上下文长度限制，不可能把整个知识库一次性塞进去。
所以先把文档切成小块，检索时只取最相关的几块，再把它们拼成上下文。

切块太小，语义不完整，检索召回率下降。一个 chunk 如果只有一句话，
模型看不到前后文，容易答偏，也说不清答案到底来自哪里。

切块太大，噪声太多，LLM 容易被无关内容干扰，答案的引用也变得难以定位。
通常取几百个中文字符，并让相邻块重叠一点，避免信息被边界切断。

## 相似度的计算

最常见的度量是余弦相似度。它衡量两个向量方向的接近程度，
不关心向量的长度，所以适合比较文本。

## 检索与生成

RAG 的做法是先检索再生成。检索负责找到相关材料，生成负责基于材料组织答案。
"""

# chunk 结尾如果落在这几个字符上，说明边界切在了句读处
SENTENCE_ENDINGS = ("。", "！", "？", "，", "、", "；", "）", ")", "。 ")


def test_chunk_count_is_reasonable():
    chunks = chunk_markdown(LONG_MD, heading="向量数据库入门", chunk_size=600, overlap=100)
    # 4 个章节，其中「为什么要切块」较长，600 字一块应落在 2~4 块
    assert 2 <= len(chunks) <= 4


def test_title_only_section_is_dropped():
    """文档 H1 单独成 section 时，不应产生十几个字的垃圾 chunk。"""
    chunks = chunk_markdown(LONG_MD, chunk_size=600)
    h1 = LONG_MD.splitlines()[0].strip()
    assert all(c.content.strip() != h1 for c in chunks)
    assert all(len(c.content) >= 20 for c in chunks)


def test_every_chunk_is_a_real_slice_of_the_original():
    """位置下标必须真的能切出 chunk 内容，否则 citation 会指错地方。

    这里用"精确切片"而不是"子串包含"：content 是 strip 过的，下标也必须收紧到
    strip 之后的边界。只断言 content in 原文会放过两头的空白，那种情况下
    原文[start:end] 指的是空白行而不是那句话。
    """
    # 小 chunk_size 强制产生很多块，让每个块都紧贴空行/换行边界
    chunks = chunk_markdown(LONG_MD, chunk_size=120, overlap=30)
    for c in chunks:
        assert c.content.strip()
        assert 0 <= c.start_position < c.end_position <= len(LONG_MD)
        assert LONG_MD[c.start_position : c.end_position] == c.content


def test_chunks_stay_under_target_size():
    """允许因 overlap 略超目标，但不能翻倍。"""
    chunks = chunk_markdown(LONG_MD, chunk_size=600, overlap=100)
    assert all(len(c.content) <= 600 * 1.5 for c in chunks)


def test_boundaries_land_on_sentence_endings():
    """chunk 结尾应落在句读上——这是「不按固定字符数硬切」的直接证据。"""
    chunks = chunk_markdown(LONG_MD, chunk_size=300, overlap=40)
    for c in chunks:
        assert c.content.rstrip()[-1] in SENTENCE_ENDINGS, f"边界劈在句子中间：{c.content[-30:]!r}"


def test_paragraph_is_the_chunking_unit():
    """段落是切分的基本单位，不能退化成按字符数切。"""
    body = "第一段内容。\n第二行还是第一段。\n\n第二段内容。\n"
    segs = _section_segments(body, 0, len(body), 800)
    assert len(segs) == 2
    assert segs[0].text.startswith("第一段内容")
    assert "第二段" in segs[1].text


def test_long_paragraph_splits_into_sentences():
    body = "这是一个很长的句子，包含足够多的信息。" * 40
    segs = _section_segments(body, 0, len(body), 100)
    assert len(segs) > 1
    assert all(len(s.text) <= 100 for s in segs)


def test_empty_or_blank_section_yields_nothing():
    for text in ("", "\n\n\n", "\n\n正文中间有空行\n\n"):
        segs = _section_segments(text, 0, len(text), 100)
        assert len(segs) in (0, 1), f"{text!r} -> {segs}"


def test_headings_are_attached_to_their_chunks():
    """检索结果要显示章节名，所以每个 chunk 必须带上它所属的标题。"""
    chunks = chunk_markdown(LONG_MD, chunk_size=600, overlap=100)
    headings = {c.heading for c in chunks}
    assert "什么是向量" in headings
    assert "为什么要切块" in headings
    assert "相似度的计算" in headings
    for c in chunks:
        assert c.heading in LONG_MD


def test_section_title_leads_its_first_chunk():
    """第一个 chunk 的正文里带着章节标题，便于 embedding 理解上下文。"""
    chunks = chunk_markdown(LONG_MD, chunk_size=600, overlap=100)
    first_of = {c.heading: c.content for c in chunks}
    assert first_of["什么是向量"].startswith("## 什么是向量")


def test_sections_do_not_get_merged_across_headings():
    """一个 chunk 只属于一个章节，避免两个章节的内容混在一起。"""
    headings = ("什么是向量", "为什么要切块", "相似度的计算", "检索与生成")
    chunks = chunk_markdown(LONG_MD, chunk_size=600, overlap=100)
    for c in chunks:
        for h in headings:
            if h != c.heading:
                assert f"## {h}" not in c.content


def test_consecutive_chunks_overlap():
    """同一章节内相邻块要有重叠，否则切在句子之间的信息会丢。"""
    chunks = chunk_markdown(LONG_MD, chunk_size=120, overlap=30)
    pairs = [(a, b) for a, b in zip(chunks, chunks[1:]) if a.heading == b.heading]
    assert pairs, "该参数下应该存在同章节的相邻块"
    for a, b in pairs:
        # a 的后半段里应有 15 字符的连续片段出现在 b 中
        tail = a.content[len(a.content) // 2 :]
        assert any(tail[i : i + 15] in b.content for i in range(len(tail) - 15))


def test_cross_section_boundaries_do_not_overlap():
    """跨章节的相邻块不重叠是刻意为之：跨章节拼接会污染语义。"""
    chunks = chunk_markdown(LONG_MD, chunk_size=600, overlap=100)
    cross = [(a, b) for a, b in zip(chunks, chunks[1:]) if a.heading != b.heading]
    for a, b in cross:
        tail = a.content[len(a.content) // 2 :]
        assert not any(tail[i : i + 15] in b.content for i in range(len(tail) - 15))


def test_short_document_produces_one_chunk():
    assert len(chunk_markdown("# 短文档\n\n只有两句话。第一句。第二句。", chunk_size=800)) == 1


def test_empty_document_produces_nothing():
    assert chunk_markdown("", chunk_size=800) == []


def test_offsets_account_for_leading_indent():
    """列表项等带缩进的段落，下标要指向第一个实字符，而不是指向缩进空格。"""
    md = "# 标题\n\n## 小节\n\n- 第一条要点，讲切块的语义单元。\n- 第二条要点，讲边界不要劈开句子。\n"
    chunks = chunk_markdown(md, chunk_size=800)
    for c in chunks:
        assert c.content == md[c.start_position : c.end_position]
        assert c.content == c.content.strip()


def test_offsets_account_for_trailing_blank_lines():
    """段落之间和文档末尾的空行必须被排除在下标之外。"""
    md = "# 标题\n\n第一段，讲向量数据库的语义检索。\n\n\n\n第二段，讲余弦相似度。\n\n\n"
    chunks = chunk_markdown(md, chunk_size=800)
    for c in chunks:
        assert c.content == md[c.start_position : c.end_position]
        assert not c.content.startswith("\n")
        assert not c.content.endswith("\n")
