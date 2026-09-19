"""context 清洗层的单元测试。

这个层只处理"明确可识别的垃圾"，所以它的失败模式不是漏清，
而是把真知识内容删掉。测试里正常文本的用例刻意比垃圾用例更多，
目的就是钉死这条边界。

纯函数模块，不加载 embedding 模型，不需要 GPU。
"""
from __future__ import annotations

import pytest

from app.context_filter import (
    ContextSelection,
    judge_chunk,
    prepare_context,
    strip_export_metadata,
)
from app.retriever import SearchResult


def _hit(chunk_id: int, content: str, title: str = "测试文档", score: float = 0.9) -> SearchResult:
    """构造一条检索命中。context_filter 只读 content，其余字段填上合法值即可。"""
    return SearchResult(
        score=score,
        chunk_id=chunk_id,
        document_id=chunk_id,
        chunk_index=0,
        title=title,
        file_path=f"{title}.md",
        heading="概述",
        source="obsidian",
        content=content,
    )


# ---------------------------------------------------------------------------
# 正常文本绝不能被改动
# ---------------------------------------------------------------------------

NORMAL_TEXTS = [
    "普通中文段落。纳什均衡存在性定理由约翰·纳什在 1950 年的博士论文中"
    "提出并证明，它保证了任何有限博弈都存在至少一个混合策略纳什均衡。",
    # 单个链接的正文：link_count=1 < NAVIGATION_MIN_LINKS，不会被判成导航，
    # 也不带 metadata 尾巴，所以整体不能有任何变化。
    "详见 [角谷不动点定理](https://en.wikipedia.org/wiki/Kakutani_fixed-point_theorem)"
    " 的原始表述，它是存在性证明的核心工具。",
    # 行内公式：括号和花括号密集，容易被粗糙的"删 JSON"逻辑误伤。
    # 用 raw string，否则 \B / \s / \a 会被当转义符吃掉。
    r"最优回应对应定义为 $BR_i(\sigma_{-i}) = \arg\max_{\sigma_i \in \Delta(S_i)} u_i(\sigma_i, \sigma_{-i})$，其中 $\Sigma = \times_i \Delta(S_i)$ 是混合策略组合空间。",
    # 围栏代码块里的链接，不该计入链接占比。
    "Obsidian 的链接语法如下。\n\n"
    "```markdown\n[[目标笔记]]\n[[目标笔记|别名]]\n[外链](https://example.com)\n```\n\n"
    "写文档时优先用 wikilink，因为笔记改名后它会自动更新。",
    # 中文 key 的伪 JSON。key 不在白名单里，不该被剥。
    '字段约定："路径": "固定位置"，由系统统一分配。',
    # 正文中间的 JSON。strip 只认末尾，中间一律不动。
    '导出示例是 {"file_path": "a.md", "title": "x"}，'
    "它只是说明格式，真正的结论在后面这一段：策略集非空紧凸即可保证存在性。",
    # 只有标题没有正文。
    "## 相关链接\n",
]


# pytest.param 显式声明"每个元素是单个值"，避免 pytest 把容器类型的参数解包。
@pytest.mark.parametrize(
    "text",
    [pytest.param(t, id=f"case-{i}") for i, t in enumerate(NORMAL_TEXTS, 1)],
)
def test_normal_text_is_untouched(text: str) -> None:
    """正常文本经过清洗后正文逐字不变。

    strip_export_metadata 本身就 rstrip 尾部空白（剥 metadata 必然动末尾），
    所以比对时把两边都 rstrip 掉，只验证正文内容没被删改。
    """
    assert strip_export_metadata(text) == text.rstrip()
    assert judge_chunk(text).is_navigation is False


def test_export_tail_is_stripped() -> None:
    """真实数据里见过的导出残留形状：清掉 metadata 尾巴，正文一个字都不少。

    路径用占位符而非真实 vault 路径，避免把本机用户名和知识库目录结构带进版本库。
    """
    raw = (
        "纳什均衡存在性定理不仅是博弈论的理论基石，也是社会科学中少数几个具有"
        "普遍存在性保证的数学结果之一，深刻影响了经济学、政治学、生物学和"
        "计算机科学等多个学科。"
        '", "file_path": "/c/Users/<username>/vault/<知识库>/<章节>/<文档>.md"}'
    )
    assert strip_export_metadata(raw) == (
        "纳什均衡存在性定理不仅是博弈论的理论基石，也是社会科学中少数几个具有"
        "普遍存在性保证的数学结果之一，深刻影响了经济学、政治学、生物学和"
        "计算机科学等多个学科。"
    )


def test_multiple_metadata_pairs_are_stripped() -> None:
    """连续多对 metadata 依次剥干净。"""
    raw = '主体内容。", "title": "t", "file_path": "/x.md", "chunk": 3}'
    assert strip_export_metadata(raw) == "主体内容。"


def test_metadata_key_outside_whitelist_kept() -> None:
    """key 不在白名单里就不剥——这是避免误删的安全阀。"""
    raw = '正文结尾。", "author": "someone", "note": "important content"}'
    assert strip_export_metadata(raw) == raw


def test_pure_json_is_not_emptied() -> None:
    """整段都是 JSON 时不能把它清成空串——返回原文。"""
    raw = '{"file_path": "a.md", "title": "t"}'
    assert strip_export_metadata(raw) == raw


def test_empty_and_whitespace_input() -> None:
    """空串和纯空白是合法输入，不能抛异常。"""
    assert strip_export_metadata("") == ""
    assert strip_export_metadata("   \n  ") == ""
    assert judge_chunk("").is_navigation is False


def test_judgement_reports_metadata_flag() -> None:
    """metadata_stripped 要如实反映是否做了清理。"""
    assert judge_chunk("正常段落。").metadata_stripped is False
    assert judge_chunk('正常段落。", "file_path": "/x.md"}').metadata_stripped is True


# ---------------------------------------------------------------------------
# 导航 chunk 识别
# ---------------------------------------------------------------------------

NAVIGATION_CHUNK = (
    "## 相关链接\n\n"
    "- [[纳什均衡 (Nash Equilibrium)]]\n"
    "- [[混合策略纳什均衡]]\n"
    "- [[占优策略 (Dominant Strategy)]]\n"
    "- [[囚徒困境]]\n"
    "- [[完全信息博弈]]\n"
    "- [[重复博弈]]\n"
)


def test_pure_wikilink_list_is_navigation() -> None:
    """一串 wikilink 判为导航。"""
    judgement = judge_chunk(NAVIGATION_CHUNK)
    assert judgement.is_navigation is True
    assert judgement.link_ratio >= 0.8
    assert judgement.link_count >= 2


def test_heading_named_links_does_not_override_content() -> None:
    """heading 叫"相关链接"，但正文有实质内容——不能因为标题就删。"""
    text = (
        "## 相关链接\n\n"
        "本页列出与纳什均衡存在性定理相关的定理与概念。\n\n"
        "Kakutani 不动点定理是证明的核心工具：它断言满足上半连续、非空紧凸值"
        "条件的对应函数必存在不动点，纳什均衡正是最优反应对应的这样一个不动点。"
        "这一构造把存在性问题从策略空间的结构性质直接推导出来。\n\n"
        "- [[角谷不动点定理]]\n"
    )
    assert judge_chunk(text).is_navigation is False


def test_two_links_with_real_prose_is_not_navigation() -> None:
    """2 个链接但正文占大头，占比远低于阈值。"""
    text = (
        "## 相关定理\n\n"
        "纳什均衡存在性定理依赖三个关键的前置结果：Kakutani 不动点定理提供不动点"
        "保证，Brouwer 不动点定理是它的前身，Glicksberg 定理把结论推广到无穷维"
        "空间。其中上半连续性是连接支付函数连续性与对应函数存在性的桥梁，也是"
        "整个证明里最需要小心验证的条件。\n\n"
        "- [[角谷不动点定理]]\n"
        "- [[布劳威尔不动点定理]]\n"
    )
    judgement = judge_chunk(text)
    assert judgement.link_ratio < 0.8
    assert judgement.is_navigation is False


def test_links_only_inside_code_fence_do_not_trigger() -> None:
    """讲链接语法的文档：链接全在代码块里，不该被判成导航。"""
    text = (
        "## Markdown 链接语法\n\n"
        "Obsidian 用 `[[wikilink]]` 表示内部链接，标准 Markdown 用 `[文本](目标)`。\n\n"
        "```markdown\n[[目标笔记]]\n[[目标笔记|别名]]\n[外链](https://example.com)\n```\n\n"
        "以上写法覆盖了几乎全部跳转场景。优先用 wikilink，因为它支持别名，"
        "且在笔记重命名后会自动更新，标准 Markdown 链接做不到这一点。"
    )
    assert judge_chunk(text).is_navigation is False


# ---------------------------------------------------------------------------
# prepare_context：Top-20 -> 过滤 -> Top-5
# ---------------------------------------------------------------------------

def test_prepare_context_drops_navigation_and_refills() -> None:
    """导航 chunk 被跳过，后面相似度低一点的真内容补上来。"""
    hits = [
        _hit(1, NAVIGATION_CHUNK, score=0.90),
        _hit(2, "第一条真实内容：Kakutani 不动点定理给出存在性。", score=0.85),
        _hit(3, "第二条真实内容：上半连续性是关键条件。", score=0.80),
        _hit(4, NAVIGATION_CHUNK, score=0.75),
        _hit(5, "第三条真实内容：策略空间必须非空紧凸。", score=0.70),
    ]
    # top_k=3 才有意义：前 3 条里混着一条导航，要看到"跳过它并从后面补上第 5 条"
    result = prepare_context(hits, top_k=3)

    assert isinstance(result, ContextSelection)
    assert [c.hit.chunk_id for c in result.selected] == [2, 3, 5]
    assert [c.hit.chunk_id for c in result.dropped] == [1, 4]
    assert result.selected[0].judgement.is_navigation is False


def test_prepare_context_returns_fewer_when_not_enough() -> None:
    """不足 top_k 时保留剩余结果，不补垃圾。"""
    hits = [
        _hit(1, NAVIGATION_CHUNK, score=0.9),
        _hit(2, "唯一一条真实内容。", score=0.8),
        _hit(3, NAVIGATION_CHUNK, score=0.7),
    ]
    result = prepare_context(hits, top_k=5)
    assert [c.hit.chunk_id for c in result.selected] == [2]
    assert len(result.dropped) == 2


def test_prepare_context_stops_early_at_top_k() -> None:
    """凑满 top_k 就停，后面的候选不必判定——省一次无谓的工作。"""
    hits = [_hit(i, f"第 {i} 条真实内容，足够长的一整段说明文字。") for i in range(1, 11)]
    result = prepare_context(hits, top_k=5)
    assert len(result.selected) == 5
    assert [c.hit.chunk_id for c in result.selected] == [1, 2, 3, 4, 5]
    assert result.dropped == []


def test_prepare_context_context_content_is_sanned_but_hit_is_raw() -> None:
    """LLM 拿到的是清理后的 context_content，sources 仍引用原始 chunk。"""
    raw = '真正的内容。", "file_path": "/x.md"}'
    result = prepare_context([_hit(1, raw)], top_k=1)

    chunk = result.selected[0]
    assert chunk.context_content == "真正的内容。"
    assert chunk.hit.content == raw
