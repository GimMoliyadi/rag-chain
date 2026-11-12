"""Evidence Gate 单元测试：JSON 解析、字段校验、失败兜底。

核心契约：
- 正常路径：LLM 输出（含脏格式）能解析成 EvidenceVerdict，字段被规整；
- 兜底路径：LLM 异常 / 输出损坏 / status 非法时按 sufficient 兜底并留
  judge_error 标记——门控绝不能成为问答链路的单点故障；
- 空检索结果直接判 insufficient，不浪费 LLM 调用。
"""
from __future__ import annotations

import json

import pytest

from app.evidence import (
    ANSWER_STATUS_UNSUPPORTED,
    ANSWER_STATUS_UNVERIFIED,
    AnswerVerifier,
    STATUS_INSUFFICIENT,
    STATUS_SUFFICIENT,
    EvidenceGate,
    _build_answer_units,
    _claim_surface_tokens_supported,
    _extract_json,
)
from app.retriever import SearchResult
from fakes import ScriptedLLM


def _hit(chunk_id: int = 1, content: str = "纳什均衡是博弈论的核心解概念。") -> SearchResult:
    return SearchResult(
        score=0.8, chunk_id=chunk_id, document_id=1, chunk_index=0,
        title="纳什均衡", file_path="博弈论-AI知识库/纳什均衡.md",
        heading="定义", source="测试库", content=content,
    )


def _verdict_json(status: str = "sufficient", confidence: float = 0.9) -> str:
    return (
        '{"status": "' + status + '", "confidence": ' + str(confidence) + ", "
        '"reason": "资料包含定义", '
        '"supported_aspects": ["定义"], "unsupported_aspects": ["证明过程"]}'
    )


def _answer_verification_json(
    status: str = "supported",
    supported: bool = True,
    citation_indexes: list[int] | None = None,
) -> str:
    return json.dumps({
        "status": status,
        "confidence": 0.92,
        "reason": "每条主张都能在对应原文中找到依据",
        "claims": [{
            "unit_id": "u1",
            "claim": "纳什均衡是博弈论的核心解概念。",
            "is_claim": True,
            "supported": supported,
            "citation_indexes": citation_indexes if citation_indexes is not None else [1],
            "evidence_quotes": [{
                "citation_index": index,
                "quote": "纳什均衡是博弈论的核心解概念。",
            } for index in (citation_indexes if citation_indexes is not None else [1])],
            "reason": "对应资料直接给出该定义",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)


def _advisor_answer_verification_json(*claims: dict, status: str = "supported") -> str:
    normalized_claims = []
    for index, claim in enumerate(claims, start=1):
        item = dict(claim)
        item.setdefault("unit_id", f"u{index}")
        item.setdefault("is_claim", True)
        normalized_claims.append(item)
    return json.dumps({
        "status": status,
        "confidence": 0.92,
        "reason": "主张分别来自个人情况和明确标注的分析",
        "claims": normalized_claims,
        "invalid_citations": [],
    }, ensure_ascii=False)


def _citation_check(status: str = "ok", cited: list[int] | None = None) -> dict:
    cited = cited if cited is not None else [1]
    return {
        "status": status,
        "cited_indexes": cited,
        "invalid_indexes": [] if status == "ok" else [99],
        "unused_indexes": [] if status == "ok" else [1],
    }


def test_valid_verdict_parsed():
    gate = EvidenceGate(ScriptedLLM([_verdict_json("partial", 0.75)]))
    verdict = gate.judge("纳什均衡是什么？", [_hit()])

    assert verdict.status == "partial"
    assert verdict.confidence == 0.75
    assert verdict.supported_aspects == ["定义"]
    assert verdict.unsupported_aspects == ["证明过程"]
    assert verdict.debug["judge_error"] is False


def test_json_in_markdown_fence_parsed():
    wrapped = "```json\n" + _verdict_json() + "\n```"
    verdict = EvidenceGate(ScriptedLLM([wrapped])).judge("问题", [_hit()])
    assert verdict.status == STATUS_SUFFICIENT


def test_json_with_surrounding_text_parsed():
    noisy = "评审结果如下：\n" + _verdict_json() + "\n以上。"
    verdict = EvidenceGate(ScriptedLLM([noisy])).judge("问题", [_hit()])
    assert verdict.status == STATUS_SUFFICIENT


def test_invalid_status_falls_back_to_sufficient():
    verdict = EvidenceGate(ScriptedLLM([_verdict_json("totally_fine")])).judge("问题", [_hit()])
    assert verdict.status == STATUS_SUFFICIENT
    assert verdict.debug["judge_error"] is True


def test_broken_json_falls_back():
    verdict = EvidenceGate(ScriptedLLM(["这里没有 JSON，只有散文。"])).judge("问题", [_hit()])
    assert verdict.status == STATUS_SUFFICIENT
    assert verdict.debug["judge_error"] is True


def test_llm_exception_falls_back_without_raising():
    class ExplodingLLM(ScriptedLLM):
        def chat(self, messages):
            self.calls.append(messages)
            raise ConnectionError("网络断了")

    verdict = EvidenceGate(ExplodingLLM([])).judge("问题", [_hit()])
    assert verdict.status == STATUS_SUFFICIENT
    assert verdict.debug["judge_error"] is True


def test_fail_closed_turns_judge_error_into_explicit_refusal():
    """严格模式不能把评审失败当成 sufficient 放行。"""
    verdict = EvidenceGate(
        ScriptedLLM(["不是 JSON"]),
        fail_closed=True,
    ).judge("问题", [_hit()])

    assert verdict.status == STATUS_INSUFFICIENT
    assert verdict.debug["judge_error"] is True
    assert verdict.debug["strict_fail_closed"] is True
    assert "不代表知识库中不存在" in verdict.reason
    assert verdict.to_dict()["strict_fail_closed"] is True


def test_fail_closed_turns_llm_exception_into_explicit_refusal():
    class ExplodingLLM(ScriptedLLM):
        def chat(self, messages):
            self.calls.append(messages)
            raise ConnectionError("网络断了")

    verdict = EvidenceGate(ExplodingLLM([]), fail_closed=True).judge("问题", [_hit()])

    assert verdict.status == STATUS_INSUFFICIENT
    assert verdict.debug["strict_fail_closed"] is True


def test_answer_verifier_checks_each_claim_against_the_exact_context():
    llm = ScriptedLLM([_answer_verification_json()])
    verifier = AnswerVerifier(llm)
    result = verifier.verify(
        "纳什均衡是什么？",
        "纳什均衡是博弈论的核心解概念。[1]",
        "[1] 文档：纳什均衡\n内容：纳什均衡是博弈论的核心解概念。",
        _citation_check(),
    )

    assert result.passed is True
    assert result.status == "supported"
    assert result.claims[0]["citation_indexes"] == [1]
    assert "最终回答" in llm.calls[0][1]["content"]
    assert "纳什均衡是博弈论的核心解概念。[1]" in llm.calls[0][1]["content"]
    assert "[1] 文档：纳什均衡" in llm.calls[0][1]["content"]


def test_answer_units_split_english_sentences_without_splitting_decimals():
    """逐单元覆盖也适用于英文回答，但不能把小数点当成句号。"""
    units = _build_answer_units("Fact one [1]. Fact two [2]. Ratio is 3.14 [3].")

    assert [unit.text for unit in units] == [
        "Fact one [1].",
        "Fact two [2].",
        "Ratio is 3.14 [3].",
    ]
    assert [unit.citation_indexes for unit in units] == [(1,), (2,), (3,)]


def test_answer_units_keep_semicolon_clause_with_following_citation():
    """A citation after a semicolon must cover the complete sentence unit."""
    units = _build_answer_units("完美回忆要求路径一致；这保证玩家不会遗忘行动[1]。")

    assert len(units) == 1
    assert units[0].citation_indexes == (1,)


def test_answer_verifier_allows_multiple_quotes_from_one_source():
    """One source may support several clauses and therefore need two excerpts."""
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "两个短摘录均来自同一来源",
        "claims": [{
            "unit_id": "u1",
            "claim": "A 和 B。[1]",
            "is_claim": True,
            "supported": True,
            "citation_indexes": [1],
            "evidence_quotes": [
                {"citation_index": 1, "quote": "A。"},
                {"citation_index": 1, "quote": "B。"},
            ],
            "reason": "同一资料分别给出两个事实",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "A 和 B。[1]", "[1] 文档：测试\n内容：A。B。", _citation_check(),
    )

    assert result.passed is True


def test_answer_verifier_normalizes_markdown_list_and_escaped_quote_excerpts():
    """Formatting-only quote differences must not turn supported evidence into a refusal."""
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "资料逐项列出条件",
        "claims": [{
            "unit_id": "u1",
            "claim": "两个条件。[1]",
            "is_claim": True,
            "supported": True,
            "citation_indexes": [1],
            "evidence_quotes": [{
                "citation_index": 1,
                "quote": "参与人集合有限。每个参与人的纯策略集合有限。",
            }],
            "reason": "逐字对应列表内容",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "两个条件。[1]",
        "[1] 文档：测试\n内容：- 参与人集合有限。\n- 每个参与人的纯策略集合有限。",
        _citation_check(),
    )

    assert result.passed is True


@pytest.mark.parametrize(
    "source, quote",
    [
        ("玩家必须完全一致。", "玩家完全一致。"),
        (r"若 $x, x' \in I$。", "若 x, x' 属于 I。"),
    ],
)
def test_answer_verifier_does_not_accept_semantic_quote_rewrites(source, quote):
    """Only presentation normalization is allowed; meaning-changing edits fail."""
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "模型误把改写当成原文",
        "claims": [{
            "unit_id": "u1",
            "claim": "事实。[1]",
            "is_claim": True,
            "supported": True,
            "citation_indexes": [1],
            "evidence_quotes": [{"citation_index": 1, "quote": quote}],
            "reason": "不应通过",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "事实。[1]", f"[1] 文档：测试\n内容：{source}", _citation_check(),
    )

    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED


def test_answer_verifier_rejects_salient_identifier_absent_from_cited_source():
    """A model verdict cannot smuggle in an uncited chip/model name."""
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "模型误判为支持",
        "claims": [{
            "unit_id": "u1",
            "claim": "2019 年分解 35，并使用 IBM 芯片[1]。",
            "is_claim": True,
            "supported": True,
            "citation_indexes": [1],
            "evidence_quotes": [{"citation_index": 1, "quote": "2019 年分解 35"}],
            "reason": "原文只出现了年份和数字",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "2019 年分解 35，并使用 IBM 芯片[1]。",
        "[1] 文档：实验\n内容：2019 年分解 35。", _citation_check(),
    )

    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED


def test_answer_verifier_rejects_identifier_attached_to_the_wrong_year():
    """Presence elsewhere in a source cannot support a new year/identifier pair."""
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "模型误把跨行事实拼成一条",
        "claims": [{
            "unit_id": "u1",
            "claim": "2019：分解 35，并使用 IBM[1]。",
            "is_claim": True,
            "supported": True,
            "citation_indexes": [1],
            "evidence_quotes": [{"citation_index": 1, "quote": "2019：分解 35"}],
            "reason": "IBM 只出现在另一年份",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "2019：分解 35，并使用 IBM[1]。",
        "[1] 文档：实验\n内容：- 2012：分解 15（IBM/NIST）\n- 2019：分解 35（超导）",
        _citation_check(),
    )

    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED


def test_surface_check_allows_identifier_on_a_wrapped_markdown_item():
    """A physical line wrap must not turn one supported bullet into a refusal."""
    source = "- 2019：分解 35（\n  IBM/NIST 设备）"
    assert _claim_surface_tokens_supported(
        "2019：分解 35，并使用 IBM[1]。", [source],
    ) is True


def test_surface_check_keeps_separate_bullets_isolated_after_block_grouping():
    """The relaxed wrapping rule must not join two separate factual bullets."""
    source = "- 2012：分解 15（IBM/NIST）\n- 2019：分解 35（超导）"
    assert _claim_surface_tokens_supported(
        "2019：分解 35，并使用 IBM[1]。", [source],
    ) is False


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        ("2019 年分解 35[1]。", "2019 年分解 135。"),
        ("2019 年使用 IBM[1]。", "2019 年使用 IBMXYZ。"),
        (r"公式含有 \\iota[1]。", r"公式含有 \\iotaX。"),
        ("比例为 3.14[1]。", "比例为 3.140。"),
    ],
)
def test_surface_check_does_not_match_a_larger_ascii_token(answer, source):
    """Numbers, identifiers and LaTeX names need ASCII token boundaries."""
    assert _claim_surface_tokens_supported(answer, [source]) is False


def test_surface_check_preserves_word_boundaries_in_natural_prose():
    """Display normalization must not concatenate neighboring English words."""
    assert _claim_surface_tokens_supported(
        "2019 年使用 IBM 芯片[1]。",
        ["2019 年使用 device IBM chip。"],
    ) is True
    assert _claim_surface_tokens_supported(
        "2019 年使用 IBM 芯片[1]。",
        ["2019 年使用 deviceIBMchip。"],
    ) is False


def test_surface_check_keeps_chinese_adjacent_tokens_valid():
    assert _claim_surface_tokens_supported(
        "2019 年分解 35 号，并使用 IBM 芯片[1]。",
        ["2019 年分解 35号，并使用 IBM芯片。"],
    ) is True


@pytest.mark.parametrize(
    ("answer", "source", "quote"),
    [
        ("2019 年分解 35[1]。", "2019 年分解 135。", "2019 年分解 135"),
        ("2019 年使用 IBM[1]。", "2019 年使用 IBMXYZ。", "2019 年使用 IBMXYZ"),
    ],
)
def test_answer_verifier_rejects_substring_surface_matches(answer, source, quote):
    """The deterministic surface guard must apply at the verifier seam."""
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "模型误把更大 token 当成相同事实",
        "claims": [{
            "unit_id": "u1",
            "claim": answer,
            "is_claim": True,
            "supported": True,
            "citation_indexes": [1],
            "evidence_quotes": [{"citation_index": 1, "quote": quote}],
            "reason": "表面 token 并不相等",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", answer, f"[1] 文档：测试\n内容：{source}", _citation_check(),
    )
    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED


def test_surface_check_accepts_a_coefficient_glued_to_its_variable():
    """``3n`` in the note must support ``3n`` in the answer.

    The digit run only has to be a *number*, not a standalone word: a following
    letter is multiplication.  Before this, a formula copied verbatim from the
    note (``F = 3n − 2P_L − P_H``) failed the numeric containment check and a
    correctly cited answer was rejected as if it had invented numbers.
    """
    source = "$$F = 3n - 2P_L - P_H$$"
    assert _claim_surface_tokens_supported("**F = 3n − 2P_L − P_H**[1]", [source]) is True
    assert _claim_surface_tokens_supported("F = 3×3−2×4=1[1]", ["F = 3n − 2P_L − P_H"]) is False


def test_surface_check_still_rejects_a_number_merged_into_a_longer_digit_run():
    """Relaxing the trailing boundary must not re-open ``35`` matching ``135``."""
    assert _claim_surface_tokens_supported("分解 35[1]", ["分解 135"]) is False
    assert _claim_surface_tokens_supported("分解 35[1]", ["分解 350 次"]) is False
    assert _claim_surface_tokens_supported("取值 3.14[1]", ["取值 3.140"]) is False
    # A coefficient is the same number whether or not the variable follows it,
    # so ``3`` legitimately occurs in ``3n``.
    assert _claim_surface_tokens_supported("取值 3[1]", ["取值 3n"]) is True


def test_answer_verifier_reattributes_a_mislabeled_quote_within_the_same_unit():
    """A clerical wrong index must not turn verbatim in-context text into a refusal.

    The verifier here quotes source [3] but labels it [2].  Both numbers are
    cited by the same unit, the excerpt is verbatim from [3], and no outside
    text is introduced, so the quote is re-attributed instead of rejected.
    """
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "摘录真实存在，只是编号标错",
        "claims": [{
            "unit_id": "u1",
            "claim": "信息集把无法区分的决策节点放在一起[2][3]。",
            "is_claim": True,
            "supported": True,
            "citation_indexes": [2, 3],
            "evidence_quotes": [
                {"citation_index": 2, "quote": "一个信息集把某玩家无法区分的决策节点放在一起"},
                {"citation_index": 3, "quote": "所有节点属于同一个玩家"},
            ],
            "reason": "两处摘录分别真实来自 [3] 和 [2]",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    context = (
        "[2] 文档：信息集 / 定义\n     内容：所有节点属于同一个玩家\n\n"
        "[3] 文档：扩展式博弈 / 形式化定义\n     内容：一个信息集把某玩家无法区分的决策节点放在一起。"
    )
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "信息集把无法区分的决策节点放在一起[2][3]。", context,
        _citation_check(cited=[2, 3]),
    )

    assert result.passed is True
    assert result.debug["relabeled_quote_count"] == 2


def test_answer_verifier_still_rejects_a_quote_from_an_uncited_source():
    """Re-attribution is scoped to the unit's own citations, never the whole context."""
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "模型拿别处原文冒充本单元依据",
        "claims": [{
            "unit_id": "u1",
            "claim": "2019 年分解 35[1]。",
            "is_claim": True,
            "supported": True,
            "citation_indexes": [1],
            "evidence_quotes": [{"citation_index": 1, "quote": "一个信息集把某玩家无法区分的决策节点放在一起"}],
            "reason": "这段原文不在 [1] 里",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    context = (
        "[1] 文档：实验\n     内容：2019 年分解 35。\n\n"
        "[2] 文档：信息集\n     内容：一个信息集把某玩家无法区分的决策节点放在一起。"
    )
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "2019 年分解 35[1]。", context, _citation_check(),
    )

    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED


def test_answer_verifier_accepts_latex_rendered_glyphs_from_a_note():
    """LaTeX commands and math minus signs are presentation, not wording.

    The note stores ``\\in`` / ``\\iota``; the answer renders ``∈`` / ``ι``.
    Rejecting that turned a verbatim excerpt into an "invented" quote.
    """
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "渲染后的符号与原文同一事实",
        "claims": [{
            "unit_id": "u1",
            "claim": "同一信息集中的节点满足 ι(x) = ι(x′)[1]。",
            "is_claim": True,
            "supported": True,
            "citation_indexes": [1],
            "evidence_quotes": [{
                "citation_index": 1,
                "quote": "若 x, x' ∈ I（同一信息集），则 ι(x) = ι(x')",
            }],
            "reason": "LaTeX 渲染差异",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    context = (
        "[1] 文档：信息集\n     内容：若 $x, x' \\in I$（同一信息集），则 $\\iota(x) = \\iota(x')$"
    )
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "同一信息集中的节点满足 ι(x) = ι(x′)[1]。", context, _citation_check(),
    )

    assert result.passed is True


def test_answer_verifier_still_rejects_a_latex_semantic_rewrite():
    """Folding presentation glyphs must not accept a changed mathematical relation."""
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "模型把属于关系改成不等",
        "claims": [{
            "unit_id": "u1",
            "claim": "节点不在信息集中[1]。",
            "is_claim": True,
            "supported": True,
            "citation_indexes": [1],
            "evidence_quotes": [{"citation_index": 1, "quote": "若 x ∉ I"}],
            "reason": "摘录与原文关系不同",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    context = "[1] 文档：信息集\n     内容：若 $x \\in I$"
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "节点不在信息集中[1]。", context, _citation_check(),
    )

    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED


def test_answer_verifier_accepts_a_verbatim_uncited_formula_line():
    """A standalone formula may carry its number on the neighbouring prose."""
    context = (
        "[1] 文档：机械设计\n     内容：$$F = 3n - 2P_L - P_H$$\n\n"
        "n = 活动构件数（**不含机架**）"
    )
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "公式逐字来自 [1]",
        "claims": [
            {
                "unit_id": "u1",
                "claim": "$$F = 3n - 2P_L - P_H$$",
                "is_claim": True,
                "supported": True,
                "citation_indexes": [1],
                "evidence_quotes": [{"citation_index": 1, "quote": "$$F = 3n - 2P_L - P_H$$"}],
                "reason": "逐字来自 [1]",
            },
            {
                "unit_id": "u2",
                "claim": "n = 活动构件数（不含机架）[1]",
                "is_claim": True,
                "supported": True,
                "citation_indexes": [1],
                "evidence_quotes": [{"citation_index": 1, "quote": "n = 活动构件数（不含机架）"}],
                "reason": "定义来自 [1]",
            },
        ],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题",
        "$$F = 3n - 2P_L - P_H$$\nn = 活动构件数（不含机架）[1]",
        context, _citation_check(cited=[1]),
    )

    assert result.passed is True


def test_answer_verifier_rejects_an_invented_uncited_formula_line():
    """The uncited line allowance must not become a hole for changed numbers."""
    context = "[1] 文档：机械设计\n     内容：$$F = 3n - 2P_L - P_H$$"
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "模型改写了系数",
        "claims": [
            {
                "unit_id": "u1",
                "claim": "$$F = 4n - 2P_L - P_H$$",
                "is_claim": True,
                "supported": True,
                "citation_indexes": [1],
                "evidence_quotes": [{"citation_index": 1, "quote": "$$F = 3n - 2P_L - P_H$$"}],
                "reason": "声称公式来自 [1]",
            },
            {
                "unit_id": "u2",
                "claim": "结论如下[1]。",
                "is_claim": False,
                "supported": True,
                "citation_indexes": [],
                "evidence_quotes": [],
                "reason": "过渡语",
            },
        ],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "$$F = 4n - 2P_L - P_H$$\n结论如下[1]。",
        context, _citation_check(cited=[1]),
    )

    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED


def test_answer_verifier_accepts_a_pure_lead_in_line():
    """``其中：`` asserts nothing; rejecting it failed otherwise correct answers."""
    context = "[1] 文档：机械设计\n     内容：n = 活动构件数（不含机架）\nP_L = 低副数\nP_H = 高副数"
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "定义逐条对应",
        "claims": [
            {
                "unit_id": "u1",
                "claim": "其中：",
                "is_claim": True,
                "supported": True,
                "citation_indexes": [],
                "evidence_quotes": [],
                "reason": "引导语",
            },
            {
                "unit_id": "u2",
                "claim": "n = 活动构件数（不含机架）[1]",
                "is_claim": True,
                "supported": True,
                "citation_indexes": [1],
                "evidence_quotes": [{"citation_index": 1, "quote": "n = 活动构件数（不含机架）"}],
                "reason": "逐字来自 [1]",
            },
        ],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "其中：\nn = 活动构件数（不含机架）[1]", context, _citation_check(),
    )

    assert result.passed is True


def test_answer_verifier_rejects_a_lead_in_that_states_a_fact():
    """The lead-in allowance must not cover a line that already asserts content."""
    context = "[1] 文档：实验\n     内容：2019 年分解 35（超导）"
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "模型把事实写成无引用行",
        "claims": [{
            "unit_id": "u1",
            "claim": "2019 年分解 99：",
            "is_claim": True,
            "supported": True,
            "citation_indexes": [],
            "evidence_quotes": [],
            "reason": "无引用",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "2019 年分解 99：", context, _citation_check(),
    )

    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED


def test_answer_verifier_rejects_an_invented_section_heading():
    """A heading whose wording never appears in the evidence is still refused."""
    context = "[1] 文档：机械设计\n     内容：n = 活动构件数"
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "模型编了一个小标题",
        "claims": [{
            "unit_id": "u1",
            "claim": "拓扑优化结论：",
            "is_claim": True,
            "supported": True,
            "citation_indexes": [],
            "evidence_quotes": [],
            "reason": "资料没有这段话",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "拓扑优化结论：", context, _citation_check(),
    )

    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED


def test_answer_verifier_accepts_structural_lead_ins_and_list_counters():
    """Colons, ordinals and bold headings structure an answer; they assert nothing.

    Each lead-in is judged against the same evidence the answer cites: a
    paraphrase built from attested words passes, an invented heading and a
    digit-bearing assertion still fail.
    """
    context = (
        "[1] 文档：机械设计\n     内容：n = 活动构件数（不含机架）\n"
        "[2] 文档：机械设计 / 例1：颚式破碎机示例验证\n     内容：例1 示例：验证活动构件 n = 3"
    )
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "结构行无事实内容",
        "claims": [
            {"unit_id": "u1", "claim": "其中：", "is_claim": True, "supported": True,
             "citation_indexes": [], "evidence_quotes": [], "reason": "引导语"},
            {"unit_id": "u2", "claim": "1.", "is_claim": True, "supported": True,
             "citation_indexes": [], "evidence_quotes": [], "reason": "序号"},
            {"unit_id": "u3", "claim": "**示例验证**：", "is_claim": True, "supported": True,
             "citation_indexes": [], "evidence_quotes": [], "reason": "小标题"},
            {"unit_id": "u4", "claim": "计算活动构件数 n 的方法：", "is_claim": True, "supported": True,
             "citation_indexes": [], "evidence_quotes": [], "reason": "引导语，用词来自资料"},
            {"unit_id": "u5", "claim": "n = 活动构件数（不含机架）[1]", "is_claim": True,
             "supported": True, "citation_indexes": [1],
             "evidence_quotes": [{"citation_index": 1, "quote": "n = 活动构件数（不含机架）"}],
             "reason": "逐字来自 [1]"},
        ],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题",
        "其中：\n1.\n**示例验证**：\n计算活动构件数 n 的方法：\nn = 活动构件数（不含机架）[1]",
        context, _citation_check(cited=[1]),
    )

    assert result.passed is True


def test_extract_json_repairs_bare_backslash_from_note_quotes():
    """A model quoting the note's ``\\'`` often emits invalid JSON.

    The repair must double the bare backslash instead of losing the verdict —
    an unverified verdict refuses a fully supported answer.
    """
    broken = (
        '{"status": "supported", "confidence": 1.0, "reason": "r", '
        '"claims": [{"unit_id": "u1", "claim": "c", "is_claim": true, "supported": true, '
        '"citation_indexes": [1], '
        '"reason": "原文为 ' + chr(92) + "' 从 x 出发\"}], \"invalid_citations\": []}"
    )
    obj = _extract_json(broken)

    assert obj["status"] == "supported"
    assert obj["claims"][0]["reason"].endswith("从 x 出发")


def _verification_payload(unit_id: str, claim: str, quote: str, *, index: int = 1,
                          supported: bool = True, status: str = "supported") -> str:
    """Build one verifier JSON response with a single knowledge claim."""
    return json.dumps({
        "status": status,
        "confidence": 1.0,
        "reason": "单条主张",
        "claims": [{
            "unit_id": unit_id,
            "claim": claim,
            "is_claim": True,
            "supported": supported,
            "citation_indexes": [index],
            "evidence_quotes": [{"citation_index": index, "quote": quote}],
            "reason": "摘录核对",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)


def test_answer_verifier_retries_once_when_only_the_quote_was_miscopied():
    """A copied-over excerpt is a reviewer defect, so one bounded retry is allowed."""
    llm = ScriptedLLM([
        _verification_payload("u1", "信息集定义[1]。", "资料里没有这段拼接口"),
        _verification_payload("u1", "信息集定义[1]。", "信息集是博弈树中一组决策节点的集合"),
    ])
    result = AnswerVerifier(llm).verify(
        "问题", "信息集定义[1]。",
        "[1] 文档：信息集\n内容：信息集是博弈树中一组决策节点的集合，这些节点具有以下性质。",
        _citation_check(),
    )

    assert result.passed is True
    assert result.debug["quote_retry_used"] is True
    assert len(llm.calls) == 2


def test_answer_verifier_does_not_retry_a_semantic_rejection():
    """The model saying "not supported" is a judgement, not a copying slip."""
    llm = ScriptedLLM([
        _verification_payload("u1", "2019 年分解 99[1]。", "2019 年分解 35",
                              supported=False, status="unsupported"),
    ])
    result = AnswerVerifier(llm).verify(
        "问题", "2019 年分解 99[1]。",
        "[1] 文档：实验\n内容：2019 年分解 35（超导）。",
        _citation_check(),
    )

    assert result.passed is False
    assert result.debug.get("quote_retry_used") is None
    assert len(llm.calls) == 1


def test_answer_verifier_does_not_retry_when_the_answer_invented_an_identifier():
    """An unlocatable quote must not launder a fabricated number."""
    llm = ScriptedLLM([
        _verification_payload("u1", "2019 年分解 99[1]。", "2019 年分解 99"),
    ])
    result = AnswerVerifier(llm).verify(
        "问题", "2019 年分解 99[1]。",
        "[1] 文档：实验\n内容：2019 年分解 35（超导）。",
        _citation_check(),
    )

    assert result.passed is False
    assert len(llm.calls) == 1


def test_answer_verifier_stays_strict_when_the_retry_also_fails():
    """A failed retry keeps the refusal; the original answer is never released."""
    llm = ScriptedLLM([
        _verification_payload("u1", "信息集定义[1]。", "不存在的一段"),
        _verification_payload("u1", "信息集定义[1]。", "仍然不存在的一段"),
    ])
    result = AnswerVerifier(llm).verify(
        "问题", "信息集定义[1]。",
        "[1] 文档：信息集\n内容：信息集是博弈树中一组决策节点的集合。",
        _citation_check(),
    )

    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED
    assert result.debug["quote_retry_used"] is True
    assert result.debug["quote_retry_status"] == "unsupported"
    assert "不存在的一段" not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_answer_verifier_rejects_an_unsupported_claim():
    verifier = AnswerVerifier(ScriptedLLM([_answer_verification_json(supported=False)]))
    result = verifier.verify(
        "问题", "回答。[1]", "[1] 内容", _citation_check(),
    )

    assert result.status == ANSWER_STATUS_UNSUPPORTED
    assert result.passed is False
    assert result.unsupported_claims


def test_answer_verifier_rejects_missing_or_invalid_citations_without_llm_call():
    llm = ScriptedLLM([])
    result = AnswerVerifier(llm).verify(
        "问题", "没有引用的回答", "[1] 内容", _citation_check("missing", []),
    )

    assert result.status == ANSWER_STATUS_UNSUPPORTED
    assert result.debug["citation_error"] is True
    assert llm.calls == []


def test_answer_verifier_advisor_accepts_personal_context_and_explicit_analysis_without_citations():
    llm = ScriptedLLM([_advisor_answer_verification_json(
        {
            "claim": "用户当前方向是 AI 应用开发",
            "basis": "personal_context",
            "supported": True,
            "citation_indexes": [],
            "personal_context_quote": "当前方向：AI 应用开发",
            "reason": "个人情况中明确写出",
        },
        {
            "claim": "建议先整理 RAG 的检索评估",
            "basis": "analysis",
            "supported": True,
            "citation_indexes": [],
            "personal_context_quote": "当前方向：AI 应用开发",
            "reason": "回答明确标注为建议，基于当前方向",
        },
    )])
    missing_citations = _citation_check("missing", [])
    # 实际 check_citations() 在“缺引用”时 invalid_indexes 为空；测试 helper
    # 的非 ok 默认值是 [99]，这里只覆盖成真实的 advisor 输入。
    missing_citations["invalid_indexes"] = []
    result = AnswerVerifier(llm).verify(
        "我下一步做什么？",
        "用户当前方向是 AI 应用开发。\n我的建议：先整理 RAG 的检索评估。",
        "[1] 文档：MVP\n内容：用最少成本验证假设。",
        missing_citations,
        advisor_mode=True,
        personal_context="【个人档案】\n- 当前方向：AI 应用开发",
        personal_context_available=True,
    )

    assert result.passed is True
    assert [claim["basis"] for claim in result.claims] == [
        "personal_context", "analysis",
    ]
    verifier_user = llm.calls[0][1]["content"]
    assert "【用户当前情况】" in verifier_user
    assert "当前方向：AI 应用开发" in verifier_user


def test_answer_verifier_advisor_rejects_uncited_knowledge_claim_without_citation():
    raw = _advisor_answer_verification_json({
        "claim": "MVP 一定能提高成功率",
        "basis": "knowledge",
        "supported": True,
        "citation_indexes": [],
        "reason": "没有引用",
    })
    missing_citations = _citation_check("missing", [])
    missing_citations["invalid_indexes"] = []
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "怎么做？",
        "MVP 一定能提高成功率。",
        "[1] 文档：MVP\n内容：用最少成本验证假设。",
        missing_citations,
        advisor_mode=True,
        personal_context="【个人档案】\n- 当前方向：AI 应用开发",
        personal_context_available=True,
    )

    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED
    assert result.unsupported_claims == ["MVP 一定能提高成功率。"]


def test_answer_verifier_rejects_when_a_text_unit_has_no_judgment():
    """逐项核验不能因为模型漏列第二个文本单元就放行。"""
    answer = "支持的第一句。[1]\n第二句没有依据。"
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "只列出了第一句",
        "claims": [{
            "unit_id": "u1",
            "claim": "支持的第一句。",
            "supported": True,
            "citation_indexes": [1],
            "evidence_quotes": [{"citation_index": 1, "quote": "支持的第一句。"}],
            "reason": "原文直接包含",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", answer, "[1] 文档：测试\n内容：支持的第一句。", _citation_check(),
    )

    assert result.passed is False
    assert result.debug["strict_fail_closed"] is True


def test_answer_verifier_requires_exact_quote_from_the_cited_source():
    """有效编号不等于有效证据：摘录必须能在对应原文中定位。"""
    raw = json.dumps({
        "status": "supported",
        "confidence": 1.0,
        "reason": "引用了一个不存在的摘录",
        "claims": [{
            "unit_id": "u1",
            "claim": "支持的第一句。",
            "supported": True,
            "citation_indexes": [1],
            "evidence_quotes": [{"citation_index": 1, "quote": "原文没有这句话"}],
            "reason": "错误摘录",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "支持的第一句。[1]", "[1] 文档：测试\n内容：真实原文。", _citation_check(),
    )

    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED


def test_advisor_personal_fact_requires_a_quote_from_the_same_context():
    raw = _advisor_answer_verification_json({
        "unit_id": "u1",
        "claim": "用户有三名协作者",
        "basis": "personal_context",
        "supported": True,
        "citation_indexes": [],
        "reason": "模型声称个人情况支持，但没有摘录",
    })
    missing_citations = _citation_check("missing", [])
    missing_citations["invalid_indexes"] = []
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "团队情况？",
        "用户有三名协作者。",
        "[1] 文档：无关\n内容：无关内容。",
        missing_citations,
        advisor_mode=True,
        personal_context="【个人档案】\n- 当前方向：AI 应用开发",
        personal_context_available=True,
    )

    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED


def test_advisor_analysis_marker_is_checked_per_text_unit():
    answer = "我的建议：先整理 RAG。\n项目已经完成上线。"
    raw = _advisor_answer_verification_json(
        {
            "unit_id": "u1",
            "claim": "我的建议：先整理 RAG。",
            "basis": "analysis",
            "supported": True,
            "citation_indexes": [],
            "personal_context_quote": "当前方向：AI 应用开发",
            "reason": "第一单元有本地建议标记",
        },
        {
            "unit_id": "u2",
            "claim": "项目已经完成上线。",
            "basis": "analysis",
            "supported": True,
            "citation_indexes": [],
            "personal_context_quote": "当前方向：AI 应用开发",
            "reason": "第二单元借用了整篇回答的建议标记",
        },
    )
    missing_citations = _citation_check("missing", [])
    missing_citations["invalid_indexes"] = []
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "下一步？",
        answer,
        "[1] 文档：无关\n内容：无关内容。",
        missing_citations,
        advisor_mode=True,
        personal_context="【个人档案】\n- 当前方向：AI 应用开发",
        personal_context_available=True,
    )

    assert result.passed is False
    assert result.status == ANSWER_STATUS_UNSUPPORTED


def test_answer_verification_public_dict_does_not_leak_rejected_claim_text():
    secret_claim = "这是不应通过 SSE 暴露的未核验用户事实"
    result = AnswerVerifier(ScriptedLLM([_answer_verification_json(supported=False)])).verify(
        "问题", "回答。[1]", "[1] 内容", _citation_check(),
    )
    # 用更明确的 claim/reason 覆盖解析后的进程内诊断，模拟模型复述敏感事实。
    result.claims[0]["claim"] = secret_claim
    result.unsupported_claims[:] = [secret_claim]
    result.reason = f"不支持：{secret_claim}"

    public = result.to_dict()
    encoded = json.dumps(public, ensure_ascii=False)
    assert "claims" not in public
    assert "unsupported_claims" not in public
    assert secret_claim not in encoded
    assert public["claim_count"] == 1
    assert public["unsupported_claim_count"] == 1
    assert public["status"] == ANSWER_STATUS_UNSUPPORTED


@pytest.mark.parametrize("raw", ["不是 JSON", '{"status":"supported"}'])
def test_answer_verifier_fails_closed_on_call_or_schema_failure(raw):
    result = AnswerVerifier(ScriptedLLM([raw])).verify(
        "问题", "回答。[1]", "[1] 内容", _citation_check(),
    )

    assert result.status == ANSWER_STATUS_UNVERIFIED
    assert result.passed is False
    assert result.debug["strict_fail_closed"] is True


def test_empty_hits_is_insufficient_without_llm_call():
    gate = EvidenceGate(ScriptedLLM([]))  # 空脚本：任何 LLM 调用都会让测试失败
    verdict = gate.judge("问题", [])

    assert verdict.status == STATUS_INSUFFICIENT
    assert gate._llm.calls == []


def test_confidence_out_of_range_clamped():
    verdict = EvidenceGate(ScriptedLLM([_verdict_json("sufficient", 87)])).judge("问题", [_hit()])
    assert verdict.confidence == 1.0


def test_non_numeric_confidence_defaults_to_zero():
    raw = '{"status": "sufficient", "confidence": "很高", "reason": "", ' \
          '"supported_aspects": [], "unsupported_aspects": []}'
    verdict = EvidenceGate(ScriptedLLM([raw])).judge("问题", [_hit()])
    assert verdict.confidence == 0.0


def test_string_aspect_normalized_to_list():
    raw = '{"status": "partial", "confidence": 0.5, "reason": "r", ' \
          '"supported_aspects": "定义", "unsupported_aspects": ["", "证明"]}'
    verdict = EvidenceGate(ScriptedLLM([raw])).judge("问题", [_hit()])
    assert verdict.supported_aspects == ["定义"]
    assert verdict.unsupported_aspects == ["证明"]


def test_judge_prompt_forbids_answering_the_question():
    """Judge Prompt 的关键约束：只评审、不回答、不用自身知识。"""
    llm = ScriptedLLM([_verdict_json()])
    EvidenceGate(llm).judge("纳什均衡是什么？", [_hit()])

    system = llm.calls[0][0]["content"]
    assert "不回答问题本身" in system
    assert "不要使用你自己记忆中的任何知识" in system
    user = llm.calls[0][1]["content"]
    assert "纳什均衡是什么？" in user
    assert "纳什均衡是博弈论的核心解概念" in user


@pytest.mark.parametrize("chunk_chars", [0, -1])
def test_invalid_chunk_chars_rejected(chunk_chars):
    with pytest.raises(ValueError):
        EvidenceGate(ScriptedLLM([]), chunk_chars=chunk_chars)


# ---- Personal Context 参与评审时的行为 ----

PERSONAL_CONTEXT_TEXT = (
    "【个人档案】\n- 当前方向：AI 应用开发\n\n"
    "【项目】\n1. Obsidian RAG Personal Advisor\n   - 状态：进行中"
)


def _json_token(value: str) -> str:
    """JSON 里 true/false 不加引号，其它取值（"yes"/"支持"）必须加引号才是合法 JSON。"""
    return value if value in ("true", "false") else f'"{value}"'


def _verdict_json_with_support(status="insufficient", knowledge="false", personal="true") -> str:
    return (
        '{"status": "' + status + '", "confidence": 0.9, "reason": "理由", '
        '"supported_aspects": ["方向"], "unsupported_aspects": ["取舍依据"], '
        '"knowledge_support": ' + _json_token(knowledge)
        + ', "personal_context_support": ' + _json_token(personal) + '}'
    )


def test_no_personal_context_keeps_base_prompt_and_empty_flags():
    llm = ScriptedLLM([_verdict_json()])
    verdict = EvidenceGate(llm).judge("问题", [_hit()])

    user = llm.calls[0][1]["content"]
    assert "【用户个人当前情况】" not in user
    assert verdict.knowledge_support is None
    assert verdict.personal_context_support is None
    assert verdict.personal_context_available is False


def test_personal_context_adds_section_and_support_flags():
    llm = ScriptedLLM([_verdict_json_with_support("partial")])
    verdict = EvidenceGate(llm).judge("我该先做哪个项目？", [_hit()], personal_context=PERSONAL_CONTEXT_TEXT)

    system = llm.calls[0][0]["content"]
    user = llm.calls[0][1]["content"]
    assert "当前情况」" in system                     # 附加条款 7
    assert "status 判 partial" in system             # 附加条款 8
    assert "knowledge_support" in system
    assert "【用户个人当前情况】" in user
    assert "AI 应用开发" in user
    assert "【用户问题】" in user and "【检索到的参考资料】" in user
    assert verdict.knowledge_support is False
    assert verdict.personal_context_support is True
    assert verdict.personal_context_available is True


def test_support_flags_missing_in_output_are_none():
    verdict = EvidenceGate(
        ScriptedLLM([_verdict_json()]),  # 没有 support 字段
    ).judge("问题", [_hit()], personal_context=PERSONAL_CONTEXT_TEXT)

    assert verdict.knowledge_support is None
    assert verdict.personal_context_support is None


def test_support_flags_accept_string_values():
    verdict = EvidenceGate(
        ScriptedLLM([_verdict_json_with_support(knowledge="yes", personal="no")]),
    ).judge("问题", [_hit()], personal_context=PERSONAL_CONTEXT_TEXT)

    assert verdict.knowledge_support is True
    assert verdict.personal_context_support is False


def test_empty_hits_with_personal_context_still_calls_llm():
    """知识库检索为空 + 有个人状态时不能直接判 insufficient：
    个人状态可能正好够用，必须让 Judge 看到「检索过但一无所获」再判断。"""
    llm = ScriptedLLM([_verdict_json_with_support("partial")])
    verdict = EvidenceGate(llm).judge("我主要方向是什么？", [], personal_context=PERSONAL_CONTEXT_TEXT)

    assert llm.calls == [llm.calls[0]]
    assert "本次检索没有返回任何资料" in llm.calls[0][1]["content"]
    assert verdict.status == "partial"


def test_insufficient_downgraded_to_partial_when_personal_context_supports():
    """确定性规则：知识库没有依据但个人状态够用时，不判 insufficient（否则会直接拒答）。"""
    verdict = EvidenceGate(
        ScriptedLLM([_verdict_json_with_support("insufficient", "false", "true")]),
    ).judge("我该先收尾哪个项目？", [_hit()], personal_context=PERSONAL_CONTEXT_TEXT)

    assert verdict.status == "partial"
    assert verdict.debug["status_adjusted"] is True


def test_insufficient_not_downgraded_when_personal_context_also_insufficient():
    verdict = EvidenceGate(
        ScriptedLLM([_verdict_json_with_support("insufficient", "false", "false")]),
    ).judge("我该先收尾哪个项目？", [_hit()], personal_context=PERSONAL_CONTEXT_TEXT)

    assert verdict.status == STATUS_INSUFFICIENT
    assert "status_adjusted" not in verdict.debug


def test_insufficient_not_downgraded_without_personal_context():
    """没有个人状态时同样的字段组合不触发改判。"""
    verdict = EvidenceGate(ScriptedLLM([_verdict_json_with_support("insufficient")])).judge(
        "问题", [_hit()],
    )

    assert verdict.status == STATUS_INSUFFICIENT
    assert verdict.knowledge_support is None


def test_fallback_verdict_keeps_support_flags_empty():
    verdict = EvidenceGate(ScriptedLLM(["没有 JSON"])).judge(
        "问题", [_hit()], personal_context=PERSONAL_CONTEXT_TEXT,
    )
    assert verdict.status == STATUS_SUFFICIENT
    assert verdict.knowledge_support is None
    assert verdict.personal_context_support is None


def test_to_dict_exposes_support_fields():
    verdict = EvidenceGate(
        ScriptedLLM([_verdict_json_with_support("partial")]),
    ).judge("问题", [_hit()], personal_context=PERSONAL_CONTEXT_TEXT)

    as_dict = verdict.to_dict()
    assert as_dict["knowledge_support"] is False
    assert as_dict["personal_context_support"] is True
    assert as_dict["personal_context_available"] is True
    assert as_dict["status_adjusted"] is False

