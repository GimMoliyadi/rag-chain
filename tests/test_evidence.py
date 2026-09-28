"""Evidence Gate 单元测试：JSON 解析、字段校验、失败兜底。

核心契约：
- 正常路径：LLM 输出（含脏格式）能解析成 EvidenceVerdict，字段被规整；
- 兜底路径：LLM 异常 / 输出损坏 / status 非法时按 sufficient 兜底并留
  judge_error 标记——门控绝不能成为问答链路的单点故障；
- 空检索结果直接判 insufficient，不浪费 LLM 调用。
"""
from __future__ import annotations

import pytest

from app.evidence import (
    STATUS_INSUFFICIENT,
    STATUS_SUFFICIENT,
    EvidenceGate,
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

