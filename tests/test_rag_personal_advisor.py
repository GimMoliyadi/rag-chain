"""advisor 模式（Personal Context + 知识库）行为测试。

核心契约：
- mode=normal（默认）一个字节都不读 Personal Context，Prompt 与历史版本一致；
- mode=advisor 用「用户当前情况 / 知识库资料 / 用户问题」三段式 Prompt，
  并把 Personal Context 传给 Evidence Gate；
- Personal Context 拿不到（未启用 / 无内容 / 结构写错）时必须让模型明确
  说明缺少用户的当前情况，绝不允许它顺着自身记忆把用户处境补全掉；
- mode 拼错直接报错，不静默走错分支。
"""
from __future__ import annotations

import json

import pytest

from app.evidence import EvidenceGate
from app.personal_context import PersonalContext, PersonalContextError, render_for_prompt
from app.rag import (
    MODE_ADVISOR,
    MODE_NORMAL,
    PERSONAL_CONTEXT_UNAVAILABLE_NOTE,
    RAGEngine,
)
from app.retriever import SearchResult
from fakes import ScriptedLLM

HIT = SearchResult(
    score=0.8, chunk_id=1, document_id=1, chunk_index=0,
    title="最小可行产品", file_path="创业/MVP.md", heading="定义",
    source="测试库", content="MVP 是用最少的成本验证一个核心假设的产品版本。",
)

HAS_DIRECTION = PersonalContext(loaded=True, source_path="context.yaml", profile={"current_direction": "AI 应用开发"})
# 文件在、但每个字段都留空：loaded=True 却没有任何可渲染的内容
EMPTY_CONTEXT = PersonalContext(loaded=True, profile={})


class FakeRetriever:
    def __init__(self, hits: list[SearchResult]):
        self.hits = hits
        self.calls: list[int] = []

    def search(self, question: str, top_k: int = 5, domains: list[str] | None = None) -> list[SearchResult]:
        self.calls.append(top_k)
        return list(self.hits)[:top_k]


class CountingLoader:
    """返回固定 PersonalContext 的 loader，同时记录被调用次数。"""

    def __init__(self, context: PersonalContext):
        self.context = context
        self.calls = 0

    def __call__(self) -> PersonalContext:
        self.calls += 1
        return self.context


def _build(llm, **kwargs) -> RAGEngine:
    return RAGEngine(retriever=FakeRetriever([HIT]), llm=llm, top_k=5, vault_name="测试知识库", **kwargs)


def _verdict(status: str) -> str:
    return json.dumps({
        "status": status, "confidence": 0.9, "reason": "理由",
        "supported_aspects": ["方向"], "unsupported_aspects": [],
    }, ensure_ascii=False)


def test_advisor_mode_uses_three_part_prompt_with_personal_context():
    llm = ScriptedLLM(["顾问回答"])
    loader = CountingLoader(HAS_DIRECTION)
    engine = _build(llm, personal_context_loader=loader)

    result = engine.answer("我现在应该先推哪个项目？", mode=MODE_ADVISOR)

    assert loader.calls == 1
    assert result.mode == MODE_ADVISOR
    assert result.personal_context is not None
    assert result.personal_context["loaded"] is True
    assert result.personal_context["available_sections"] == ["profile"]

    system, user = llm.calls[0][0]["content"], llm.calls[0][1]["content"]
    assert "个人顾问" in system
    assert "硬性规则" in system
    assert "绝不用你记忆中的知识补全" in system
    assert user.startswith("【用户当前情况】")
    assert "【知识库资料】" in user and "【用户问题】" in user
    assert "AI 应用开发" in user and render_for_prompt(HAS_DIRECTION) in user
    assert "MVP 是用最少的成本" in user


def test_normal_mode_never_reads_personal_context():
    """mode=normal 时 Personal Context 一个字节都不读，Prompt 保持历史写法。"""
    llm = ScriptedLLM([_verdict("sufficient"), "普通回答"])
    loader = CountingLoader(HAS_DIRECTION)
    engine = _build(
        llm,
        evidence_gate=EvidenceGate(llm),
        personal_context_loader=loader,
    )

    result = engine.answer("我目前主要方向是什么？")

    assert loader.calls == 0
    assert result.mode == MODE_NORMAL
    assert result.personal_context is None
    user = llm.calls[1][1]["content"]
    assert "【参考资料】" in user
    assert "【用户当前情况】" not in user
    assert "【用户个人当前情况】" not in llm.calls[0][1]["content"]  # Judge 也走原 Prompt
    assert "AI 应用开发" not in user
    assert "personal_context_ms" not in result.timings


def test_personal_context_is_passed_to_evidence_gate():
    """有个人状态时 Judge 也必须看得到它，否则会判成「知识库没有依据」而误拒答。"""
    llm = ScriptedLLM([_verdict("partial"), "回答"])
    engine = _build(
        llm,
        evidence_gate=EvidenceGate(llm),
        personal_context_loader=lambda: HAS_DIRECTION,
    )

    engine.answer("我该先做哪个项目？", mode=MODE_ADVISOR)

    judge_user = llm.calls[0][1]["content"]
    assert "【用户个人当前情况】" in judge_user
    assert "AI 应用开发" in judge_user


def test_personal_context_absent_shows_explicit_missing_note():
    """Personal Context 没内容时，Prompt 里必须显式说明缺失，而不是留一块空标题。"""
    llm = ScriptedLLM(["一般性分析"])
    engine = _build(llm, personal_context_loader=lambda: EMPTY_CONTEXT)

    result = engine.answer("我主要方向是什么？", mode=MODE_ADVISOR)

    assert PERSONAL_CONTEXT_UNAVAILABLE_NOTE in llm.calls[0][1]["content"]
    assert result.personal_context is not None
    assert result.personal_context["loaded"] is True
    assert result.personal_context["available_sections"] == []
    assert result.personal_context["reason"] == "Personal Context 没有内容"


def test_personal_context_not_enabled_shows_missing_note():
    llm = ScriptedLLM(["一般性分析"])
    engine = _build(llm)  # 没有 loader

    result = engine.answer("我主要方向是什么？", mode=MODE_ADVISOR)

    assert PERSONAL_CONTEXT_UNAVAILABLE_NOTE in llm.calls[0][1]["content"]
    assert result.personal_context == {"loaded": False, "reason": "Personal Context 未启用"}


def test_personal_context_schema_error_is_reported_not_fatal():
    """用户把字段写错了：问答照常，错误原因出现在 personal_context.error 里。"""
    def broken_loader() -> PersonalContext:
        raise PersonalContextError("projects[0] 缺少非空的 name 字段")

    llm = ScriptedLLM(["一般性分析"])
    engine = _build(llm, personal_context_loader=broken_loader)

    result = engine.answer("我主要方向是什么？", mode=MODE_ADVISOR)

    assert result.answer == "一般性分析"
    assert PERSONAL_CONTEXT_UNAVAILABLE_NOTE in llm.calls[0][1]["content"]
    assert result.personal_context == {
        "loaded": False, "error": "projects[0] 缺少非空的 name 字段",
    }


def test_advisor_partial_evidence_adds_advisor_addendum():
    llm = ScriptedLLM([_verdict("partial"), "分两部分回答"])
    engine = _build(
        llm,
        evidence_gate=EvidenceGate(llm),
        personal_context_loader=lambda: HAS_DIRECTION,
    )

    result = engine.answer("取舍类问题", mode=MODE_ADVISOR)

    assert result.evidence is not None and result.evidence.status == "partial"
    assert "知识库能够支持的部分" in llm.calls[1][0]["content"]
    assert "属于你的判断" in llm.calls[1][0]["content"]


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode 只能是"):
        _build(ScriptedLLM(["x"])).answer("问题？", mode="advisior")


def test_stream_done_carries_mode_and_personal_context():
    llm = ScriptedLLM(["流式回答"])
    engine = _build(llm, personal_context_loader=lambda: HAS_DIRECTION)

    events = list(engine.answer_stream("我主要方向是什么？", mode=MODE_ADVISOR))
    done = events[-1]["data"]

    assert done["mode"] == MODE_ADVISOR
    assert done["personal_context"]["loaded"] is True
    assert PERSONAL_CONTEXT_UNAVAILABLE_NOTE not in llm.calls[0][1]["content"]
    assert [event["event"] for event in events][0] == "retrieval_started"
