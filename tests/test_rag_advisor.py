"""RAG 编排接入 Router / Evidence Gate 后的行为测试。

核心契约：
- 开关全关时（默认），answer() 的对外行为与历史版本一致：不路由、不评审、
  Prompt 不变，只是 RAGAnswer 多了 timings 这类空值过程字段；
- Router 启用时：规则命中不调 LLM、手动 domains 跳过路由；
- Evidence Gate 启用时：insufficient 拒答且不调生成 LLM、partial 换用
  带区分要求的 Prompt、Judge 失败兜底不阻断；
- 流式链路的事件顺序与非流式行为一致。

用 conftest 的离线小库（local embedding + 两篇文档），不碰真实数据。
"""
from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.evidence import EvidenceGate
from app.personal_context import PersonalContext, render_for_prompt
from app.rag import RAGEngine
from app.retriever import Retriever, SearchResult
from app.router import QueryRouter
from fakes import ScriptedLLM

# 空术语表路由器：对任何问题都给 global——用来测「Router 已注入但零命中」
EMPTY_TERMS: dict[str, list[str]] = {}

# 术语表只认「心理时间」；测试库里的文档都落在 unknown 域，
# 所以路由命中 philosophy 后必然触发空域 fallback 回全库，结果依然非空。
ONE_TERM = {"philosophy": ["心理时间"]}


def _verdict(status: str, confidence: float = 0.8) -> str:
    return json.dumps({
        "status": status,
        "confidence": confidence,
        "reason": "评审理由",
        "supported_aspects": ["方面A"],
        "unsupported_aspects": ["方面B"],
    }, ensure_ascii=False)


def _answer_verification(status: str = "supported", supported: bool = True) -> str:
    return json.dumps({
        "status": status,
        "confidence": 0.9,
        "reason": "回答中的主张都能由对应资料支持",
        "claims": [{
            "unit_id": "u1",
            "claim": "回答中的事实主张",
            "is_claim": True,
            "supported": supported,
            "citation_indexes": [1],
            "evidence_quotes": [{"citation_index": 1, "quote": "合同违约"}],
            "reason": "资料直接支持",
        }],
        "invalid_citations": [],
    }, ensure_ascii=False)


def _advisor_answer_verification(*claims: dict, status: str = "supported") -> str:
    normalized_claims = []
    for index, claim in enumerate(claims, start=1):
        item = dict(claim)
        item.setdefault("unit_id", f"u{index}")
        item.setdefault("is_claim", True)
        normalized_claims.append(item)
    return json.dumps({
        "status": status,
        "confidence": 0.9,
        "reason": "按来源逐项核对",
        "claims": normalized_claims,
        "invalid_citations": [],
    }, ensure_ascii=False)


class _ChunkedScriptedLLM(ScriptedLLM):
    """为严格流式测试把生成答案拆成多个 token。"""

    def chat_stream(self, messages):
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("ScriptedLLM 脚本耗尽：出现了未预期的流式调用")
        answer = self._responses.pop(0)
        midpoint = max(1, len(answer) // 2)
        yield answer[:midpoint]
        if answer[midpoint:]:
            yield answer[midpoint:]


def _build(settings: Settings, ingested, llm, **kwargs) -> RAGEngine:
    retriever = Retriever(
        embedding=ingested[1], store=ingested[0], default_top_k=settings.top_k,
    )
    return RAGEngine(
        retriever=retriever, llm=llm, top_k=settings.top_k,
        vault_name=settings.vault_name, **kwargs,
    )


def test_default_answer_behavior_unchanged(settings, ingested):
    """开关全关：正常回答，无路由无评审，timings 只有两个基础阶段。"""
    llm = ScriptedLLM(["这是回答。"])
    engine = _build(settings, ingested, llm)

    result = engine.answer("合同违约怎么处理？")

    assert result.answer == "这是回答。"
    assert result.route is None
    assert result.evidence is None
    assert set(result.timings) == {"retrieval_ms", "answer_generation_ms"}
    assert len(llm.calls) == 1  # 只有生成一次调用
    # system prompt 不带 partial 附加条款
    assert "缺少依据的部分" not in llm.calls[0][0]["content"]


def test_strict_evidence_requires_gate(settings, ingested):
    retriever = Retriever(embedding=ingested[1], store=ingested[0], default_top_k=settings.top_k)
    with pytest.raises(ValueError, match="strict_evidence 需要 evidence_gate"):
        RAGEngine(retriever=retriever, llm=ScriptedLLM([]), strict_evidence=True)


def test_router_zero_hit_keeps_global_without_llm_call(settings, ingested):
    llm = ScriptedLLM(["回答"])
    engine = _build(
        settings, ingested, llm,
        router=QueryRouter(EMPTY_TERMS), router_domains=["unknown"],
    )

    result = engine.answer("合同违约怎么处理？")

    assert result.route is not None
    assert result.route.mode == "global"
    assert result.route.origin == "rules"
    assert result.evidence is None
    assert len(llm.calls) == 1
    assert "router_ms" in result.timings


def test_router_hit_scopes_retrieval(settings, ingested):
    """规则命中：scoped 检索（空域回退全库），路由不花 LLM 调用。"""
    llm = ScriptedLLM(["回答"])
    engine = _build(
        settings, ingested, llm,
        router=QueryRouter(ONE_TERM), router_domains=["unknown", "philosophy"],
    )

    result = engine.answer("什么是心理时间？")

    assert result.route.mode == "single_domain"
    assert result.route.domains == ["philosophy"]
    assert len(llm.calls) == 1
    assert result.sources  # 空域 fallback 后仍有结果


def test_manual_domains_skip_router(settings, ingested):
    llm = ScriptedLLM(["回答"])
    engine = _build(
        settings, ingested, llm,
        router=QueryRouter(ONE_TERM), router_domains=["unknown"],
    )

    result = engine.answer("什么是心理时间？", domains=["unknown"])

    assert result.route is None


def test_router_llm_fallback_called_only_on_zero_hit(settings, ingested):
    """规则零命中 + fallback 开：第一次 LLM 调用是路由，第二次才是生成。"""
    llm = ScriptedLLM([json.dumps({"domains": [], "confidence": 0.4, "reason": "拿不准"}), "回答"])
    engine = _build(
        settings, ingested, llm,
        router=QueryRouter(EMPTY_TERMS), router_domains=["unknown"],
        llm_router_fallback_enabled=True,
    )

    result = engine.answer("合同违约怎么处理？")

    assert result.route.origin == "llm_fallback"
    assert len(llm.calls) == 2
    assert "路由器" in llm.calls[0][0]["content"]  # 第一次是 Router prompt
    assert "问答助手" in llm.calls[1][0]["content"]  # 第二次才是生成 prompt
    assert result.timings["router_ms"] >= 0


def test_evidence_insufficient_refuses_without_generation(settings, ingested):
    llm = ScriptedLLM([_verdict("insufficient")])
    engine = _build(settings, ingested, llm, evidence_gate=EvidenceGate(llm))

    result = engine.answer("合同违约怎么处理？")

    assert "本次检索到的资料不足以支持" in result.answer
    assert "方面B" in result.answer  # Judge 给出的缺失信息出现在拒答里
    assert result.evidence is not None and result.evidence.status == "insufficient"
    assert len(llm.calls) == 1  # 只有 Judge，没有生成
    assert "answer_generation_ms" not in result.timings
    assert result.sources  # 来源照常返回，供前端打「仅供参考」标记


def test_evidence_partial_adds_disclosure_requirement(settings, ingested):
    llm = ScriptedLLM([_verdict("partial"), "分两部分回答。"])
    engine = _build(settings, ingested, llm, evidence_gate=EvidenceGate(llm))

    result = engine.answer("合同违约怎么处理？")

    assert result.answer == "分两部分回答。"
    assert "缺少依据的部分" in llm.calls[1][0]["content"]  # 生成 prompt 带附加条款
    assert result.evidence is not None and result.evidence.status == "partial"


@pytest.mark.parametrize("stream", [False, True])
def test_evidence_gate_sees_the_same_clean_context_as_generation(stream):
    """Judge 与生成器必须共享清洗后的完整 context，不能只看原文前缀。"""
    long_content = "前置说明。" + ("补充文字。" * 140) + "关键证据在这里。"
    long_content += '", "file_path": "/exported/path.md"}'
    long_hit = SearchResult(
        score=0.8, chunk_id=2, document_id=1, chunk_index=0,
        title="长文档", file_path="长文档.md", heading="正文",
        source="测试库", content=long_content,
    )

    class LongRetriever:
        def search(self, question, top_k=1, domains=None):
            return [long_hit]

    llm = ScriptedLLM([_verdict("sufficient"), "回答"])
    engine = RAGEngine(
        retriever=LongRetriever(), llm=llm, top_k=1,
        context_filter_enabled=True, filter_retrieve_top_k=1,
        evidence_gate=EvidenceGate(llm),
    )

    if stream:
        list(engine.answer_stream("问题？"))
    else:
        engine.answer("问题？")

    judge_user = llm.calls[0][1]["content"]
    generation_user = llm.calls[1][1]["content"]
    assert "关键证据在这里" in judge_user
    assert "关键证据在这里" in generation_user
    assert '"file_path": "/exported/path.md"' not in judge_user
    assert '"file_path": "/exported/path.md"' not in generation_user


def test_evidence_sufficient_uses_normal_prompt(settings, ingested):
    llm = ScriptedLLM([_verdict("sufficient"), "正常回答。"])
    engine = _build(settings, ingested, llm, evidence_gate=EvidenceGate(llm))

    result = engine.answer("合同违约怎么处理？")

    assert result.answer == "正常回答。"
    assert "缺少依据的部分" not in llm.calls[1][0]["content"]
    assert set(result.timings) == {
        "retrieval_ms", "evidence_judge_ms",
        "first_evidence_status", "final_evidence_status", "answer_generation_ms",
    }
    assert result.timings["first_evidence_status"] == "sufficient"
    assert result.timings["final_evidence_status"] == "sufficient"
    assert "rescue_triggered" not in result.timings


def test_evidence_judge_error_falls_back_to_normal_answer(settings, ingested):
    """Judge 输出损坏：按 sufficient 兜底，生成照常执行，留 debug 标记。"""
    llm = ScriptedLLM(["完全不是 JSON 的输出", "兜底回答。"])
    engine = _build(settings, ingested, llm, evidence_gate=EvidenceGate(llm))

    result = engine.answer("合同违约怎么处理？")

    assert result.answer == "兜底回答。"
    assert result.evidence is not None
    assert result.evidence.status == "sufficient"
    assert result.evidence.debug["judge_error"] is True


@pytest.mark.parametrize("stream", [False, True])
def test_strict_evidence_judge_error_refuses_without_generation(settings, ingested, stream):
    """严格模式在阻塞/流式入口都不得把 Judge 失败交给生成模型。"""
    llm = ScriptedLLM(["完全不是 JSON 的输出", "不应生成的回答"])
    engine = _build(
        settings, ingested, llm,
        evidence_gate=EvidenceGate(llm), strict_evidence=True,
    )

    if stream:
        events = list(engine.answer_stream("合同违约怎么处理？"))
        done = events[-1]["data"]
        assert events[-1]["event"] == "done"
        assert "generation_started" not in [event["event"] for event in events]
        evidence = done["evidence"]
        answer = done["answer"]
    else:
        result = engine.answer("合同违约怎么处理？")
        evidence = result.evidence.to_dict()
        answer = result.answer

    assert "证据评审未完成" in answer
    assert "不代表知识库中不存在" in answer
    assert evidence["status"] == "insufficient"
    assert evidence["judge_error"] is True
    assert evidence["strict_fail_closed"] is True
    assert len(llm.calls) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_strict_evidence_gate_exception_is_not_a_generation_path(settings, ingested, stream):
    class ExplodingGate:
        def judge(self, *args, **kwargs):
            raise RuntimeError("证据评审不可用")

    llm = ScriptedLLM(["不应生成的回答"])
    engine = _build(
        settings, ingested, llm,
        evidence_gate=ExplodingGate(), strict_evidence=True,
    )

    if stream:
        events = list(engine.answer_stream("合同违约怎么处理？"))
        assert events[-1]["event"] == "done"
        answer = events[-1]["data"]["answer"]
        evidence = events[-1]["data"]["evidence"]
    else:
        result = engine.answer("合同违约怎么处理？")
        answer = result.answer
        evidence = result.evidence.to_dict()

    assert "证据评审未完成" in answer
    assert evidence["strict_fail_closed"] is True
    assert len(llm.calls) == 0


@pytest.mark.parametrize("stream", [False, True])
def test_strict_evidence_rejects_malformed_gate_result(settings, ingested, stream):
    """严格模式不把注入 Gate 的无效返回值当成评审通过。"""
    class MalformedGate:
        def judge(self, *args, **kwargs):
            return None

    llm = ScriptedLLM(["不应生成的回答"])
    engine = _build(
        settings, ingested, llm,
        evidence_gate=MalformedGate(), strict_evidence=True,
    )

    if stream:
        events = list(engine.answer_stream("合同违约怎么处理？"))
        assert events[-1]["event"] == "done"
        answer = events[-1]["data"]["answer"]
        evidence = events[-1]["data"]["evidence"]
    else:
        result = engine.answer("合同违约怎么处理？")
        answer = result.answer
        evidence = result.evidence.to_dict()

    assert "证据评审未完成" in answer
    assert evidence["strict_fail_closed"] is True
    assert len(llm.calls) == 0


def test_strict_evidence_verifies_final_answer_before_returning(settings, ingested):
    generated = "依据资料，合同违约的处理需要继续履行或采取补救措施。[1]"
    llm = ScriptedLLM([_verdict("sufficient"), generated, _answer_verification()])
    engine = _build(
        settings, ingested, llm,
        evidence_gate=EvidenceGate(llm), strict_evidence=True,
    )

    result = engine.answer("合同违约怎么处理？")

    assert result.answer == generated
    assert result.answer_verification is not None
    assert result.answer_verification.passed is True
    assert result.timings["answer_verification_ms"] >= 0
    assert len(llm.calls) == 3
    verifier_user = llm.calls[2][1]["content"]
    assert generated in verifier_user
    assert "[1]" in verifier_user


def test_strict_advisor_verifier_sees_same_personal_context_and_allows_advice_without_citations(
    settings, ingested,
):
    personal = PersonalContext(
        loaded=True,
        source_path="context.yaml",
        profile={"current_direction": "AI 应用开发"},
    )
    generated = "你的当前方向是 AI 应用开发。\n我的建议：先整理 RAG 的检索评估。"
    llm = ScriptedLLM([
        _verdict("partial"),
        generated,
        _advisor_answer_verification(
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
                "reason": "回答明确标注为建议",
            },
        ),
    ])
    engine = _build(
        settings,
        ingested,
        llm,
        evidence_gate=EvidenceGate(llm),
        strict_evidence=True,
        personal_context_loader=lambda: personal,
    )

    result = engine.answer("我下一步做什么？", mode="advisor")

    assert result.answer == generated
    assert result.answer_verification is not None
    assert result.answer_verification.passed is True
    verifier_user = llm.calls[2][1]["content"]
    assert render_for_prompt(personal) in verifier_user
    assert "【用户当前情况】" in verifier_user
    assert "【编号参考资料】" in verifier_user


def test_strict_advisor_verifier_rejects_user_fact_not_in_personal_context(settings, ingested):
    personal = PersonalContext(
        loaded=True,
        source_path="context.yaml",
        profile={"current_direction": "AI 应用开发"},
    )
    generated = "你目前已经有三名全职协作者。"
    llm = ScriptedLLM([
        _verdict("partial"),
        generated,
        _advisor_answer_verification(
            {
                "claim": "用户已经有三名全职协作者",
                "basis": "personal_context",
                "supported": False,
                "citation_indexes": [],
                "reason": "个人情况没有写出",
            },
        ),
    ])
    engine = _build(
        settings,
        ingested,
        llm,
        evidence_gate=EvidenceGate(llm),
        strict_evidence=True,
        personal_context_loader=lambda: personal,
    )

    result = engine.answer("我的团队情况？", mode="advisor")

    assert result.answer != generated
    assert "最终答案未通过逐项证据核对" in result.answer
    assert result.answer_verification is not None
    assert result.answer_verification.passed is False
    assert result.answer_verification.status == "unsupported"


def test_strict_evidence_rejects_final_answer_with_unsupported_claim(settings, ingested):
    generated = "资料没有依据但模型仍然给出了结论。[1]"
    llm = ScriptedLLM([_verdict("sufficient"), generated, _answer_verification(supported=False)])
    engine = _build(
        settings, ingested, llm,
        evidence_gate=EvidenceGate(llm), strict_evidence=True,
    )

    result = engine.answer("合同违约怎么处理？")

    assert result.answer != generated
    assert "最终答案未通过逐项证据核对" in result.answer
    assert result.answer_verification is not None
    assert result.answer_verification.passed is False
    assert result.answer_verification.status == "unsupported"
    assert len(llm.calls) == 3


def test_strict_evidence_rejects_invalid_citation_without_review_call(settings, ingested):
    generated = "这条结论引用了不存在的来源。[99]"
    llm = ScriptedLLM([_verdict("sufficient"), generated])
    engine = _build(
        settings, ingested, llm,
        evidence_gate=EvidenceGate(llm), strict_evidence=True,
    )

    result = engine.answer("合同违约怎么处理？")

    assert "最终答案未通过逐项证据核对" in result.answer
    assert result.answer_verification is not None
    assert result.answer_verification.debug["citation_error"] is True
    assert result.answer_verification.invalid_citations == [99]
    assert len(llm.calls) == 2  # Gate + generation；编号硬失败不再调用 verifier


def test_strict_evidence_verifier_exception_is_fail_closed(settings, ingested):
    class ExplodingVerifier:
        def verify(self, *args, **kwargs):
            raise RuntimeError("答案评审不可用")

    generated = "模型生成的回答。[1]"
    llm = ScriptedLLM([_verdict("sufficient"), generated])
    engine = _build(
        settings, ingested, llm,
        evidence_gate=EvidenceGate(llm), strict_evidence=True,
        answer_verifier=ExplodingVerifier(),
    )

    result = engine.answer("合同违约怎么处理？")

    assert result.answer != generated
    assert result.answer_verification is not None
    assert result.answer_verification.debug["strict_fail_closed"] is True


def test_strict_stream_buffers_answer_until_verification_passes(settings, ingested):
    generated = "已核对的回答。[1]"
    llm = _ChunkedScriptedLLM([_verdict("sufficient"), generated, _answer_verification()])
    engine = _build(
        settings, ingested, llm,
        evidence_gate=EvidenceGate(llm), strict_evidence=True,
    )

    events = list(engine.answer_stream("合同违约怎么处理？"))
    names = [event["event"] for event in events]
    done = events[-1]["data"]

    assert names.index("generation_started") < names.index("token") < names.index("done")
    assert "已核对的回答。[1]" == done["answer"]
    assert "".join(event["data"]["text"] for event in events if event["event"] == "token") == generated
    assert done["answer_verification"]["status"] == "supported"


def test_strict_stream_never_leaks_answer_when_verification_rejects(settings, ingested):
    generated = "这段内容没有原文依据。[1]"
    llm = _ChunkedScriptedLLM([
        _verdict("sufficient"), generated, _answer_verification(supported=False),
    ])
    engine = _build(
        settings, ingested, llm,
        evidence_gate=EvidenceGate(llm), strict_evidence=True,
    )

    events = list(engine.answer_stream("合同违约怎么处理？"))
    done = events[-1]["data"]

    assert events[-1]["event"] == "done"
    assert not [event for event in events if event["event"] == "token"]
    assert generated not in done["answer"]
    assert "最终答案未通过逐项证据核对" in done["answer"]
    assert done["answer_verification"]["status"] == "unsupported"
    assert "claims" not in done["answer_verification"]
    assert "unsupported_claims" not in done["answer_verification"]
    assert "没有原文依据" not in json.dumps(done["answer_verification"], ensure_ascii=False)


def test_stream_emits_routing_and_evidence_events(settings, ingested):
    llm = ScriptedLLM([_verdict("sufficient"), "流式回答"])
    engine = _build(
        settings, ingested, llm,
        router=QueryRouter(EMPTY_TERMS), router_domains=["unknown"],
        evidence_gate=EvidenceGate(llm),
    )

    events = list(engine.answer_stream("合同违约怎么处理？"))
    names = [e["event"] for e in events]

    assert names[0] == "routing_completed"
    pos = {n: names.index(n) for n in set(names)}
    assert (
        pos["routing_completed"] < pos["retrieval_started"]
        < pos["retrieval_completed"] < pos["sources"]
        < pos["evidence_completed"] < pos["generation_started"] < pos["done"]
    )
    done = events[-1]["data"]
    assert done["evidence"]["status"] == "sufficient"
    assert done["route"]["mode"] == "global"


def test_stream_insufficient_skips_generation(settings, ingested):
    llm = ScriptedLLM([_verdict("insufficient")])
    engine = _build(settings, ingested, llm, evidence_gate=EvidenceGate(llm))

    events = list(engine.answer_stream("合同违约怎么处理？"))
    names = [e["event"] for e in events]

    assert "generation_started" not in names
    assert "token" not in names
    done = events[-1]["data"]
    assert "本次检索到的资料不足以支持" in done["answer"]
    assert done["evidence"]["status"] == "insufficient"
