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
from app.rag import RAGEngine
from app.retriever import Retriever
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

    assert "没有足够资料" in result.answer
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
    assert "没有足够资料" in done["answer"]
    assert done["evidence"]["status"] == "insufficient"
