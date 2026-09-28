"""Evidence Rescue（证据救援）行为测试。

核心契约：
- 只在第一次 Evidence Gate 判定 insufficient 时触发；sufficient / partial 一步不多花；
- 救援 = 扩大候选池再评审一次，不做重排、不改写问题；
- 救援池不比首轮多出任何 chunk 时跳过第二次评审（同一批证据只会得到同一个结论）；
- 救援后仍然 insufficient 就正常拒答，不会因为候选池变大就硬判 sufficient。
- 开关默认关闭：关闭时链路只调一次 Judge，没有任何 rescue 相关的 timings key。
"""
from __future__ import annotations

import json

import pytest

from app.evidence import EvidenceGate
from app.rag import RAGEngine
from app.retriever import SearchResult
from fakes import ScriptedLLM


class FakeRetriever:
    """按 top_k 返回不同候选池的假检索器。

    用来精确控制「首轮看到的 5 条」和「救援看到的 20 条」；
    calls 记录每次检索的 (top_k, domains)，断言能落在检索参数上。
    """

    def __init__(self, pools: dict[int, list[SearchResult]]):
        self.pools = pools
        self.calls: list[tuple[int, list[str] | None]] = []
        self._diversity_enabled = False  # 与 Retriever 默认一致：不做每文档限额

    def search(self, question: str, top_k: int = 5, domains: list[str] | None = None) -> list[SearchResult]:
        self.calls.append((top_k, domains))
        return [hit for hit in self.pools.get(top_k, [])][:top_k]


def _hit(chunk_id: int, content: str) -> SearchResult:
    return SearchResult(
        score=0.8 - 0.01 * chunk_id, chunk_id=chunk_id, document_id=1, chunk_index=chunk_id,
        title="合同违约的认定与救济", file_path="法律/合同违约.md", heading="定义",
        source="测试库", content=content,
    )


def _verdict(status: str) -> str:
    return json.dumps({
        "status": status, "confidence": 0.8, "reason": "评审理由",
        "supported_aspects": ["方面A"], "unsupported_aspects": ["方面B"],
    }, ensure_ascii=False)


# 救援池前 5 条和首轮完全相同，第 6、7 条是只有扩大候选池才拿得到的资料。
# 第 7 条就是真实案例里那道凸轮运动规律题的答案所在 chunk（真实排名 #7）。
FIRST_FIVE = [_hit(i, f"资料 {i}") for i in range(1, 6)]
RESCUE_POOL = FIRST_FIVE + [_hit(6, "运动规律的分类"), _hit(7, "等速 / 等加速 / 摆线运动规律")]


def _build(retriever, llm, **kwargs) -> RAGEngine:
    return RAGEngine(retriever=retriever, llm=llm, top_k=5, vault_name="测试知识库", **kwargs)


def test_rescue_disabled_refuses_without_extra_call():
    llm = ScriptedLLM([_verdict("insufficient")])
    engine = _build(FakeRetriever({5: FIRST_FIVE}), llm, evidence_gate=EvidenceGate(llm))

    result = engine.answer("问题？")

    assert "没有足够资料" in result.answer
    assert result.evidence is not None and result.evidence.status == "insufficient"
    assert len(llm.calls) == 1
    assert result.timings["first_evidence_status"] == "insufficient"
    assert result.timings["final_evidence_status"] == "insufficient"
    assert not any(key.startswith("rescue_") for key in result.timings)
    assert "answer_generation_ms" not in result.timings


def test_rescue_recovers_a_partial_answer():
    llm = ScriptedLLM([_verdict("insufficient"), _verdict("partial"), "救援后的回答"])
    gate_llm = llm
    rescue = FakeRetriever({20: RESCUE_POOL})
    engine = _build(
        FakeRetriever({5: FIRST_FIVE}), gate_llm,
        evidence_gate=EvidenceGate(gate_llm),
        evidence_rescue_enabled=True, evidence_rescue_k=20, rescue_retriever=rescue,
    )

    result = engine.answer("问题？")

    assert result.answer == "救援后的回答"
    assert result.evidence is not None and result.evidence.status == "partial"
    assert result.evidence.debug["rescue_used"] is True
    assert result.timings["first_evidence_status"] == "insufficient"
    assert result.timings["final_evidence_status"] == "partial"
    assert result.timings["rescue_triggered"] is True
    assert result.timings["rescue_retrieval_ms"] >= 0
    assert result.timings["rescue_evidence_ms"] >= 0
    assert len(llm.calls) == 3  # Judge 两次 + 生成一次
    assert rescue.calls == [(20, None)]
    # 救援池进入 sources，回答的依据就是扩大后的这批资料
    assert len(result.sources) == 7
    assert "摆线运动规律" in llm.calls[2][1]["content"]
    assert "运动规律的分类" in llm.calls[2][1]["content"]


def test_rescue_still_insufficient_refuses_normally():
    """库真的没有这块内容时，救援后仍然是 insufficient，不能因为池子变大就硬判 sufficient。"""
    llm = ScriptedLLM([_verdict("insufficient"), _verdict("insufficient")])
    engine = _build(
        FakeRetriever({5: FIRST_FIVE}), llm,
        evidence_gate=EvidenceGate(llm),
        evidence_rescue_enabled=True, evidence_rescue_k=20,
        rescue_retriever=FakeRetriever({20: RESCUE_POOL}),
    )

    result = engine.answer("问题？")

    assert "没有足够资料" in result.answer
    assert result.evidence is not None and result.evidence.status == "insufficient"
    assert result.timings["rescue_triggered"] is True
    assert result.timings["final_evidence_status"] == "insufficient"
    assert len(llm.calls) == 2  # 两次 Judge，没有生成
    assert "answer_generation_ms" not in result.timings


@pytest.mark.parametrize("status", ["sufficient", "partial"])
def test_no_rescue_when_first_verdict_is_not_insufficient(status):
    llm = ScriptedLLM([_verdict(status), "回答"])
    rescue = FakeRetriever({20: RESCUE_POOL})
    engine = _build(
        FakeRetriever({5: FIRST_FIVE}), llm,
        evidence_gate=EvidenceGate(llm),
        evidence_rescue_enabled=True, evidence_rescue_k=20, rescue_retriever=rescue,
    )

    result = engine.answer("问题？")

    assert result.answer == "回答"
    assert rescue.calls == []  # 救援检索一次都没发生
    assert len(llm.calls) == 2  # Judge + 生成
    assert result.timings["rescue_triggered"] is False
    assert "rescue_retrieval_ms" not in result.timings
    assert "rescue_evidence_ms" not in result.timings
    assert result.timings["final_evidence_status"] == status


def test_rescue_skips_second_judge_when_pool_adds_nothing():
    """救援池和首轮一模一样时，直接复用首轮结论，不多花一次 LLM 调用。"""
    llm = ScriptedLLM([_verdict("insufficient")])
    rescue = FakeRetriever({20: FIRST_FIVE})
    engine = _build(
        FakeRetriever({5: FIRST_FIVE}), llm,
        evidence_gate=EvidenceGate(llm),
        evidence_rescue_enabled=True, evidence_rescue_k=20, rescue_retriever=rescue,
    )

    result = engine.answer("问题？")

    assert result.evidence is not None and result.evidence.status == "insufficient"
    assert result.timings["rescue_triggered"] is False
    # 扩大候选池这一次检索确实发生了（这是唯一的额外成本），第二次评审被省掉
    assert "rescue_retrieval_ms" in result.timings
    assert "rescue_evidence_ms" not in result.timings
    assert rescue.calls == [(20, None)]
    assert len(llm.calls) == 1  # 只有第一次 Judge


def test_scoped_retrieval_passes_domains_to_rescue():
    llm = ScriptedLLM([_verdict("insufficient"), _verdict("partial"), "回答"])
    rescue = FakeRetriever({20: RESCUE_POOL})
    engine = _build(
        FakeRetriever({5: FIRST_FIVE}), llm,
        evidence_gate=EvidenceGate(llm),
        evidence_rescue_enabled=True, evidence_rescue_k=20, rescue_retriever=rescue,
    )

    engine.answer("问题？", domains=["unknown"])

    assert rescue.calls == [(20, ["unknown"])]


def test_rescue_does_not_use_a_reranker():
    """救援明确不做重排：只扩大候选池，避免把 frozen 的重排层拉进这条路径。"""
    llm = ScriptedLLM([_verdict("insufficient"), _verdict("partial"), "回答"])

    class RecordingReranker:
        def __init__(self):
            self.calls = 0

        def rerank(self, question, pool):
            self.calls += 1
            return pool

    reranker = RecordingReranker()
    _build(
        FakeRetriever({5: FIRST_FIVE, 50: RESCUE_POOL}), llm,
        reranker=reranker, reranker_candidate_top_k=50,
        evidence_gate=EvidenceGate(llm),
        evidence_rescue_enabled=True, evidence_rescue_k=20,
        rescue_retriever=FakeRetriever({20: RESCUE_POOL}),
    ).answer("问题？")

    assert reranker.calls == 1  # 只有首轮那次


@pytest.mark.parametrize(
    "gate, enabled, k, retriever, message",
    [
        (None, True, 20, None, "evidence_gate"),
        (None, True, 20, FakeRetriever({20: []}), "evidence_gate"),
        (EvidenceGate(ScriptedLLM([])), True, 20, None, "rescue_retriever"),
        (EvidenceGate(ScriptedLLM([])), True, 0, FakeRetriever({20: []}), "正整数"),
    ],
)
def test_rescue_config_guardrails(gate, enabled, k, retriever, message):
    with pytest.raises(ValueError, match=message):
        _build(
            FakeRetriever({5: FIRST_FIVE}), ScriptedLLM([]),
            evidence_gate=gate, evidence_rescue_enabled=enabled,
            evidence_rescue_k=k, rescue_retriever=retriever,
        )


def test_rescue_stream_emits_rescue_event_with_final_sources():
    llm = ScriptedLLM([_verdict("insufficient"), _verdict("partial"), "流式回答"])
    engine = _build(
        FakeRetriever({5: FIRST_FIVE}), llm,
        evidence_gate=EvidenceGate(llm),
        evidence_rescue_enabled=True, evidence_rescue_k=20,
        rescue_retriever=FakeRetriever({20: RESCUE_POOL}),
    )

    events = list(engine.answer_stream("问题？"))
    names = [event["event"] for event in events]

    assert names.index("evidence_rescue_completed") > names.index("evidence_completed")
    assert names.index("evidence_rescue_completed") < names.index("generation_started")
    rescue_event = events[names.index("evidence_rescue_completed")]["data"]
    assert rescue_event["rescue_triggered"] is True
    assert rescue_event["used_chunks"] == 7
    assert rescue_event["verdict"]["status"] == "partial"
    assert [s["index"] for s in rescue_event["sources"]] == list(range(1, 8))
    assert "摆线运动规律" in rescue_event["sources"][-1]["chunk"]
    done = events[-1]["data"]
    assert done["timings"]["final_evidence_status"] == "partial"
    assert done["evidence"]["rescue_used"] is True
    # 前一个事件（救援之前）的 verdict 不携带 rescue 标记
    assert events[names.index("evidence_completed")]["data"]["verdict"]["rescue_used"] is False


def test_stream_has_no_rescue_event_when_rescue_is_disabled():
    """只开 Evidence Gate 不开救援时，事件序列必须和历史上完全一致。"""
    llm = ScriptedLLM([_verdict("sufficient"), "流式回答"])
    engine = _build(
        FakeRetriever({5: FIRST_FIVE}), llm,
        evidence_gate=EvidenceGate(llm),
    )

    names = [event["event"] for event in engine.answer_stream("问题？")]

    assert "evidence_rescue_completed" not in names
    assert names.index("evidence_completed") < names.index("generation_started")
