"""Verify configured retrieval behavior through the shared engine assembly."""
from unittest.mock import Mock

import pytest

from app.config import Settings
from app.engine import build_engine
from app.evidence import EvidenceGate
from app.knowledge_domains import load_router
from app.rag import RAGEngine
from app.retriever import Retriever, SearchResult
from app.router import ORIGIN_LLM_FALLBACK, QueryRouter
from fakes import ScriptedLLM


def test_settings_repr_does_not_expose_api_keys():
    configured = Settings(
        _env_file=None,
        embedding_api_key="test-embedding-secret",
        llm_api_key="test-llm-secret",
    )

    assert "test-embedding-secret" not in repr(configured)
    assert "test-llm-secret" not in str(configured)


def test_engine_applies_document_diversity(settings, ingested):
    configured = settings.model_copy(update={
        "diversity_enabled": True,
        "max_chunks_per_document": 1,
    })
    engine, _ = build_engine(
        configured, embedding=ingested[1], store=ingested[0],
        llm=ScriptedLLM(["回答"]),
    )

    result = engine.answer("合同违约", domains=[])

    paths = [source.file_path for source in result.sources]
    assert len(paths) == 2
    assert len(paths) == len(set(paths))


def test_engine_keeps_diversity_disabled_by_default(settings, ingested):
    engine, _ = build_engine(
        settings, embedding=ingested[1], store=ingested[0],
        llm=ScriptedLLM(["回答"]),
    )

    result = engine.answer("合同违约", domains=[])

    assert result.used_chunks == settings.top_k
    assert len({source.file_path for source in result.sources}) < result.used_chunks


def test_engine_search_and_answer_share_one_reranker(settings, ingested, monkeypatch):
    from app.reranker import CrossEncoderReranker
    class Model:
        def predict(self, pairs, **kwargs):
            return list(range(len(pairs)))
    reranker = CrossEncoderReranker(model=Model(), include_metadata=True)
    monkeypatch.setattr("app.engine.build_reranker", lambda settings: reranker)
    configured = settings.model_copy(update={"reranker_enabled": True})
    engine, _ = build_engine(configured, embedding=ingested[1], store=ingested[0],
                             llm=ScriptedLLM(["回答"]))

    hits = engine.retrieve("合同违约", top_k=3, domains=[]).hits
    assert reranker.stats.calls == 1
    answer = engine.answer("合同违约", top_k=3, domains=[])

    assert reranker.stats.calls == 2  # exactly once per request, not twice in ask
    assert [(h.file_path, h.heading) for h in hits] == [(s.file_path, s.heading) for s in answer.sources]
    assert [h.content for h in hits] == [s.content for s in answer.sources]
    assert [round(h.rerank_score, 4) for h in hits] == [s.score for s in answer.sources]
    assert engine.rerank_stats.calls == 2


def test_engine_retrieve_routes_and_cleans_before_evidence_review():
    hit = SearchResult(
        score=0.8, chunk_id=1, document_id=1, chunk_index=0,
        title="合同违约", file_path="合同违约.md", heading="", source="测试库",
        content='违约可能产生赔偿责任。", "file_path": "/exported/path.md"}',
    )
    retriever = Mock(spec=Retriever)
    retriever.search.return_value = [hit]
    rescue_retriever = Mock(spec=Retriever)
    llm = ScriptedLLM(['{"domains": ["law"], "confidence": 0.9}'])
    engine = RAGEngine(
        retriever=retriever, llm=llm,
        router=QueryRouter({}), router_domains=["law"], llm_router_fallback_enabled=True,
        context_filter_enabled=True, filter_retrieve_top_k=10,
        evidence_gate=EvidenceGate(llm), evidence_rescue_enabled=True,
        rescue_retriever=rescue_retriever,
    )

    result = engine.retrieve("合同违约怎么办？", top_k=1, routed=True)

    retriever.search.assert_called_once_with("合同违约怎么办？", top_k=10, domains=["law"])
    rescue_retriever.search.assert_not_called()
    assert len(llm.calls) == 1
    assert result.route.origin == ORIGIN_LLM_FALLBACK
    assert result.hits == [hit]
    assert result.contents == {hit.chunk_id: "违约可能产生赔偿责任。"}


def test_strict_evidence_requires_evidence_gate():
    configured = Settings(
        _env_file=None,
        embedding_provider="local",
        llm_provider="mock",
        strict_evidence_enabled=True,
        evidence_gate_enabled=False,
    )

    with pytest.raises(ValueError, match="STRICT_EVIDENCE_ENABLED=true"):
        configured.validate_providers()


def test_engine_wires_strict_evidence(settings, ingested):
    configured = settings.model_copy(update={
        "evidence_gate_enabled": True,
        "strict_evidence_enabled": True,
    })
    # build_engine must construct the gate in fail-closed mode. A malformed Judge
    # output should therefore stop before the generation LLM is called.
    engine, _ = build_engine(
        configured, embedding=ingested[1], store=ingested[0],
        llm=ScriptedLLM(["not json", "should not generate"]),
    )

    result = engine.answer("合同违约")

    assert result.evidence is not None
    assert result.evidence.debug["strict_fail_closed"] is True
    assert result.answer.startswith("证据评审未完成")


@pytest.mark.parametrize("question, domain", [
    ("什么是 POVM？", "quantum_mechanics"),
    ("Cournot 竞争的模型设定是什么？", "game_theory"),
    ("古诺竞争的模型设定是什么？", "game_theory"),
])
def test_configured_router_recognizes_existing_query_terms(settings, question, domain):
    router, _ = load_router(settings.knowledge_domains_path_resolved)

    assert router.route(question).domains == [domain]
