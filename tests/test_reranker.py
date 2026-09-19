"""精排层的单元测试。

这里的所有用例都注入假模型对象，绝不真的加载 cross-encoder 权重——
真加载一次要占约 1.1 GB 显存、花几十秒，单测不该做这件事。
`_load_cross_encoder` 只用 monkeypatch 替换掉。

覆盖的重点是"精排失败不能拖垮问答"这条约束：模型抛异常时必须回落
到粗排顺序，而不是让上层收到 500。
"""
from __future__ import annotations

import pytest

from app.config import Settings
from app.reranker import (
    CrossEncoderReranker,
    RerankStats,
    _resolve_device,
)
from app.retriever import SearchResult


def _hit(chunk_id: int, content: str, score: float = 0.5) -> SearchResult:
    """构造一条检索命中，其余字段填合法占位值。"""
    return SearchResult(
        score=score,
        chunk_id=chunk_id,
        document_id=chunk_id,
        chunk_index=0,
        title=f"文档{chunk_id}",
        file_path=f"文档{chunk_id}.md",
        heading="第一节",
        source="测试知识库",
        content=content,
    )


class FakeModel:
    """按 (问题, 段落) 对返回固定分数的假 cross-encoder。

    scores 顺序和 inputs 一一对应，用来验证"分数有没有对齐到正确的候选"。
    记录 last_pairs 是为了断言传给模型的确实是 (question, content) 而不是别的字段。
    """

    def __init__(self, scores, *, raise_error: Exception | None = None):
        self.scores = list(scores)
        self.raise_error = raise_error
        self.last_pairs: list[tuple[str, str]] = []
        self.calls = 0

    def predict(self, inputs, batch_size=None, max_length=None, apply_softmax=None,
                show_progress_bar=None, convert_to_numpy=None, **kwargs):
        self.calls += 1
        self.last_pairs = list(inputs)
        if self.raise_error is not None:
            raise self.raise_error
        assert len(self.scores) == len(inputs), "假模型要求 scores 长度和候选数一致"
        return self.scores


@pytest.fixture
def hits() -> list[SearchResult]:
    return [_hit(i, f"内容 {i}") for i in range(3)]


def test_rerank_sorts_by_score_descending(hits):
    """精排的核心行为：按模型给的分重新排序，而不是沿用粗排顺序。"""
    model = FakeModel([1.0, 9.0, 5.0])
    reranked = CrossEncoderReranker(model=model).rerank("问题", hits)
    assert [h.chunk_id for h in reranked] == [1, 2, 0]
    assert [round(h.score, 4) for h in reranked] == [9.0, 5.0, 1.0]


def test_rerank_writes_score_onto_result(hits):
    """分数必须写进 SearchResult.score，上层 sources 才拿得到精排分数。"""
    model = FakeModel([1.0, 9.0, 5.0])
    reranked = CrossEncoderReranker(model=model).rerank("问题", hits)
    assert reranked[0].score == 9.0
    assert isinstance(reranked[0].score, float)


def test_rerank_truncates_to_top_k(hits):
    """top_k 截断放在排序之后，取的是精排分数最高的那几条。"""
    model = FakeModel([1.0, 9.0, 5.0])
    reranked = CrossEncoderReranker(model=model).rerank("问题", hits, top_k=2)
    assert [h.chunk_id for h in reranked] == [1, 2]


def test_rerank_returns_all_when_top_k_is_none(hits):
    """不传 top_k 就返回全部候选，截断交给上层决定。"""
    model = FakeModel([1.0, 9.0, 5.0])
    reranked = CrossEncoderReranker(model=model).rerank("问题", hits)
    assert len(reranked) == 3


def test_rerank_empty_hits_is_noop():
    """候选为空时直接返回空列表，不碰模型也不记调用次数。"""
    model = FakeModel([])
    reranker = CrossEncoderReranker(model=model)
    assert reranker.rerank("问题", []) == []
    assert model.calls == 0
    assert reranker.stats.calls == 0


def test_rerank_passes_question_and_content_pairs(hits):
    """喂给模型的必须是 (question, content)，用错字段等于白排。"""
    model = FakeModel([1.0, 2.0, 3.0])
    CrossEncoderReranker(model=model).rerank("焦糖色要写吗", hits)
    assert model.last_pairs == [
        ("焦糖色要写吗", "内容 0"),
        ("焦糖色要写吗", "内容 1"),
        ("焦糖色要写吗", "内容 2"),
    ]


def test_rerank_does_not_mutate_input(hits):
    """返回的应该是新对象：调用方手里的原始 hits 不能被改写，
    否则 A/B 评估里 raw 和 reranker 两组会互相污染。"""
    original_scores = [h.score for h in hits]
    original_ids = [h.chunk_id for h in hits]
    model = FakeModel([1.0, 9.0, 5.0])
    CrossEncoderReranker(model=model).rerank("问题", hits)
    assert [h.score for h in hits] == original_scores
    assert [h.chunk_id for h in hits] == original_ids


def test_rerank_falls_back_on_exception(hits):
    """模型打分抛异常时保留粗排顺序继续走，上层照常能出答案。"""
    model = FakeModel([0.0, 0.0, 0.0], raise_error=RuntimeError("CUDA out of memory"))
    reranker = CrossEncoderReranker(model=model)
    out = reranker.rerank("问题", hits, top_k=3)
    assert [h.chunk_id for h in out] == [0, 1, 2]
    assert [h.score for h in out] == [0.5, 0.5, 0.5]
    assert reranker.stats.fallback == 1
    assert reranker.stats.pairs_scored == 0


def test_rerank_counts_stats():
    """耗时和计数是 A/B 表里"查询耗时增加多少"那一栏的唯一数据来源。"""
    model = FakeModel([3.0, 1.0])
    reranker = CrossEncoderReranker(model=model)
    reranker.rerank("问题", [_hit(0, "a"), _hit(1, "b")])
    stats = reranker.stats
    assert stats.calls == 1
    assert stats.pairs_scored == 2
    assert stats.latency_ms_total > 0
    assert stats.fallback == 0


def test_stats_returns_a_copy():
    """stats 暴露的是拷贝，外面改它不能污染内部计数。"""
    reranker = CrossEncoderReranker(model=FakeModel([1.0]))
    reranker.stats.calls = 999
    assert reranker.stats.calls == 0


def test_stats_defaults_are_zero():
    stats = RerankStats()
    assert stats.calls == 0
    assert stats.pairs_scored == 0
    assert stats.latency_ms_total == 0.0
    assert stats.fallback == 0


def test_preload_is_noop_when_model_injected():
    """注入了模型就不该再加载权重，这是测试和 A/B 脚本能共用同一个类的原因。"""
    model = FakeModel([1.0])
    reranker = CrossEncoderReranker(model=model)
    assert reranker.preload() == 0.0
    assert reranker.loaded
    assert reranker._model is model


def test_preload_loads_model_when_missing(monkeypatch):
    """没注入模型时 preload 走真实加载路径；这里只验证它被调用了。"""
    loaded = FakeModel([1.0])
    monkeypatch.setattr("app.reranker._load_cross_encoder", lambda name, device, dtype: loaded)
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3", device="cpu", dtype="fp32")
    assert not reranker.loaded
    assert reranker.preload() >= 0.0
    assert reranker.loaded
    assert reranker._model is loaded


def test_preload_loads_once(monkeypatch):
    """重复 preload 不该反复加载权重（加载一次几十秒）。"""
    calls = {"n": 0}

    def loader(name, device, dtype):
        calls["n"] += 1
        return FakeModel([1.0])

    monkeypatch.setattr("app.reranker._load_cross_encoder", loader)
    reranker = CrossEncoderReranker(model=None, device="cpu", dtype="fp32")
    reranker.preload()
    reranker.preload()
    assert calls["n"] == 1


def test_model_name_is_exposed():
    """eval 的 meta 里要记录用了哪个 reranker，得能读到。"""
    assert CrossEncoderReranker(model_name="自定义模型").model_name == "自定义模型"


def test_resolve_device_explicit_wins_without_torch():
    """显式指定设备时完全不碰 torch，CUDA 挂了也不影响启动。"""
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda") == "cuda"
    assert _resolve_device("mps") == "mps"


def test_resolve_device_auto_returns_real_device(monkeypatch):
    """auto 走 torch 探测；把 is_available 打平成 False 时应该落到 cpu，
    这样"显式 cuda 但机器没 CUDA"的场景在测试里也能复现。"""
    torch = __import__("torch")
    monkeypatch.setattr(torch, "cuda", type("FakeCUDA", (), {"is_available": staticmethod(lambda: False)})())
    assert _resolve_device("auto") == "cpu"


def test_config_defaults_keep_reranker_off():
    """RERANKER_ENABLED 必须默认 false：baseline 要随时复现，显存不该被占。"""
    settings = Settings()
    assert settings.reranker_enabled is False
    assert settings.reranker_candidate_top_k == 50
    assert settings.reranker_batch_size >= 1


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"reranker_enabled": True, "reranker_device": "gpu"}, "RERANKER_DEVICE"),
        ({"reranker_enabled": True, "reranker_dtype": "fp64"}, "RERANKER_DTYPE"),
        ({"reranker_enabled": True, "reranker_candidate_top_k": 0}, "RERANKER_CANDIDATE_TOP_K"),
        ({"reranker_enabled": True, "reranker_batch_size": -1}, "RERANKER_BATCH_SIZE"),
    ],
)
def test_config_rejects_bad_reranker_values(overrides, message):
    """开启精排时校验配置，关着的时候不该挡启动。

    provider 显式设成离线值，避免 .env 里的真实配置让校验提前在别处失败。
    """
    settings = Settings(embedding_provider="local", llm_provider="mock", **overrides)
    with pytest.raises(ValueError, match=message):
        settings.validate_providers()


def test_config_accepts_disabled_reranker_with_bad_device():
    """关掉精排时即使填了非法设备名也不能让服务起不来。"""
    settings = Settings(
        embedding_provider="local", llm_provider="mock",
        reranker_enabled=False, reranker_device="gpu",
    )
    settings.validate_providers()


def test_config_accepts_legal_reranker_values():
    settings = Settings(
        embedding_provider="local_semantic",
        llm_provider="mock",
        reranker_enabled=True,
        reranker_device="cuda",
        reranker_dtype="fp16",
        reranker_candidate_top_k=50,
        reranker_batch_size=4,
    )
    settings.validate_providers()
