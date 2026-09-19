"""精排层：对粗排候选用 cross-encoder 重新打分排序。

这层为什么存在
    BGE-M3 是 bi-encoder：把问题和 chunk 各自独立编码成向量，再算相似度。
    它看不到「这个问题和这段文字放在一起是什么意思」，所以会把字面上命中
    「焦糖色」「囚徒困境」的段落排得很高，哪怕那才是错的答案。
    cross-encoder 把 (问题, 段落) 拼成一条输入一起过一遍模型，能捕捉到
    bi-encoder 漏掉的交互信号。

    代价：cross-encoder 不能离线建索引，必须在线对每个候选现算。
    所以它只能作用在粗排后的少量候选上，绝不能对全库跑。
    整条链路是：向量检索 Top-N -> cross-encoder 打分 -> 重排 -> 取 Top-K。

两条硬约束
    1. RERANKER_ENABLED 默认 False。评估阶段要随时拿回原始检索结果当 baseline，
       开关一开一关必须能复现同一个数字；而且加载模型要占约 1.1 GB 显存，
       没开的时候不该碰。
    2. rerank 失败不能拖垮整次问答。模型打分抛异常时保留粗排顺序继续走，
       并把原因记进 stats，而不是让 /api/chat 返回 500。

分数口径（重要）
    rerank 返回的 score 是 cross-encoder 的原始 logit，不是概率也不是余弦相似度。
    它只能「在同一个模型内部比大小」，不能和 bge-m3 的 0~1 相似度放到一张表里比。
    eval 里用它只比较排名，不比较绝对值。
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

from .config import DEFAULT_RERANKER_MODEL
from .retriever import SearchResult

# 单条 (问题, 段落) 的 token 上限。取 1024 和 embedding 层一致：
# 实测 chunk 长度 p50=304 / p99=780 token，加问题后 p99 约 810，1024 只截最长那 1%。
MAX_RERANK_SEQ_LENGTH = 1024

# 单次 predict 的 batch。cross-encoder 的激活内存远大于 bi-encoder
# （每对都要跑一遍完整 transformer），batch 太大在 6GB 显存上会 OOM。
RERANK_BATCH_SIZE = 8


@dataclass
class RerankStats:
    """精排的观测数据，给 eval 统计耗时和吞吐用。"""

    calls: int = 0            # rerank() 被调用的次数
    pairs_scored: int = 0     # 实际打过分数的 (问题, 段落) 对数
    latency_ms_total: float = 0.0
    fallback: int = 0         # 打分失败、退回粗排顺序的次数


class CrossEncoderReranker:
    """用本地 cross-encoder 给候选重新打分。

    model 参数用来注入假模型做单元测试——真加载一次要占 1.1 GB 显存、
    花几十秒，测试里不应该做这件事。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "auto",
        dtype: str = "auto",
        model=None,
        batch_size: int = RERANK_BATCH_SIZE,
        max_length: int = MAX_RERANK_SEQ_LENGTH,
    ):
        self._model_name = model_name
        self._device = device
        self._dtype = dtype
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = model
        self._stats = RerankStats()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def stats(self) -> RerankStats:
        """返回一份拷贝，调用方改了不会影响内部计数。"""
        return dataclasses.replace(self._stats)

    def preload(self) -> float:
        """加载模型权重，返回耗时秒数。在启动时调用，别留到第一个请求上。"""
        if self._model is not None:
            return 0.0
        started = time.perf_counter()
        self._model = _load_cross_encoder(self._model_name, self._device, self._dtype)
        return time.perf_counter() - started

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def rerank(self, question: str, hits: list[SearchResult], top_k: int | None = None) -> list[SearchResult]:
        """给候选重新打分并排序，返回按精排分数降序的 hits。

        候选为空时直接返回空列表。打分失败时返回原始顺序（不改变分数），
        这样上层照常能出答案，只是没有精排。
        """
        if not hits:
            return []

        started = time.perf_counter()
        self._stats.calls += 1
        try:
            scores = self._score_pairs(question, hits)
        except Exception:
            self._stats.fallback += 1
            self._stats.latency_ms_total += (time.perf_counter() - started) * 1000
            return list(hits)

        self._stats.pairs_scored += len(hits)
        self._stats.latency_ms_total += (time.perf_counter() - started) * 1000

        ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
        ordered = [dataclasses.replace(hit, score=float(score)) for hit, score in ranked]
        return ordered[:top_k] if top_k else ordered

    def _score_pairs(self, question: str, hits: list[SearchResult]) -> list[float]:
        """把候选送进 cross-encoder，返回和 hits 一一对应的分数。"""
        if self._model is None:
            self.preload()
        pairs = [(question, hit.content) for hit in hits]
        raw = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            max_length=self._max_length,
            apply_softmax=False,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(x) for x in raw]


def _load_cross_encoder(model_name: str, device: str, dtype: str):
    """加载 cross-encoder，返回可直接 predict 的模型。

    无条件设 HF_HUB_OFFLINE=1：transformers 加载模型时不管权重是否已在本地缓存，
    都会先去 huggingface.co 发一串 HEAD 请求。外网受限时这次"探测"本身就可能
    比加载权重慢得多（见 RETRIEVAL_BASELINE.md 第 8 节），而且这个等待随时段
    波动，不引用具体数字——直接跳过是最省事的。
    """
    import os
    import torch

    if not os.environ.get("HF_HUB_OFFLINE"):
        os.environ["HF_HUB_OFFLINE"] = "1"

    from sentence_transformers import CrossEncoder

    resolved_device = _resolve_device(device)
    # trust_remote_code 是 CrossEncoder 的直接参数（实测 inspect.signature 确认），
    # model_kwargs 只透传给底层的 transformers 模型，放错位置会被当成
    # "模型不认识这个参数"。这里显式关掉：只跑官方开源权重，不执行仓库里的自定义代码。
    model_kwargs: dict[str, object] = {}
    if dtype == "fp16":
        model_kwargs["torch_dtype"] = torch.float16
    elif dtype == "fp32":
        model_kwargs["torch_dtype"] = torch.float32

    model = CrossEncoder(
        model_name, device=resolved_device, trust_remote_code=False, model_kwargs=model_kwargs
    )
    print(f"Reranker: {model_name} / device={resolved_device} / dtype={dtype}")
    return model


def _resolve_device(device: str) -> str:
    """把 auto 解析成实际设备。cuda 不可用时回落 cpu，不让启动直接崩。"""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
