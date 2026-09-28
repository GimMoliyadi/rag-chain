"""检索器：把一个问题变成 Top-K 相关 chunk。

这条链路是 RAG 的"R"（Retrieval）部分，也是整个系统里最该先验证的一环——
检索错了，LLM 再强也答不对。所以 search.py 可以独立跑它，不用碰 LLM。
"""
from __future__ import annotations

from dataclasses import dataclass

from .embedding import BaseEmbedding
from .vectorstore import ChromaStore


@dataclass
class SearchResult:
    """一条检索命中。score 是相似度，0~1，越大越相关。"""

    score: float
    chunk_id: int
    document_id: int
    chunk_index: int
    title: str
    file_path: str
    heading: str
    source: str
    content: str
    domain: str = ""         # 知识域（阶段二 metadata 回填之前是空串）
    source_type: str = ""


class EmbeddingMismatchError(RuntimeError):
    """向量库里的向量不是当前 embedding 模型算出来的。

    不同模型的向量不在同一个空间里，混着用算出来的相似度是随机噪声，
    而且不会报任何错——这正是最危险的一种失败方式。所以这里必须显式抛错。
    """


# 多样性开启时先取多少条候选。候选池太小，文档限额会把结果压到不足 Top-K；
# 20 是"每篇最多 2 条时，Top-5 还有 4 倍冗余"的量级，不是精调值。
DIVERSITY_CANDIDATE_K = 20


class Retriever:
    """用 embedding + 向量库做相似度检索。"""

    def __init__(
        self,
        embedding: BaseEmbedding,
        store: ChromaStore,
        default_top_k: int = 5,
        diversity_enabled: bool = False,
        max_chunks_per_document: int = 2,
    ):
        self._embedding = embedding
        self._store = store
        self._default_top_k = default_top_k
        self._diversity_enabled = diversity_enabled
        self._max_chunks_per_document = max_chunks_per_document
        # 最近一次 search 的调试信息。search 的返回值必须保持 list[SearchResult]
        # 不变（eval / API / 测试都依赖它），domain_empty_fallback 这类过程信息
        # 只能挂在旁边，不能改返回类型。
        self.last_debug: dict = {}

    def search(self, question: str, top_k: int | None = None, domains: list[str] | None = None) -> list[SearchResult]:
        """检索与 question 最相似的 top_k 条 chunk，按相似度从高到低排列。

        domains=None 或 [] 时是全库检索，行为与历史版本完全一致（回归基线）；
        domains=[...] 时用 Chroma 的 metadata 过滤只在指定知识域里找。
        指定的域在库里一条 chunk 都没有时（如 fiction），不做空返回，
        回退全库检索，并记录 domain_empty_fallback=True。
        """
        k = top_k or self._default_top_k
        count = self._store.count()
        if count == 0:
            self.last_debug = {"scoped": False, "domains": [], "domain_empty_fallback": False,
                               "diversity_applied": False, "candidates": 0, "returned": 0}
            return []

        # 库非空时才检查指纹：还没 ingest 过时空库本来就没有可比对的指纹
        _check_index_matches(self._embedding, self._store)

        scoped = bool(domains)
        where = {"domain": {"$in": list(domains)}} if scoped else None
        candidate_k = DIVERSITY_CANDIDATE_K if self._diversity_enabled else k
        question_vector = self._embedding.embed_query(question)
        rows = self._store.query(question_vector, n_results=min(candidate_k, count), where=where)

        fallback = False
        if scoped and not rows:
            # 指定的域是 0 chunk（比如 fiction 只有 .txt 源、从未入库）。
            # 过滤本身没错，错的是"拿一个空结果当答案"，所以退回全库。
            fallback = True
            rows = self._store.query(question_vector, n_results=min(candidate_k, count))

        if self._diversity_enabled:
            rows = _cap_per_document(rows, k, self._max_chunks_per_document)

        hits = [_to_hit(row) for row in rows]
        self.last_debug = {
            "scoped": scoped,
            "domains": list(domains) if domains else [],
            "domain_empty_fallback": fallback,
            "diversity_applied": self._diversity_enabled,
            "candidates": len(rows),
            "returned": len(hits),
        }
        return hits


def _cap_per_document(rows: list[dict], top_k: int, max_per_document: int) -> list[dict]:
    """同文档 chunk 限额：每个 document_id 最多保留 max_per_document 条，直到凑满 top_k。

    解决的实测问题（baseline #22）：长章节切成多个 chunk 后每段都高分，
    Top-5 被"同一段话的 5 个切片"占满。限额后名额让给次优文档，
    喂给 LLM 的 context 才有多样性。不做 MMR：先用最简单的文档限额验证
    它能不能解决真实失败案例，不够再谈更复杂的。
    """
    kept: list[dict] = []
    per_document: dict[int, int] = {}
    for row in rows:
        document_id = int(row["document_id"])
        if per_document.get(document_id, 0) >= max_per_document:
            continue
        per_document[document_id] = per_document.get(document_id, 0) + 1
        kept.append(row)
        if len(kept) >= top_k:
            break
    return kept


def _to_hit(row: dict) -> SearchResult:
    return SearchResult(
        score=_to_similarity(row["distance"]),
        chunk_id=int(row["chunk_id"]),
        document_id=int(row["document_id"]),
        chunk_index=int(row["chunk_index"]),
        title=str(row.get("title", "")),
        file_path=str(row.get("file_path", "")),
        heading=str(row.get("heading", "")),
        source=str(row.get("source", "")),
        content=row["content"],
        domain=str(row.get("domain", "")),
        source_type=str(row.get("source_type", "")),
    )


def _to_similarity(distance: float) -> float:
    """把余弦距离转成相似度。

    余弦距离 range 是 [0, 2]，相似度 = 1 - distance，范围 [-1, 1]。
    这里把负值截断到 0，方便展示和做阈值过滤。
    """
    return max(0.0, min(1.0, 1.0 - float(distance)))


_REBUILD_HINT = "请运行 `python ingest.py --rebuild-vectors` 重建向量库。"


def _check_index_matches(embedding: BaseEmbedding, store: ChromaStore) -> None:
    """确认向量库是用当前模型建的。对不上就报错，绝不静默继续。

    典型事故：昨天用 bge-small 建的索引，今天把 .env 换成 bge-m3，
    但 Chroma 里的向量没动。这时检索数字还在返回，看着一切正常，
    实际上每一组相似度都是两个无关向量空间的巧合重合。
    """
    info = store.read_index_info()
    if info is None:
        raise EmbeddingMismatchError(
            "Embedding model mismatch. Please rebuild vector index. "
            f"向量库 {store.count()} 条向量没有模型指纹（可能是旧版代码建的），"
            f"当前模型是 {embedding.model_name}。{_REBUILD_HINT}"
        )
    if info != embedding.fingerprint:
        raise EmbeddingMismatchError(
            "Embedding model mismatch. Please rebuild vector index. "
            f"向量库是用 {info.get('model')} (dim={info.get('dimensions')}) 建的，"
            f"当前模型是 {embedding.model_name} (dim={embedding.dimensions})。{_REBUILD_HINT}"
        )
