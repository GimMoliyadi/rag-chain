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


class EmbeddingMismatchError(RuntimeError):
    """向量库里的向量不是当前 embedding 模型算出来的。

    不同模型的向量不在同一个空间里，混着用算出来的相似度是随机噪声，
    而且不会报任何错——这正是最危险的一种失败方式。所以这里必须显式抛错。
    """


class Retriever:
    """用 embedding + 向量库做相似度检索。"""

    def __init__(self, embedding: BaseEmbedding, store: ChromaStore, default_top_k: int = 5):
        self._embedding = embedding
        self._store = store
        self._default_top_k = default_top_k

    def search(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        """检索与 question 最相似的 top_k 条 chunk，按相似度从高到低排列。"""
        k = top_k or self._default_top_k
        count = self._store.count()
        if count == 0:
            return []

        # 库非空时才检查指纹：还没 ingest 过时空库本来就没有可比对的指纹
        _check_index_matches(self._embedding, self._store)

        question_vector = self._embedding.embed_query(question)
        rows = self._store.query(question_vector, n_results=min(k, count))

        hits: list[SearchResult] = []
        for row in rows:
            hits.append(
                SearchResult(
                    score=_to_similarity(row["distance"]),
                    chunk_id=int(row["chunk_id"]),
                    document_id=int(row["document_id"]),
                    chunk_index=int(row["chunk_index"]),
                    title=str(row.get("title", "")),
                    file_path=str(row.get("file_path", "")),
                    heading=str(row.get("heading", "")),
                    source=str(row.get("source", "")),
                    content=row["content"],
                )
            )
        return hits


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
