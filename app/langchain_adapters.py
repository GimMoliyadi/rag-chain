"""把现有 Embedding provider 接到 LangChain 官方 Embeddings 接口。

不重新加载模型，也不重算旧向量；保留本地 bge-m3 的精度/截断/指纹策略，
并且正确保留查询侧 instruction prefix（有些模型要求）。
新的检索策略或 LangChain VectorStore 可以直接使用此适配器。
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from .embedding import BaseEmbedding


class LangChainEmbeddingsAdapter(Embeddings):
    """LangChain Embeddings 接口与项目现有 provider 之间的双向语义桥接。"""

    def __init__(self, backend: BaseEmbedding) -> None:
        self.backend = backend

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.backend.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        # 必须调用 backend.embed_query，不能直接 embed([text])；
        # 否则有 query prefix 的 embedding 模型检索会静默退化。
        return self.backend.embed_query(text)
