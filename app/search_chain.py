"""HTTP、CLI 和 MCP 的共享检索与正文展示入口。"""
from __future__ import annotations

from .rag import RAGEngine, SearchChainResult
from .retriever import SearchResult


def collect_search_results(
    engine: RAGEngine,
    question: str,
    top_k: int,
    *,
    routed: bool = False,
    domains: list[str] | None = None,
) -> SearchChainResult:
    """通过 RAGEngine 的公开接口取得整理后的候选。"""
    return engine.retrieve(question, top_k, routed=routed, domains=domains)


def content_for_hit(hit: SearchResult, contents: dict[int, str] | None) -> str:
    """优先展示清洗后的正文，缺少清洗结果时展示原文。"""
    if contents is None:
        return hit.content
    return contents.get(hit.chunk_id, hit.content)
