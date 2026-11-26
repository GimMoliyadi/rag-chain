"""只检索入口的共享胶水。

/api/search、search.py 和 MCP 的 search 工具都属于“只看检索结果、不调 LLM”的
排查入口。它们不应该自己绕过 RAGEngine 重新调 Retriever，否则一打开
context filter / reranker / diversity 之类开关，就会出现：search 看到的片段
和 /api/chat 真正进入 Prompt 的片段不一致。

这个模块只复用 RAGEngine 的检索整理阶段：路由 → 检索候选 → 去重 → 可选正文清洗。
它刻意不走 Evidence Gate、Evidence Rescue 和 LLM 生成，因为这些属于完整问答链路。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .rag import RAGEngine
from .retriever import SearchResult
from .router import RouteDecision


@dataclass
class SearchChainResult:
    """一次只检索调用的结果。"""

    hits: list[SearchResult]
    contents: dict[int, str] | None = None
    route: RouteDecision | None = None
    timings: dict = field(default_factory=dict)


def collect_search_results(
    engine: RAGEngine,
    question: str,
    top_k: int,
    *,
    routed: bool = False,
    domains: list[str] | None = None,
) -> SearchChainResult:
    """按 RAGEngine 的检索整理规则返回最终候选，不调用 LLM。

    routed=True 时走和 chat 相同的路由入口；domains 手动指定时跳过路由，
    语义与完整 RAGEngine.answer(domains=...) 一致。

    这里保留在 RAGEngine 边界外的薄胶水里：HTTP / CLI / MCP 只依赖这个
    函数，不再各自手写检索流程。若未来 RAGEngine 暴露正式 public 方法，
    只需要改这一处适配层。
    """
    if routed and domains is not None:
        raise ValueError("routed=True 与手动 domains 不能同时使用")

    timings: dict = {}
    route = engine._route_if_needed(question, domains, timings) if routed else None
    effective_domains = (route.domains or None) if route is not None else domains
    hits, contents = engine._collect(question, top_k, effective_domains)
    return SearchChainResult(hits=hits, contents=contents, route=route, timings=timings)


def content_for_hit(hit: SearchResult, contents: dict[int, str] | None) -> str:
    """返回 search 入口应该展示的正文。

    context filter 开启时，RAGEngine._collect 会返回清洗后的正文；没开启时
    contents 为 None，此时展示原始 chunk。
    """
    if contents is None:
        return hit.content
    return contents.get(hit.chunk_id, hit.content)
