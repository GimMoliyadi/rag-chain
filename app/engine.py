"""RAG 引擎组装：把 .env 配置变成一个可运行的 engine。

三个入口（ask.py / app.main / mcp_server.py）都需要同一套组装逻辑。
如果各写一份，迟早漂移——已经发生过一次：ask.py 那份漏传了 reranker 和
context_filter 的四个参数，导致 CLI 与 HTTP 两条路径行为不一致（CLI 用户开
RERANKER_ENABLED 不会生效）。所以现在只留这一份实现。
"""
from __future__ import annotations

from collections.abc import Callable

from .config import Settings
from .evidence import EvidenceGate
from .embedding import BaseEmbedding, get_embedding_client
from .knowledge_domains import load_router
from .llm import BaseLLM, get_llm_client
from .personal_context import PersonalContext, load_personal_context
from .rag import RAGEngine
from .reranker import CrossEncoderReranker
from .retriever import Retriever
from .vectorstore import ChromaStore


def build_engine(
    settings: Settings,
    *,
    embedding: BaseEmbedding | None = None,
    store: ChromaStore | None = None,
    llm: BaseLLM | None = None,
) -> tuple[RAGEngine, ChromaStore]:
    """按 .env 组装完整链路：embedding → 向量库 → 检索器 → Router → RAG 引擎。

    三个关键字参数允许注入替身，用途是给测试留的：用临时目录和假客户端构造
    自己的 engine，既不碰真实数据，也不需要真的加载模型。

    返回 (engine, store) 而不是只返回 engine：调用方要读 chunk 总数，并且退出时
    得 close store——store 内部握着 Chroma 的 SQLite handle，不关的话进程退出
    时会留下锁文件。
    """
    settings.validate_providers()

    embedding = embedding or get_embedding_client(settings)
    store = store or ChromaStore(settings.chroma_path_resolved)
    llm = llm or get_llm_client(settings)

    retriever = Retriever(embedding=embedding, store=store, default_top_k=settings.top_k)

    # Router 始终注入：规则术语表路由本身零成本，没有开关。
    router, allowed_domains = load_router(settings.knowledge_domains_path_resolved)

    # Evidence Gate 只在开关开启时构造，和 reranker 同一纪律：关着的时候
    # 一个字节资源都不该占。RAGEngine 只在开关开启时表现出新行为，
    # 关闭时与历史版本完全一致。
    evidence_gate = None
    if settings.evidence_gate_enabled:
        evidence_gate = EvidenceGate(llm, chunk_chars=settings.evidence_judge_chunk_chars)

    # Rescue 用独立的 Retriever：主检索器可能开着 diversity 的「每文档最多
    # N chunk」限额，而救援候选池必须不带这个限额，否则「答案在同文档第 3 个
    # chunk」这类漏检会被限额挤掉、救援失效。
    rescue_retriever = None
    if settings.evidence_gate_enabled and settings.evidence_rescue_enabled:
        rescue_retriever = Retriever(
            embedding=embedding, store=store, default_top_k=settings.evidence_rescue_k,
        )

    # Personal Context 每次 advisor 请求全量读取一次：文件小，而且它是「当前
    # 状态」，缓存会让用户在文件里改了内容却看不到生效。
    personal_context_loader: Callable[[], PersonalContext] | None = None
    if settings.personal_context_enabled:
        personal_context_path = settings.personal_context_path_resolved

        def _load_personal_context() -> PersonalContext:
            return load_personal_context(personal_context_path)

        personal_context_loader = _load_personal_context

    engine = RAGEngine(
        retriever=retriever,
        llm=llm,
        top_k=settings.top_k,
        vault_name=settings.vault_name,
        context_filter_enabled=settings.context_filter_enabled,
        filter_retrieve_top_k=settings.filter_retrieve_top_k,
        reranker=build_reranker(settings),
        reranker_candidate_top_k=settings.reranker_candidate_top_k,
        router=router,
        router_domains=allowed_domains,
        llm_router_fallback_enabled=settings.llm_router_fallback_enabled,
        evidence_gate=evidence_gate,
        evidence_rescue_enabled=settings.evidence_rescue_enabled,
        evidence_rescue_k=settings.evidence_rescue_k,
        rescue_retriever=rescue_retriever,
        personal_context_loader=personal_context_loader,
    )
    return engine, store


def build_reranker(settings: Settings) -> CrossEncoderReranker | None:
    """只有显式开启精排时才构造并加载模型，关着的时候一个字节显存都不占。

    同步加载而不是留到第一次问答上：加载一次约 7 秒（393 个分片），留到
    请求上会表现为「第一次问答莫名多等 7 秒」。
    """
    if not settings.reranker_enabled:
        return None
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        device=settings.reranker_device,
        dtype=settings.reranker_dtype,
        batch_size=settings.reranker_batch_size,
    )
    seconds = reranker.preload()
    print(f"Reranker loaded: {settings.reranker_model} in {seconds:.1f}s")
    return reranker
