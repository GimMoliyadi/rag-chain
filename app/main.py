"""FastAPI 入口。

用工厂函数 create_app() 构造 app，原因有两个：
    1. 测试可以用临时的数据库和向量库构造自己的 app 实例，不污染真实数据
    2. 启动权交给 uvicorn，而不是在 import 时就构造——否则任何
       `import app.main`（包括 pytest 收集测试时）都会真的去加载 embedding 模型

启动方式（注意是 --factory，调的是 create_app 这个函数）：
    uvicorn app.main:create_app --factory --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import Settings
from .database import connect, get_document, init_db, list_documents
from .embedding import BaseEmbedding, get_embedding_client
from .ingest import IngestStats, run_ingest
from .llm import BaseLLM, get_llm_client
from .rag import RAGEngine
from .retriever import EmbeddingMismatchError, Retriever
from .vectorstore import ChromaStore


def create_app(
    settings: Settings | None = None,
    embedding: BaseEmbedding | None = None,
    store: ChromaStore | None = None,
    llm: BaseLLM | None = None,
) -> FastAPI:
    """构造一个 FastAPI 应用。所有依赖都在这里组装，端点里不再 new 东西。"""
    settings = settings or Settings()
    settings.validate_providers()

    # 组装依赖链：embedding/store 是底层，retriever 和 rag 建立在其上
    embedding = embedding or get_embedding_client(settings)
    store = store or ChromaStore(settings.chroma_path_resolved)
    llm = llm or get_llm_client(settings)
    conn = connect(settings.db_path_resolved)
    init_db(conn)

    retriever = Retriever(embedding=embedding, store=store, default_top_k=settings.top_k)
    engine = RAGEngine(retriever=retriever, llm=llm, top_k=settings.top_k, vault_name=settings.vault_name)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        conn.close()

    app = FastAPI(
        title="Obsidian RAG Backend",
        description="Obsidian 知识库 → SQLite → Chunk → Embedding → ChromaDB → RAG 问答",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(EmbeddingMismatchError)
    async def _embedding_mismatch_handler(request: Request, exc: EmbeddingMismatchError):
        """不注册这个处理器的话，FastAPI 会把它当成普通 500，只回
        {"detail": "Internal Server Error"}，把"需要重建向量库"这条关键信息吞掉。"""
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    # 把依赖挂到 app.state，端点直接取，不用层层传参
    app.state.settings = settings
    app.state.conn = conn
    app.state.embedding = embedding
    app.state.store = store
    app.state.llm = llm
    app.state.retriever = retriever
    app.state.rag = engine

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    """集中注册所有路由，方便一眼看完接口清单。"""

    @app.get("/health")
    def health():
        """健康检查：确认服务活着，顺便报一下库里的规模。"""
        return {
            "status": "ok",
            "documents": _doc_count(app),
            "chunks": app.state.store.count(),
            "embedding_model": app.state.embedding.model_name,
            "llm_model": app.state.llm.model_name,
        }

    @app.get("/api/documents")
    def list_docs(limit: int = Query(default=50, ge=1, le=500), offset: int = Query(default=0, ge=0)):
        """分页列出已入库的文档。"""
        return {"documents": list_documents(app.state.conn, limit=limit, offset=offset)}

    @app.get("/api/documents/{document_id}")
    def read_doc(document_id: int):
        """查看单个文档的元信息和内容。"""
        doc = get_document(app.state.conn, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"文档 {document_id} 不存在")
        doc.pop("content_hash", None)  # 增量入库用的内部字段，不暴露给前端
        return doc

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=1, description="检索关键词"), k: int = Query(default=5, ge=1, le=20)):
        """只检索不调用 LLM，用来单独验证检索效果。"""
        hits = app.state.retriever.search(q, top_k=k)
        return {
            "query": q,
            "top_k": k,
            "count": len(hits),
            "results": [
                {
                    "rank": i + 1,
                    "score": h.score,
                    "title": h.title,
                    "file_path": h.file_path,
                    "heading": h.heading,
                    "source": h.source,
                    "chunk_index": h.chunk_index,
                    "content": h.content,
                }
                for i, h in enumerate(hits)
            ],
        }

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        """完整 RAG 问答：检索 + 生成 + 引用来源。"""
        result = app.state.rag.answer(request.question, top_k=request.top_k)
        return {
            "answer": result.answer,
            "used_chunks": result.used_chunks,
            "sources": [
                {
                    "index": s.index,
                    "title": s.title,
                    "file_path": s.file_path,
                    "heading": s.heading,
                    "score": s.score,
                    "chunk": s.content,
                }
                for s in result.sources
            ],
        }

    @app.post("/api/ingest")
    def ingest(request: IngestRequest):
        """触发入库。path_filter 只导某个子目录，limit 限制条数，都用于调试。"""
        vault = app.state.settings.vault_path_resolved
        if not vault.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"VAULT_PATH 不是有效目录：{vault}，请检查 .env 里的 VAULT_PATH",
            )
        stats = run_ingest(
            app.state.settings,
            app.state.embedding,
            app.state.store,
            path_filter=request.path_filter,
            limit=request.limit,
        )
        return _stats_payload(stats)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int | None = Field(default=None, ge=1, le=20, description="检索条数，不传用配置里的 TOP_K")


class IngestRequest(BaseModel):
    path_filter: str | None = Field(default=None, description="只导入路径包含该片段的文件，如 '01-精益创业2.0'")
    limit: int | None = Field(default=None, ge=1, le=5000, description="最多导入多少篇，用于小规模试跑")


def _doc_count(app: FastAPI) -> int:
    return app.state.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


def _stats_payload(stats: IngestStats) -> dict:
    """把 IngestStats 转成 API 响应格式。"""
    return {
        "scanned": stats.scanned,
        "added": stats.added,
        "updated": stats.updated,
        "unchanged": stats.unchanged,
        "failed": stats.failed,
        "chunks_written": stats.chunks_written,
        "errors": stats.errors,
    }


# 注意：这里故意不放 `app = create_app()`。
# create_app 会真的加载 embedding 模型（bge-m3 权重 2.17 GB），
# 写在模块末尾会让任何 import 都触发一次下载和加载。所以交给 uvicorn --factory 调。
