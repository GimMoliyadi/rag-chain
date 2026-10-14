"""独立检索命令，不调用 LLM。

先跑这个确认检索正常，再去调 LLM。检索错了后面全错，这是最该先验证的一环。

用法：
    python search.py "合同违约需要承担什么责任？"
    python search.py "什么是 MVP" --k 10
    python search.py "纳什均衡为什么存在？" --router          # 先路由再 scoped 检索
    python search.py "纳什均衡为什么存在？" --domains game_theory,han_li_framework
"""
from __future__ import annotations

import argparse
import json

from app.config import Settings
from app.database import connect, get_chunks_for_reindex
from app.hybrid_retrieval import HybridRetriever
from app.embedding import get_embedding_client
from app.knowledge_domains import load_router_terms
from app.retriever import Retriever
from app.router import QueryRouter
from app.vectorstore import ChromaStore


def main() -> None:
    parser = argparse.ArgumentParser(description="在知识库向量索引里做相似度检索（不调用 LLM）")
    parser.add_argument("question", help="要检索的问题或关键词")
    parser.add_argument("--k", type=int, default=None, help="返回条数，默认用 .env 里的 TOP_K")
    parser.add_argument("--domains", default=None, help="只在这些知识域里检索，逗号分隔（手动指定，跳过 Router）")
    parser.add_argument("--router", action="store_true", help="先用 QueryRouter 判断知识域，再 scoped 检索")
    parser.add_argument("--hybrid", action="store_true", help="实验性：BM25 + Dense + RRF（输出的是相对融合排名分数）")
    args = parser.parse_args()
    if args.router and args.domains:
        raise SystemExit("--router 和 --domains 只能二选一：一个是路由决定域，一个是手动指定域")

    settings = Settings()
    settings.validate_providers()

    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    retriever = Retriever(
        embedding=embedding,
        store=store,
        default_top_k=settings.top_k,
        diversity_enabled=settings.diversity_enabled,
        max_chunks_per_document=settings.max_chunks_per_document,
    )

    total = store.count()
    print(f"向量库已有 {total} 个 chunk（embedding: {embedding.model_name}）")
    if total == 0:
        raise SystemExit("向量库是空的，先运行 python ingest.py 导入文档")

    if args.hybrid:
        conn = connect(settings.db_path_resolved)
        try:
            lexical_rows = get_chunks_for_reindex(conn)
        finally:
            conn.close()
        if not lexical_rows:
            raise SystemExit("SQLite 中没有可供 BM25 检索的 chunk，请先运行 python ingest.py")
        retriever = HybridRetriever(retriever, lexical_rows)
        print("实验模式：Hybrid BM25 + Dense + RRF（分数不是余弦相似度）")

    domains = args.domains.split(",") if args.domains else None
    decision = None
    if args.router:
        router = QueryRouter(load_router_terms(settings.knowledge_domains_path_resolved))
        decision = router.route(args.question)
        print("\n路由决定:")
        print(json.dumps(
            {"mode": decision.mode, "domains": decision.domains,
             "confidence": decision.confidence, "reason": decision.reason},
            ensure_ascii=False, indent=2,
        ))
        # global 时 domains 保持 None，走的是和不用 Router 完全相同的全库路径
        domains = decision.domains or None

    hits = retriever.search(args.question, top_k=args.k or settings.top_k, domains=domains)
    if retriever.last_debug.get("domain_empty_fallback"):
        print("domain_empty_fallback = true（指定域 0 chunk，已回退全库检索）")
    if not hits:
        print("没有检索到任何结果。")
        return

    print(f"\n问题：{args.question}\n" + "=" * 60)
    for i, hit in enumerate(hits, start=1):
        content = hit.content
        preview = content if len(content) <= 300 else content[:300] + "…"
        print(f"\nTop {i}")
        label = "RRF分数" if args.hybrid else "相似度"
        print(f"{label} : {hit.score:.4f}")
        print(f"文档   : {hit.title}")
        print(f"文件   : {hit.file_path}")
        print(f"领域   : {hit.domain} / {hit.source_type}")
        print(f"章节   : {hit.heading or '(无标题)'}")
        print(f"内容   :\n{preview}")
        print("-" * 60)

    debug = retriever.last_debug
    distinct_docs = len({hit.document_id for hit in hits})
    print(f"\n[debug] Top-{len(hits)} 覆盖 {distinct_docs} 篇独立文档 | 候选 {debug['candidates']} 条"
          f" | diversity={'on' if debug['diversity_applied'] else 'off'}")


if __name__ == "__main__":
    main()
