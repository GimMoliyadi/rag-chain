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
from app.engine import build_engine
from app.llm import MockLLM
from app.search_chain import collect_search_results, content_for_hit


def main() -> None:
    parser = argparse.ArgumentParser(description="在知识库向量索引里按综合相关性检索（不调用 LLM）")
    parser.add_argument("question", help="要检索的问题或关键词")
    parser.add_argument("--k", type=int, default=None, help="返回条数，默认用 .env 里的 TOP_K")
    parser.add_argument("--domains", default=None, help="只在这些知识域里检索，逗号分隔（手动指定，跳过 Router）")
    parser.add_argument("--router", action="store_true", help="先用 QueryRouter 判断知识域，再 scoped 检索")
    args = parser.parse_args()
    if args.router and args.domains:
        raise SystemExit("--router 和 --domains 只能二选一：一个是路由决定域，一个是手动指定域")

    settings = Settings()
    # search.py 只排查检索，不应该因为真实 LLM 凭证缺失而启动失败；
    # build_engine 仍然复用同一套 RAG 装配，但 LLM 注入 MockLLM，避免调外部服务。
    settings.llm_provider = "mock"
    store = None
    try:
        engine, store = build_engine(settings, llm=MockLLM())
        total = store.count()
        print(f"向量库已有 {total} 个 chunk")
        if total == 0:
            raise SystemExit("向量库是空的，先运行 python ingest.py 导入文档")

        domains = args.domains.split(",") if args.domains else None
        search_result = collect_search_results(
            engine,
            args.question,
            args.k or settings.top_k,
            routed=args.router,
            domains=domains,
        )

        if args.router:
            decision = search_result.route
            print("\n路由决定:")
            print(json.dumps(
                {
                    "mode": decision.mode if decision else "global",
                    "domains": decision.domains if decision else [],
                    "confidence": decision.confidence if decision else 0,
                    "reason": decision.reason if decision else "未启用 Router",
                },
                ensure_ascii=False, indent=2,
            ))

        retriever = engine.retriever
        if retriever.last_debug.get("domain_empty_fallback"):
            print("domain_empty_fallback = true（指定域 0 chunk，已回退全库检索）")
        hits = search_result.hits
        if not hits:
            print("没有检索到任何结果。")
            return

        print(f"\n问题：{args.question}\n" + "=" * 60)
        for i, hit in enumerate(hits, start=1):
            content = content_for_hit(hit, search_result.contents)
            preview = content if len(content) <= 300 else content[:300] + "…"
            print(f"\nTop {i}")
            print(f"原始向量相似度 : {hit.score:.4f}")
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
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    main()
