"""独立检索命令，不调用 LLM。

先跑这个确认检索正常，再去调 LLM。检索错了后面全错，这是最该先验证的一环。

用法：
    python search.py "合同违约需要承担什么责任？"
    python search.py "什么是 MVP" --k 10
"""
from __future__ import annotations

import argparse

from app.config import Settings
from app.embedding import get_embedding_client
from app.retriever import Retriever
from app.vectorstore import ChromaStore


def main() -> None:
    parser = argparse.ArgumentParser(description="在知识库向量索引里做相似度检索（不调用 LLM）")
    parser.add_argument("question", help="要检索的问题或关键词")
    parser.add_argument("--k", type=int, default=None, help="返回条数，默认用 .env 里的 TOP_K")
    args = parser.parse_args()

    settings = Settings()
    settings.validate_providers()

    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    retriever = Retriever(embedding=embedding, store=store, default_top_k=settings.top_k)

    total = store.count()
    print(f"向量库已有 {total} 个 chunk（embedding: {embedding.model_name}）")
    if total == 0:
        raise SystemExit("向量库是空的，先运行 python ingest.py 导入文档")

    hits = retriever.search(args.question, top_k=args.k or settings.top_k)
    if not hits:
        print("没有检索到任何结果。")
        return

    print(f"\n问题：{args.question}\n" + "=" * 60)
    for i, hit in enumerate(hits, start=1):
        content = hit.content
        preview = content if len(content) <= 300 else content[:300] + "…"
        print(f"\nTop {i}")
        print(f"相似度 : {hit.score:.4f}")
        print(f"文档   : {hit.title}")
        print(f"文件   : {hit.file_path}")
        print(f"章节   : {hit.heading or '(无标题)'}")
        print(f"内容   :\n{preview}")
        print("-" * 60)


if __name__ == "__main__":
    main()
