"""命令行问答入口：完整 RAG，检索 + 生成 + 引用来源。

和 search.py 的分工：search.py 只跑检索那一步，这个把检索结果拼进 Prompt
再调 LLM，返回带 [1][2] 引用的回答。检索结果不对就先跑 search.py 定位问题。

用法：
    python ask.py "合同违约需要承担什么责任？"   # 单次问答
    python ask.py --k 8 "什么是 MVP"            # 覆盖检索条数
    python ask.py                               # 交互模式，反复提问
"""
from __future__ import annotations

import argparse
import time

from app.config import Settings
from app.embedding import get_embedding_client
from app.llm import get_llm_client
from app.rag import RAGAnswer, RAGEngine
from app.retriever import Retriever
from app.vectorstore import ChromaStore

# 终端里引用来源只展示开头这么多字符，剩下的让用户自己打开文件看
SOURCE_PREVIEW_CHARS = 200

# 交互模式下输入这些词退出
EXIT_WORDS = {"exit", "quit", "q"}

# 终端分隔线宽度，跟 search.py 保持一致
LINE_WIDTH = 60


def build_engine(settings: Settings) -> tuple[RAGEngine, ChromaStore]:
    """按 .env 组装完整链路：embedding → 向量库 → 检索器 → RAG 引擎。

    一并返回 store 是让调用方能读 chunk 总数、退出时 close。store 内部握着
    Chroma 的 SQLite handle，不关的话进程退出时会留下锁文件。
    """
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    retriever = Retriever(embedding=embedding, store=store, default_top_k=settings.top_k)
    engine = RAGEngine(
        retriever=retriever,
        llm=get_llm_client(settings),
        top_k=settings.top_k,
        vault_name=settings.vault_name,
    )
    return engine, store


def ask_once(engine: RAGEngine, question: str, top_k: int) -> None:
    """跑一次完整 RAG，把答案和引用来源打印到终端。"""
    started = time.perf_counter()
    print("正在检索知识库并生成回答…")
    result = engine.answer(question, top_k=top_k)
    elapsed = time.perf_counter() - started

    print("\n" + "=" * LINE_WIDTH)
    print(result.answer)
    print("=" * LINE_WIDTH)
    print_sources(result, elapsed)


def print_sources(result: RAGAnswer, elapsed: float) -> None:
    """打印引用来源，编号和答案里的 [1][2] 一一对应。"""
    print(f"\n引用 {result.used_chunks} 条 | 耗时 {elapsed:.1f} 秒")
    for source in result.sources:
        heading = f" / {source.heading}" if source.heading else ""
        content = source.content
        preview = (
            content
            if len(content) <= SOURCE_PREVIEW_CHARS
            else content[:SOURCE_PREVIEW_CHARS] + "…"
        )
        print(f"\n[{source.index}] 相似度 {source.score:.4f}")
        print(f"    文档：{source.title}{heading}")
        print(f"    文件：{source.file_path}")
        print(f"    原文：{preview}")


def repl(engine: RAGEngine, top_k: int) -> None:
    """交互模式：反复提问，同一个 engine 只加载一次模型。

    如果每次提问都新起进程，bge-m3 那 2.17 GB 权重每问一次就要重新加载十几秒，
    所以交互式使用必须复用同一个 engine。
    """
    print("交互模式：直接输入问题回车；exit 或 Ctrl+D 退出。")
    while True:
        try:
            question = input("\n问题 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            return
        ask_once(engine, question, top_k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Obsidian 知识库 RAG 问答：检索 + 生成 + 引用来源。"
        "只想看检索结果不调 LLM 请用 search.py。"
    )
    parser.add_argument("question", nargs="?", help="要问的问题；不填则进入交互模式")
    parser.add_argument("--k", type=int, default=None, help="检索条数，默认用 .env 里的 TOP_K")
    args = parser.parse_args()

    if args.k is not None and args.k < 1:
        raise SystemExit("--k 必须是正整数")

    settings = Settings()
    settings.validate_providers()
    top_k = args.k or settings.top_k

    # bge-m3 权重 2.17 GB，加载要十几秒。先给一行提示，
    # 否则终端卡在原地十几秒，看起来跟程序死掉没区别。
    print("正在加载知识库（约十几秒）…")
    engine, store = build_engine(settings)
    count = store.count()
    if count == 0:
        store.close()
        raise SystemExit("向量库是空的，先运行 python ingest.py 导入文档")
    print(f"向量库已有 {count} 个 chunk，检索条数 top_k={top_k}\n")

    try:
        if args.question:
            ask_once(engine, args.question, top_k)
        else:
            repl(engine, top_k)
    finally:
        store.close()


if __name__ == "__main__":
    main()
