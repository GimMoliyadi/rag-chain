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
from app.engine import build_engine
from app.evidence import STATUS_INSUFFICIENT
from app.personal_context import load_personal_context
from app.rag import RAGAnswer, RAGEngine

# 终端里引用来源只展示开头这么多字符，剩下的让用户自己打开文件看
SOURCE_PREVIEW_CHARS = 200

# 交互模式下输入这些词退出
EXIT_WORDS = {"exit", "quit", "q"}

# 终端分隔线宽度，跟 search.py 保持一致
LINE_WIDTH = 60


def ask_once(engine: RAGEngine, question: str, top_k: int, mode: str) -> None:
    """跑一次完整 RAG，把答案、路由决策、证据评审和引用来源打印到终端。"""
    started = time.perf_counter()
    print("正在检索知识库并生成回答…")
    result = engine.answer(question, top_k=top_k, mode=mode)
    elapsed = time.perf_counter() - started

    print("\n" + "=" * LINE_WIDTH)
    print(result.answer)
    print("=" * LINE_WIDTH)
    print_advisor_info(result)
    print_sources(result, elapsed)


def print_advisor_info(result: RAGAnswer) -> None:
    """打印 Router / Evidence Gate / 证据救援 / Personal Context / 各阶段耗时。

    打印而不是藏在 debug 日志里：CLI 的用途之一就是真实测试时人眼核对
    每个阶段发生了什么，延迟数字直接印出来才不会被质疑「掩盖开销」。
    """
    if result.route is not None:
        print(f"\n[router] mode={result.route.mode} domains={result.route.domains} "
              f"origin={result.route.origin} confidence={result.route.confidence:.2f}")
        print(f"[router] reason: {result.route.reason}")
    if result.personal_context is not None:
        info = result.personal_context
        state = (
            "、".join(info.get("available_sections") or [])
            if info.get("loaded") else (info.get("error") or info.get("reason") or "无内容")
        )
        print(f"[personal_context] 来源：{info.get('source_path')} 段：{state}")
    if result.evidence is not None:
        print(f"[evidence] status={result.evidence.status} confidence={result.evidence.confidence:.2f}"
              + ("（Judge 失败兜底，非真实评审）" if result.evidence.debug.get("judge_error") else ""))
        if result.evidence.knowledge_support is not None:
            print(f"[evidence] 知识库支撑={result.evidence.knowledge_support} "
                  f"个人状态支撑={result.evidence.personal_context_support}"
                  + ("（规则改判 insufficient→partial）" if result.evidence.debug.get("status_adjusted") else ""))
        print(f"[evidence] reason: {result.evidence.reason}")
    if result.timings:
        parts = [f"{name}={value}" for name, value in result.timings.items()]
        print("[timings] " + " | ".join(parts))


def print_sources(result: RAGAnswer, elapsed: float) -> None:
    """打印引用来源，编号和答案里的 [1][2] 一一对应。

    证据评审判定 insufficient 时，这些来源只是语义相近的候选，
    必须打上「不可作为答案依据」的标记，避免读者误当成支撑材料。
    """
    source_note = ""
    if result.evidence is not None and result.evidence.status == STATUS_INSUFFICIENT:
        source_note = "（语义相近资料，不足以作为答案依据）"
    print(f"\n引用 {result.used_chunks} 条{source_note} | 耗时 {elapsed:.1f} 秒")
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


def repl(engine: RAGEngine, top_k: int, mode: str) -> None:
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
        ask_once(engine, question, top_k, mode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Obsidian 知识库 RAG 问答：检索 + 生成 + 引用来源。"
        "只想看检索结果不调 LLM 请用 search.py。"
    )
    parser.add_argument("question", nargs="?", help="要问的问题；不填则进入交互模式")
    parser.add_argument("--k", type=int, default=None, help="检索条数，默认用 .env 里的 TOP_K")
    parser.add_argument(
        "--mode", choices=("normal", "advisor"), default="normal",
        help="advisor 模式读取 personal_context/context.yaml 作为「用户当前情况」",
    )
    args = parser.parse_args()

    if args.k is not None and args.k < 1:
        raise SystemExit("--k 必须是正整数")

    settings = Settings()
    top_k = args.k or settings.top_k

    # bge-m3 权重 2.17 GB，加载要十几秒。先给一行提示，
    # 否则终端卡在原地十几秒，看起来跟程序死掉没区别。
    print("正在加载知识库（约十几秒）…")
    engine, store = build_engine(settings)
    count = store.count()
    if count == 0:
        store.close()
        raise SystemExit("向量库是空的，先运行 python ingest.py 导入文档")
    print(f"向量库已有 {count} 个 chunk，检索条数 top_k={top_k}")
    if args.mode == "advisor":
        pc_path = settings.personal_context_path_resolved
        context = load_personal_context(pc_path)
        if context.has_content:
            print(f"mode=advisor，已读取 Personal Context：{pc_path}"
                  f"（{'、'.join(context.available_sections)}）")
        else:
            print(f"mode=advisor，但没有可用的 Personal Context（{pc_path}）——"
                  "回答时会明确说明缺少用户的当前情况")
    print()

    try:
        if args.question:
            ask_once(engine, args.question, top_k, args.mode)
        else:
            repl(engine, top_k, args.mode)
    finally:
        store.close()


if __name__ == "__main__":
    main()
