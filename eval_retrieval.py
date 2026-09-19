"""检索质量评估。

单独成脚本而不挂在 FastAPI 上的原因：评估要反复对比不同配置
（原始检索 / +context filter / +reranker），每次都重启服务太重，
而且服务进程里的模型加载会占用显存、干扰延迟测量。这里直接走
Settings -> embedding -> ChromaStore -> Retriever 的同一套依赖链，
拿到的检索结果和线上完全一致，只是没有 HTTP 那一层。

指标口径（题目级，跨题取均值）：
    Recall@k     = 该题 expected_documents 里只要有任意一篇落在 Top-k，记 1，否则 0
    recall_all@k = expected_documents 全部落在 Top-k 才记 1（多文档题的真实难点）
    MRR@5        = 第一篇命中文档的排名倒数；5 名之外才命中记 0

expected_documents 为空的题（category=unknown）不计入上面任何指标，
单独列为覆盖度检查——它们考的是"敢不敢说不知道"，由 eval_rag.py 负责。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from app.config import Settings
from app.context_filter import ContextSelection, prepare_context
from app.embedding import get_embedding_client
from app.reranker import CrossEncoderReranker
from app.retriever import Retriever, SearchResult
from app.vectorstore import ChromaStore

CATEGORIES = ("direct", "paraphrase", "multi_chunk", "hard_negative", "unknown")
METRIC_KS = (1, 3, 5)
MRR_CUTOFF = 5

# recall_all@k 的截断点，和 MRR 用同一个窗口
ALL_K = 5

# 每个 CLI 模式对应实际跑哪几组。保留 "both" 是为了不让
# RETRIEVAL_BASELINE.md 第 1 节里记的命令失效。
MODE_GROUPS = {
    "raw": ("raw",),
    "filter": ("filter",),
    "rerank": ("rerank",),
    "filter_rerank": ("filter_rerank",),
    "both": ("raw", "filter"),
    "all4": ("raw", "filter", "rerank", "filter_rerank"),
}

# 哪些模式需要精排模型（决定要不要花那 7 秒加载权重）
_MODES_NEEDING_RERANKER = ("rerank", "filter_rerank")


def _build_reranker(settings: Settings) -> CrossEncoderReranker | None:
    """构造并加载精排模型，返回加载耗时供报告记录。"""
    started = time.perf_counter()
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        device=settings.reranker_device,
        dtype=settings.reranker_dtype,
        batch_size=settings.reranker_batch_size,
    )
    reranker.preload()
    return reranker


def _describe_mode(mode: str, top_k: int, retrieve_top_k: int, candidate_top_k: int, model: str) -> str:
    """这组模式实际做了什么，写成一行：报告里看不出差别就是失败。"""
    parts = [f"top_k={top_k}"]
    if mode in ("filter", "filter_rerank"):
        parts.append(f"filter_pool={retrieve_top_k}")
    if mode in _MODES_NEEDING_RERANKER:
        parts.append(f"rerank_pool={candidate_top_k}")
        parts.append(f"model={model}")
    return "  ".join(parts)


def _norm_path(path: str) -> str:
    """路径归一化：反斜杠转正斜杠、去首尾空白、转小写。

    JSON 里的 expected_documents 是人工填的，Windows 上很容易写成
    带反斜杠的形式；归一化之后比较才不会因为分隔符差异误判成 0。
    """
    return path.replace("\\", "/").strip().lower()


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list):
        raise ValueError(f"题目文件顶层必须是数组：{path}")
    return questions


def select_questions(questions: list[dict], include_unverified: bool) -> tuple[list[dict], list[dict], int, int]:
    """分成可计分的题、只查覆盖度的题，并报告被跳过的数量。"""
    scorable: list[dict] = []
    coverage: list[dict] = []
    skipped_unverified = 0
    skipped_bad_category = 0
    for q in questions:
        if q.get("category") not in CATEGORIES:
            skipped_bad_category += 1
            continue
        if not include_unverified and not q.get("verified", False):
            skipped_unverified += 1
            continue
        (scorable if q.get("expected_documents") else coverage).append(q)
    return scorable, coverage, skipped_unverified, skipped_bad_category


def retrieve(
    retriever: Retriever,
    question: str,
    top_k: int,
    mode: str,
    retrieve_top_k: int,
    reranker: CrossEncoderReranker | None = None,
    candidate_top_k: int = 0,
) -> tuple[list[SearchResult], ContextSelection | None]:
    """一次检索。返回命中列表，以及 filter 相关模式下的过滤明细。

    mode 里带 rerank 就先用更大的候选池精排，带 filter 就在精排之后再过滤，
    顺序和 app/rag.py::_collect 一致，这样 A/B 数字和线上链路对得上。

    这里刻意不做 chunk_id 去重：raw 模式从来没做过，baseline 也是不
    去重的数字，只在其它模式上去重会让组间比较带上系统性偏差。
    """
    if mode == "raw":
        return retriever.search(question, top_k=top_k), None

    # 用模式名而不是"模型是否在手上"来决定要不要精排：多组 A/B 时精排模型会加载一次
    # 然后传给所有组，只看 reranker is not None 会让 filter 组偷偷也精排，指标全部作废。
    use_rerank = mode in _MODES_NEEDING_RERANKER
    use_filter = mode in ("filter", "filter_rerank")
    pool_size = top_k
    if use_filter:
        pool_size = max(pool_size, retrieve_top_k)
    if use_rerank:
        pool_size = max(pool_size, candidate_top_k)

    pool = retriever.search(question, top_k=pool_size)
    if use_rerank and reranker is not None:
        pool = reranker.rerank(question, pool)

    if not use_filter:
        return pool[:top_k], None
    selection = prepare_context(pool, top_k=top_k)
    return [chunk.hit for chunk in selection.selected], selection


def _hit_ranks(expected_norm: list[str], hits: list[SearchResult]) -> list[int | None]:
    """每个 expected 文档在 hits 里的最低排名（1-based），没检索到记 None。

    同一篇文档可能占多个 chunk 位置，取最低那个，反映"这篇文档最早出现在第几位"。
    """
    ranks: list[int | None] = []
    for exp in expected_norm:
        ranks.append(next((i + 1 for i, h in enumerate(hits) if _norm_path(h.file_path) == exp), None))
    return ranks


def _keyword_hit(question: dict, hits: list[SearchResult]) -> bool:
    """Top-5 的正文里是否出现过任一 expected keyword。

    这是失败分析用的辅助信号：能区分"完全没检索到相关段落"和
    "检索到了内容但落错了文档"，两种失败的修法完全不同。
    """
    pool = "\n".join(h.content for h in hits[:ALL_K])
    return any(kw in pool for kw in question.get("expected_keywords", []))


def score_question(question: dict, hits: list[SearchResult], latency_ms: float) -> dict:
    """算一题的全部指标。"""
    expected_norm = [_norm_path(p) for p in question["expected_documents"]]
    ranks = _hit_ranks(expected_norm, hits)
    found = [r for r in ranks if r is not None]
    first = min(found) if found else None

    result: dict = {
        "id": question.get("id"),
        "question": question["question"],
        "category": question["category"],
        "expected_documents": question["expected_documents"],
        "ranks": ranks,
        "first_rank": first,
        "recall_all_at_5": 1 if found and all(r <= ALL_K for r in found) else 0,
        "mrr5": round(1.0 / first, 4) if first is not None and first <= MRR_CUTOFF else 0.0,
        "latency_ms": round(latency_ms, 1),
        "keyword_hit": _keyword_hit(question, hits),
        "top5": [
            {
                "rank": i + 1,
                "score": round(h.score, 4),
                "title": h.title,
                "heading": h.heading,
                "file_path": h.file_path,
            }
            for i, h in enumerate(hits[:ALL_K])
        ],
    }
    for k in METRIC_KS:
        result[f"recall_at_{k}"] = 1 if first is not None and first <= k else 0
    return result


def _aggregate(results: list[dict]) -> dict:
    """把题目级结果压成一组均值。"""
    metric_keys = [f"recall_at_{k}" for k in METRIC_KS] + ["recall_all_at_5", "mrr5"]
    agg: dict = {"count": len(results)}
    for key in metric_keys:
        vals = [r[key] for r in results]
        agg[key] = round(statistics.fmean(vals), 4) if vals else None
    latencies = [r["latency_ms"] for r in results]
    agg["latency_ms_mean"] = round(statistics.fmean(latencies), 1) if latencies else None
    agg["latency_ms_p95"] = round(sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))], 1) if latencies else None
    return agg


def _by_category(results: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for category in CATEGORIES:
        sub = [r for r in results if r["category"] == category]
        if sub:
            out[category] = _aggregate(sub)
    return out


def _print_banner(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def _print_summary(label: str, agg: dict) -> None:
    """打印一行汇总表，方便直接抄进报告。"""
    print(
        f"  {label:<18}{agg['count']:>5}"
        f"{agg.get('recall_at_1') or 0:>10.2f}"
        f"{agg.get('recall_at_3') or 0:>10.2f}"
        f"{agg.get('recall_at_5') or 0:>10.2f}"
        f"{agg.get('recall_all_at_5') or 0:>10.2f}"
        f"{agg.get('mrr5') or 0:>8.3f}"
        f"{agg.get('latency_ms_mean') or 0:>12.0f}"
    )


def _print_header_row() -> None:
    print("  模式              题目数  Recall@1  Recall@3  Recall@5  all@5    MRR@5  检索耗时ms")
    print("  " + "-" * 70)


def _print_failures(results: list[dict]) -> None:
    """逐条打印失败题：题目 / 期望文档 / 实际 Top-5。"""
    failed = [r for r in results if r["first_rank"] is None]
    _print_banner(f"失败明细：{len(failed)} / {len(results)} 题的期望文档没进 Top-5")
    if not failed:
        print("  无失败题。")
        return
    for result in failed:
        print()
        print(f"  [{result['id']}] category={result['category']}")
        print(f"      Q: {result['question']}")
        print(f"      期望: {result['expected_documents']}")
        print(f"      关键词命中Top-5正文: {result['keyword_hit']}")
        print("      实际 Top-5:")
        for hit in result["top5"]:
            print(f"        #{hit['rank']} score={hit['score']:.4f}  {hit['title']}  /  {hit['heading']}")
            print(f"             {hit['file_path']}")


def _print_coverage(coverage: list[dict], hits_by_question: dict[str, list[SearchResult]]) -> None:
    """unknown 类题目：看系统在没有答案时会抓什么。"""
    _print_banner(f"覆盖度检查（expected_documents 为空，共 {len(coverage)} 题）")
    for q in coverage:
        hits = hits_by_question[q["question"]]
        top = hits[0] if hits else None
        grabbed = f"{top.file_path} / {top.heading} (score={top.score:.4f})" if top else "(检索为空)"
        pool = "\n".join(h.content for h in hits[:ALL_K])
        leaked = [kw for kw in q.get("expected_keywords", []) if kw in pool]
        print(f"  [{q.get('id')}] {q['question']}")
        print(f"      抓到的 top-1: {grabbed}")
        print(f"      关键词在Top-5正文中出现: {leaked if leaked else '无'}")


def _print_dropped(results: list[dict]) -> None:
    """filter 模式下去掉的条目，供报告"去掉了哪些垃圾结果"。"""
    rows: list[tuple[dict, dict]] = []
    for result in results:
        for dropped in result.get("dropped", []):
            rows.append((result, dropped))
    _print_banner(f"Context filter 过滤掉的条目：{len(rows)} 条")
    seen: set[tuple[str, str]] = set()
    shown = 0
    for _, dropped in rows:
        key = (dropped["file_path"], dropped["heading"])
        if key in seen:
            continue
        seen.add(key)
        shown += 1
        print(
            f"  score={dropped['score']:.4f}  链接占比={dropped['link_ratio']:.2f}  "
            f"链接数={dropped['link_count']}  {dropped['file_path']} / {dropped['heading']}"
        )
    print(f"  (去重后 {shown} 个不同位置)")


def run(
    retriever: Retriever,
    questions: list[dict],
    top_k: int,
    retrieve_top_k: int,
    mode: str,
    reranker: CrossEncoderReranker | None = None,
    candidate_top_k: int = 0,
) -> dict:
    """对一组题跑一个模式，返回聚合结果。"""
    results: list[dict] = []
    for question in questions:
        started = time.perf_counter()
        hits, selection = retrieve(
            retriever, question["question"], top_k, mode, retrieve_top_k, reranker, candidate_top_k
        )
        latency = (time.perf_counter() - started) * 1000
        result = score_question(question, hits, latency)
        if selection is not None:
            result["dropped"] = [
                {
                    "score": round(chunk.hit.score, 4),
                    "title": chunk.hit.title,
                    "file_path": chunk.hit.file_path,
                    "heading": chunk.hit.heading,
                    "link_ratio": round(chunk.judgement.link_ratio, 3),
                    "link_count": chunk.judgement.link_count,
                    "metadata_stripped": chunk.judgement.metadata_stripped,
                }
                for chunk in selection.dropped
            ]
        results.append(result)
    out: dict = {"overall": _aggregate(results), "by_category": _by_category(results), "results": results}
    # 精排模式和粗排模式的分数不在一个尺度上（logit vs 余弦相似度），
    # 只能比排名不能比大小，报告里引用这些数字前要先看这一项。
    # 这里看 mode 而不是 reranker 参数：同一次 A/B 里精排模型只加载一次。
    out["score_scale"] = "cross_encoder_logit" if mode in _MODES_NEEDING_RERANKER else "cosine_similarity"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索质量评估：Recall@k / MRR / 失败明细")
    parser.add_argument("--questions", default="eval/retrieval_questions.json", help="题目 JSON 路径")
    parser.add_argument(
        "--mode", choices=tuple(MODE_GROUPS), default="raw",
        help="raw=直接 Top-K；filter=先取 Top-N 过滤再取 Top-K；rerank=精排后取 Top-K；"
             "filter_rerank=精排后再过滤；both=同进程跑 raw+filter；all4=四组全跑做 A/B",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieve-top-k", type=int, default=20, help="filter 相关模式先多取多少条候选")
    parser.add_argument(
        "--reranker-candidate-top-k", type=int, default=0,
        help="精排前取多少条候选。0 = 用 .env 的 RERANKER_CANDIDATE_TOP_K（默认 50）",
    )
    parser.add_argument(
        "--all", dest="include_unverified", action="store_true",
        help="纳入 verified=false 的候选题（默认只用人工确认过的题）",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题，调试脚本本身用")
    parser.add_argument("--out", default="", help="结果 JSON 输出路径；留空则只打印不写文件")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    questions = load_questions(Path(args.questions))
    scorable, coverage, skipped_unverified, skipped_bad_category = select_questions(
        questions, args.include_unverified
    )
    if args.limit:
        scorable = scorable[: args.limit]
        coverage = coverage[: args.limit]

    _print_banner(f"题目加载：{Path(args.questions)}")
    print(f"  原始题目        : {len(questions)}")
    print(f"  参与计分        : {len(scorable)}  (expected_documents 非空)")
    print(f"  覆盖度检查      : {len(coverage)}  (category=unknown)")
    print(f"  跳过未确认题    : {skipped_unverified}")
    print(f"  跳过非法category: {skipped_bad_category}")
    if not scorable:
        print("  没有可计分的题目。人工确认题目（把 verified 改成 true）或加 --all 后再跑。")
        return 2

    # 不校验 LLM 配置：这个脚本只做检索，不该因为 ANTHROPIC_* 没配就拒绝运行
    settings = Settings()
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    retriever = Retriever(embedding=embedding, store=store, default_top_k=args.top_k)

    print(
        f"  向量库          : {store.count()} 条 chunk  /  "
        f"模型={embedding.model_name}  设备={embedding.device_name}"
    )

    # 只在真的有精排模式时才加载权重：加载一次约 7 秒、占 1.1 GB 显存。
    modes = MODE_GROUPS[args.mode]
    candidate_top_k = args.reranker_candidate_top_k or settings.reranker_candidate_top_k
    reranker = None
    reranker_load_seconds = None
    if any(m in _MODES_NEEDING_RERANKER for m in modes):
        started = time.perf_counter()
        reranker = _build_reranker(settings)
        reranker_load_seconds = round(time.perf_counter() - started, 1)
        print(
            f"  精排模型        : {settings.reranker_model}  加载 {reranker_load_seconds:.1f}s  "
            f"batch={settings.reranker_batch_size}  候选池={candidate_top_k}"
        )

    # 先跑一次不计时的检索。CUDA 首查询会触发 kernel 编译和内存分配，
    # 不预热的话第一题的耗时会把整组均值抬高几十倍（实测首查 220ms / 后续 29ms）。
    # 精排也要单独预热：第一次 predict 会编译 cross-encoder 的 kernel。
    warmup_q = "预热查询，不计入耗时统计"
    warmup_pool = retriever.search(warmup_q, top_k=candidate_top_k or args.top_k)
    retriever.search(warmup_q, top_k=args.top_k)
    if reranker is not None:
        reranker.rerank(warmup_q, warmup_pool)

    runs: dict[str, dict] = {}
    for mode in modes:
        _print_banner(f"模式 {mode}：{_describe_mode(mode, args.top_k, args.retrieve_top_k, candidate_top_k, settings.reranker_model)}")
        # 精排耗时单独记一份：latency_ms 里混着检索和精排，报告里要分开算。
        # 只在精排组做快照——filter 组根本不碰模型，记进去是脏数据。
        before = reranker.stats if mode in _MODES_NEEDING_RERANKER else None
        runs[mode] = run(retriever, scorable, args.top_k, args.retrieve_top_k, mode, reranker, candidate_top_k)
        if before is not None:
            after = reranker.stats
            calls = after.calls - before.calls
            runs[mode]["rerank_calls"] = calls
            runs[mode]["rerank_ms_mean"] = round(
                (after.latency_ms_total - before.latency_ms_total) / calls, 1
            ) if calls else None
            runs[mode]["rerank_fallbacks"] = after.fallback - before.fallback

        _print_header_row()
        _print_summary(f"{mode}", runs[mode]["overall"])
        for category, agg in runs[mode]["by_category"].items():
            _print_summary(f"  - {category}", agg)
        if runs[mode].get("rerank_ms_mean") is not None:
            print(
                f"  精排耗时（不含检索）: 均值 {runs[mode]['rerank_ms_mean']:.0f} ms/次  "
                f"回落 {runs[mode]['rerank_fallbacks']} 次"
            )
        _print_failures(runs[mode]["results"])
        if mode in ("filter", "filter_rerank"):
            _print_dropped(runs[mode]["results"])

    # 覆盖度检查：unknown 题没有 expected_documents，不参与 Recall/MRR，
    # 只回答"知识库没有的时候，检索会把什么抓上来"。这里固定用 raw 模式，
    # 因为要看的是原始检索的能力边界，和 context filter 无关。
    coverage_hits: dict[str, list[SearchResult]] = {}
    if coverage:
        for q in coverage:
            coverage_hits[q["question"]] = retriever.search(q["question"], top_k=args.top_k)
        _print_coverage(coverage, coverage_hits)

    # A/B 对比：只有两组都跑过才有意义
    if len(runs) >= 2:
        _print_banner("A/B 对比")
        _print_header_row()
        for mode in modes:
            _print_summary(mode, runs[mode]["overall"])
        print()
        print(f"  {'模式':<18}{'精排耗时ms':>14}{'检索+精排ms':>16}")
        print("  " + "-" * 50)
        for mode in modes:
            run_ = runs[mode]
            rerank_ms = f"{run_['rerank_ms_mean']:.0f}" if run_.get("rerank_ms_mean") is not None else "  -"
            print(f"  {mode:<18}{rerank_ms:>14}{run_['overall']['latency_ms_mean']:>16.0f}")

    if args.out:
        payload = {
            "meta": {
                "script": "eval_retrieval.py",
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "questions_file": str(Path(args.questions)),
                "modes": list(modes),
                "top_k": args.top_k,
                "retrieve_top_k": args.retrieve_top_k,
                "include_unverified": args.include_unverified,
                "embedding_model": embedding.model_name,
                "chunks_in_index": store.count(),
                "skipped_unverified": skipped_unverified,
                "skipped_bad_category": skipped_bad_category,
                # 精排相关：没跑精排模式时全部记 null，读报告的人一眼知道这组不涉及精排
                "reranker_model": settings.reranker_model if reranker is not None else None,
                "reranker_candidate_top_k": candidate_top_k if reranker is not None else None,
                "reranker_batch_size": settings.reranker_batch_size if reranker is not None else None,
                "reranker_load_seconds": reranker_load_seconds,
            },
            "runs": runs,
            "coverage_check": [
                {
                    "id": q.get("id"),
                    "question": q["question"],
                    "category": q["category"],
                    "expected_keywords": q.get("expected_keywords", []),
                    "top1": (
                        {
                            "file_path": coverage_hits[q["question"]][0].file_path,
                            "heading": coverage_hits[q["question"]][0].heading,
                            "score": round(coverage_hits[q["question"]][0].score, 4),
                        }
                        if coverage_hits[q["question"]]
                        else None
                    ),
                }
                for q in coverage
            ],
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
