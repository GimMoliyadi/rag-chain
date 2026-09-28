"""Personal Advisor 检索层评测：路由准确性 / 隔离性 / 证据充分性 / 个人上下文效应。

和 eval_retrieval.py 的分工：那个只算 Recall@K（需要人工 verified 的题），
这个针对 Advisor 层的指标，重点回答四个问题——
    1. Router 把问题送对领域了吗？禁入的领域有没有混进结果？
    2. （开 --gate 时）Evidence Gate 能不能从「总会返回一些相似 chunk」
       升级成「这些 chunk 不足以支持答案」？
    3. （开 --rescue 时）第一次判不足之后，扩大候选池能不能把答案捞回来？
       捞不回来的题（库真没有）会不会被硬判成 sufficient？
    4. （开 --personal-context 时）军师类问题在拿到用户当前情况后，
       评审结论有没有从 insufficient 翻成 partial / sufficient？

题目全部来自 eval/personal_advisor_questions.json，verified:false，
得分只用于内部参考，不作为对外质量承诺。

用法：
    python eval/eval_personal_advisor.py --router-only          # 只测路由层，不调 LLM
    python eval/eval_personal_advisor.py --gate                 # 路由 + 检索 + Evidence Gate
    python eval/eval_personal_advisor.py --gate --router-fallback
    python eval/eval_personal_advisor.py --gate --rescue        # 加一层证据救援
    python eval/eval_personal_advisor.py --gate --personal-context
    # 同一批题跑两遍做 A/B：--personal-context 与 --no-personal-context
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# 本脚本在 eval/ 子目录里被直接运行时，项目根不在 sys.path 上，先补进来
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.embedding import get_embedding_client
from app.evidence import (
    STATUS_INSUFFICIENT,
    STATUS_PARTIAL,
    STATUS_SUFFICIENT,
    EvidenceGate,
)
from app.knowledge_domains import load_router
from app.llm import get_llm_client
from app.personal_context import load_personal_context, render_for_prompt
from app.retriever import Retriever
from app.router import route_with_fallback
from app.vectorstore import ChromaStore


# 评审结论的高低序，只用于 A/B 对比里判断「变好」还是「变差」。
STATUS_RANK = {STATUS_INSUFFICIENT: 0, STATUS_PARTIAL: 1, STATUS_SUFFICIENT: 2}


@dataclass
class EvalOptions:
    """一题的执行配置。全部开关默认跟 .env，命令行 --flag 覆盖。"""

    top_k: int
    personal_context_text: str | None
    rescue_enabled: bool
    rescue_k: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal Advisor 层评测（路由 / 隔离 / 证据充分性）")
    parser.add_argument("--questions", default=str(PROJECT_ROOT / "eval" / "personal_advisor_questions.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gate", dest="gate", action="store_true", default=None,
                        help="启用 Evidence Gate（默认跟 .env 的 EVIDENCE_GATE_ENABLED）")
    parser.add_argument("--no-gate", dest="gate", action="store_false")
    parser.add_argument("--router-fallback", dest="router_fallback", action="store_true", default=None,
                        help="启用 LLM Router fallback（默认跟 .env 的 LLM_ROUTER_FALLBACK_ENABLED）")
    parser.add_argument("--no-router-fallback", dest="router_fallback", action="store_false")
    parser.add_argument("--rescue", dest="rescue", action="store_true", default=None,
                        help="启用 Evidence Rescue（默认跟 .env 的 EVIDENCE_RESCUE_ENABLED）")
    parser.add_argument("--no-rescue", dest="rescue", action="store_false")
    parser.add_argument("--personal-context", dest="personal_context", action="store_true", default=None,
                        help="读取 Personal Context 并参与评审（默认跟 .env 的 PERSONAL_CONTEXT_ENABLED）")
    parser.add_argument("--no-personal-context", dest="personal_context", action="store_false")
    parser.add_argument("--router-only", action="store_true", help="只跑路由层，不检索不评审")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--compare-with", default=None, metavar="REPORT_JSON",
                        help="和另一份报告逐题对比评审结论（典型用法：先跑 --no-personal-context 存一份，"
                             "再用 --personal-context --compare-with 那份做 A/B）")
    parser.add_argument("--out", default=None, help="结果 JSON 输出路径，留空自动命名")
    args = parser.parse_args()

    settings = Settings()
    settings.validate_providers()
    use_gate = settings.evidence_gate_enabled if args.gate is None else args.gate
    use_fallback = (
        settings.llm_router_fallback_enabled if args.router_fallback is None else args.router_fallback
    )
    use_rescue = settings.evidence_rescue_enabled if args.rescue is None else args.rescue
    use_personal_context = (
        settings.personal_context_enabled if args.personal_context is None else args.personal_context
    )
    if args.router_only and (use_gate or use_fallback):
        raise SystemExit("--router-only 不调 LLM，不能和 --gate / --router-fallback 同用")
    if use_rescue and not use_gate:
        raise SystemExit("--rescue 需要 --gate：救援是在第一次评审判定不足之后才跑的流程")
    if args.compare_with and (args.router_only or not use_gate):
        raise SystemExit("--compare-with 需要 --gate：对比的是 Evidence Gate 的评审结论")
    if args.compare_with and (args.router_only or not use_gate):
        raise SystemExit("--compare-with 需要 --gate：对比的是 Evidence Gate 的评审结论")

    questions = _load_questions(Path(args.questions), args.limit)

    router, allowed_domains = load_router(settings.knowledge_domains_path_resolved)
    llm = get_llm_client(settings) if use_fallback else None

    gate = None
    retriever = None
    rescue_retriever = None
    if not args.router_only:
        embedding = get_embedding_client(settings)
        store = ChromaStore(settings.chroma_path_resolved)
        retriever = Retriever(embedding=embedding, store=store, default_top_k=args.top_k)
        if use_gate:
            gate = EvidenceGate(get_llm_client(settings), settings.evidence_judge_chunk_chars)
        # 救援检索器单独一个实例：主检索器可能开着「每文档最多 N chunk」的 diversity 限额，
        # 而救援候选池必须不带这个限额，否则「答案在同文档第 3 个 chunk」这类题捞不回来。
        if use_rescue:
            rescue_retriever = Retriever(
                embedding=embedding, store=store, default_top_k=settings.evidence_rescue_k,
            )

    personal_context_text, personal_context_info = _load_personal_context(
        settings, use_personal_context,
    )
    options = EvalOptions(
        top_k=args.top_k,
        personal_context_text=personal_context_text,
        rescue_enabled=use_rescue and gate is not None,
        rescue_k=settings.evidence_rescue_k,
    )

    print(f"题目        : {len(questions)} 道")
    print(f"Router      : 规则术语表（{len(allowed_domains)} 个可选域）"
          + (" + LLM fallback" if use_fallback else ""))
    print(f"检索        : {'跳过（--router-only）' if args.router_only else f'top_k={args.top_k}'}")
    print(f"Evidence    : {'启用（LLM Judge）' if gate else '关闭'}")
    if rescue_retriever is not None:
        print(f"Rescue      : 启用（候选池 {options.rescue_k}）")
    else:
        print("Rescue      : 关闭")
    if personal_context_text:
        print(f"PersonalCtx : 有内容（{'、'.join(personal_context_info['available_sections'])}）")
    else:
        print("PersonalCtx : 无内容 / 未启用")
    print("=" * 72)

    records = [_run_one(q, router, allowed_domains, llm, use_fallback,
                        retriever, rescue_retriever, gate, options) for q in questions]

    report = {
        "config": {
            "gate": gate is not None, "router_fallback": use_fallback,
            "router_only": args.router_only, "top_k": args.top_k,
            "rescue": rescue_retriever is not None, "rescue_k": settings.evidence_rescue_k,
            "personal_context_enabled": use_personal_context,
            "personal_context": personal_context_info,
            "llm_model": llm.model_name if llm else None,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": _compute_metrics(records),
        "records": records,
    }
    if args.compare_with:
        report["metrics"]["ab_vs_baseline"] = _ab_compare(records, args.compare_with)
    _print_report(report)

    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "eval" / "results" /
        f"personal_advisor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out_path}")


def _load_personal_context(settings: Settings, enabled: bool) -> tuple[str | None, dict]:
    """读取并渲染 Personal Context。返回 (Prompt 正文或 None, 调试信息)。

    正文为 None 有两种情况：开关没开、文件里没有可渲染的内容。
    两种都要落在报告里，否则「军师类题判不足」到底是库的问题还是没填文件，看不出来。
    """
    if not enabled:
        return None, {"enabled": False, "loaded": False, "reason": "Personal Context 未启用"}
    context = load_personal_context(settings.personal_context_path_resolved)
    info = context.to_debug_dict()
    info["enabled"] = True
    rendered = render_for_prompt(context)
    if not rendered:
        info["reason"] = "Personal Context 没有内容"
        return None, info
    return rendered, info


def _run_one(
    q: dict, router, allowed_domains: list[str], llm, use_fallback: bool,
    retriever: Retriever | None, rescue_retriever: Retriever | None,
    gate: EvidenceGate | None, options: EvalOptions,
) -> dict:
    """单题执行：路由 → （检索）→（评审）→（救援），返回逐题记录。任何一步失败都记录而不是中断。"""
    record: dict = {
        "id": q["id"], "category": q["category"], "question": q["question"],
        "expected_domains": q.get("expected_domains", []),
        "forbidden_domains": q.get("forbidden_domains", []),
        "requires_personal_context": bool(q.get("requires_personal_context")),
        "personal_context_available": options.personal_context_text is not None,
    }
    started = time.perf_counter()
    try:
        decision = route_with_fallback(
            router, q["question"], allowed_domains,
            llm=llm, fallback_enabled=use_fallback,
        )
        record["route"] = {"mode": decision.mode, "domains": decision.domains,
                           "origin": decision.origin, "reason": decision.reason}
    except Exception as exc:
        record["route"] = {"mode": "error", "domains": [], "origin": "error", "reason": str(exc)}
        return record
    record["router_ms"] = int((time.perf_counter() - started) * 1000)

    if retriever is None:
        return record

    domains = decision.domains or None
    started = time.perf_counter()
    try:
        hits = retriever.search(q["question"], top_k=options.top_k, domains=domains)
    except Exception as exc:
        record["error"] = f"检索失败: {exc}"
        return record
    record["retrieval_ms"] = int((time.perf_counter() - started) * 1000)
    record["fallback"] = retriever.last_debug.get("domain_empty_fallback", False)
    record["retrieved_domains"] = sorted({h.domain for h in hits if h.domain})
    record["retrieved_files"] = [h.file_path for h in hits[:options.top_k]]

    if gate is None:
        return record

    record.update(_judge_and_rescue(q["question"], domains, hits, gate, rescue_retriever, options))
    return record


def _judge_and_rescue(
    question: str, domains: list[str] | None, hits: list,
    gate: EvidenceGate, rescue_retriever: Retriever | None, options: EvalOptions,
) -> dict:
    """第一次评审 + 至多一次救援，返回逐题字段。

    字段名与 /api/chat 响应的 timings 对齐（first_evidence_status / rescue_triggered /
    rescue_retrieval_ms / rescue_evidence_ms / final_evidence_status），两边结果可对照。
    """
    started = time.perf_counter()
    verdict = gate.judge(question, hits, personal_context=options.personal_context_text)
    fields: dict = {
        "judge_ms": int((time.perf_counter() - started) * 1000),
        "first_evidence_status": verdict.status,
        "rescue_triggered": False,
    }

    if verdict.status == STATUS_INSUFFICIENT and options.rescue_enabled and rescue_retriever is not None:
        started = time.perf_counter()
        rescue_hits = rescue_retriever.search(question, top_k=options.rescue_k, domains=domains)
        fields["rescue_retrieval_ms"] = int((time.perf_counter() - started) * 1000)

        # 救援池不比首轮多出新 chunk 时跳过第二次评审：同一批证据只会得到同一个结论
        if {h.chunk_id for h in rescue_hits} - {h.chunk_id for h in hits}:
            fields["rescue_triggered"] = True
            started = time.perf_counter()
            verdict = gate.judge(question, rescue_hits, personal_context=options.personal_context_text)
            fields["rescue_evidence_ms"] = int((time.perf_counter() - started) * 1000)
            fields["rescue_top_files"] = [h.file_path for h in rescue_hits]

    fields["final_evidence_status"] = verdict.status
    fields["evidence"] = verdict.to_dict()
    fields["knowledge_support"] = verdict.knowledge_support
    fields["personal_context_support"] = verdict.personal_context_support
    return fields


def _ab_compare(records: list[dict], baseline_path: str) -> dict:
    """和另一份报告逐题对比评审结论：同一道题在不同配置下的最终 status 差异。

    baseline 通常是 --no-personal-context 那份，当前这份是 --personal-context。
    「军师题是不是因为个人状态而不再被拒答」靠这张表判定，不靠感觉。
    """
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    baseline_records = baseline.get("records", [])
    baseline_status = {r["id"]: r.get("final_evidence_status") for r in baseline_records}
    # 兜底结论不能进对比：一边是「Judge 解析失败」的 sufficient，另一边是真评审，
    # 这种差不是配置差异造成的，会被误读成「加了个人状态就答上了」。
    baseline_ok = {
        r["id"]: not (r.get("evidence") or {}).get("judge_error") for r in baseline_records
    }

    improved: list[dict] = []
    regressed: list[dict] = []
    unreliable: list[dict] = []
    for record in records:
        before = baseline_status.get(record["id"])
        after = record.get("final_evidence_status")
        if before is None or after is None or before == after:
            continue
        entry = {
            "id": record["id"], "category": record["category"],
            "question": record["question"], "before": before, "after": after,
            "judge_error": bool(record.get("evidence", {}).get("judge_error"))
            or not baseline_ok.get(record["id"], True),
        }
        # 按结论高低排序而不是按 before 是什么来分类：
        # partial → sufficient 同样是变好，只按 before==insufficient 判会把改善误记成退化。
        moved = STATUS_RANK[after] - STATUS_RANK.get(before, STATUS_RANK[after])
        if entry["judge_error"]:
            unreliable.append(entry)
        elif moved > 0:
            improved.append(entry)
        else:
            regressed.append(entry)

    pool = [r for r in records if r.get("requires_personal_context")]
    return {
        "baseline": baseline_path,
        "baseline_generated_at": baseline.get("config", {}).get("generated_at"),
        "baseline_personal_context": baseline.get("config", {}).get("personal_context"),
        "personal_context_pool_n": len(pool),
        "improved_ids": [e["id"] for e in improved],
        "regressed_ids": [e["id"] for e in regressed],
        "unreliable_ids": [e["id"] for e in unreliable],
        "improved": improved,
        "regressed": regressed,
        "unreliable": unreliable,
    }


def _compute_metrics(records: list[dict]) -> dict:
    """七项指标。所有分母都是「该类题数」，0 题时记 null 而不是 0（避免误读）。"""
    router_pool = [r for r in records if r["expected_domains"]]
    strict = sum(
        1 for r in router_pool if set(r["route"]["domains"]) == set(r["expected_domains"])
    )
    loose = sum(
        1 for r in router_pool if set(r["expected_domains"]) <= set(r["route"]["domains"])
    )

    forbidden_pool = [r for r in records if r["forbidden_domains"]]
    intrusions = [
        r for r in forbidden_pool
        if set(r.get("retrieved_domains", [])) & set(r["forbidden_domains"])
    ]

    cross_pool = [r for r in records if r["category"] == "cross_domain"]
    covered = [
        r for r in cross_pool
        if set(r["expected_domains"]) <= set(r.get("retrieved_domains", []))
    ]

    def judge_ok(record: dict) -> bool:
        """这条记录的评审结论是不是 Judge 真评审出来的。

        Judge 调用/解析失败会按 sufficient 兜底（fail-open）：链路不挂是好事，
        但这类结论不是真实判断，进分布里就等于凭空多造「足够」样本，所以一律剔除。
        """
        return not (record.get("evidence") or {}).get("judge_error")

    judge_errors = [r for r in records if not judge_ok(r)]
    judged = [r for r in records if r.get("evidence") and judge_ok(r)]
    status_by_category: dict[str, dict[str, int]] = {}
    for r in judged:
        counts = status_by_category.setdefault(r["category"], {})
        counts[r["evidence"]["status"]] = counts.get(r["evidence"]["status"], 0) + 1

    # 兜底题同样不进分母：否则一次解析失败会被记成「没检测出不足」，凭空算成误报。
    insufficient_pool = [
        r for r in records if r["category"] == "insufficient_evidence" and judge_ok(r)
    ]
    detected = [r for r in insufficient_pool if r.get("evidence", {}).get("status") == "insufficient"]
    first_detected = [
        r for r in insufficient_pool if r.get("first_evidence_status") == STATUS_INSUFFICIENT
    ]

    # Evidence Rescue：分母是「第一次判不足」的题，分子是「扩大候选池后翻回来」的题。
    # 救援池没带出新 chunk 的题不算触发，它们没有多花第二次评审的钱。
    insufficient_first_pool = [
        r for r in records if r.get("first_evidence_status") == STATUS_INSUFFICIENT and judge_ok(r)
    ]
    triggered = [r for r in insufficient_first_pool if r.get("rescue_triggered")]
    recovered = [r for r in triggered if r["final_evidence_status"] != STATUS_INSUFFICIENT]
    rescue_latency = [
        r.get("rescue_retrieval_ms", 0) + r.get("rescue_evidence_ms", 0) for r in triggered
    ]

    # Personal Context 类题在本轮内的状态分布。注意：开了 --personal-context 时
    # 「首轮」就已经带着个人状态了，所以这里只能看出救援的作用；
    # 「没有 PC 时是什么样」必须靠 --compare-with 两份报告对比，见 ab_vs_baseline。
    pc_pool = [r for r in records if r.get("requires_personal_context")]
    pc_judged = [r for r in pc_pool if judge_ok(r)]
    pc_rescued = [
        r for r in pc_judged
        if r.get("first_evidence_status") == STATUS_INSUFFICIENT
        and r.get("final_evidence_status") != STATUS_INSUFFICIENT
    ]
    pc_status_first: dict[str, int] = {}
    pc_status_final: dict[str, int] = {}
    for r in pc_judged:
        for bucket, key in ((pc_status_first, "first_evidence_status"), (pc_status_final, "final_evidence_status")):
            bucket[r[key]] = bucket.get(r[key], 0) + 1

    # 隔离指标换个角度再看一遍：路由本身选错域也是「禁入域混入」的前兆
    forbidden_routed = [
        r for r in forbidden_pool
        if set(r["route"]["domains"]) & set(r["forbidden_domains"])
    ]

    def ratio(hit: int, pool: list) -> float | None:
        return round(hit / len(pool), 4) if pool else None

    return {
        "judge_errors": {
            # 兜底成 sufficient 的题。这些不是 Judge 的判断，读其它指标前先看这里。
            "excluded": len(judge_errors),
            "question_ids": [r["id"] for r in judge_errors],
        },
        "router_domain_accuracy": {
            "strict_match": ratio(strict, router_pool),
            "expected_subset_of_routed": ratio(loose, router_pool),
            "n": len(router_pool),
        },
        "forbidden_domain_intrusion": {
            "intruded_in_retrieval": ratio(len(intrusions), forbidden_pool),
            "routed_into_forbidden": ratio(len(forbidden_routed), forbidden_pool),
            "n": len(forbidden_pool),
            "intruded_question_ids": [r["id"] for r in intrusions],
        },
        "cross_domain_coverage": {
            "both_domains_in_topk": ratio(len(covered), cross_pool),
            "n": len(cross_pool),
            "covered_question_ids": [r["id"] for r in covered],
        },
        "evidence_status_by_category": status_by_category,
        "insufficient_evidence_detection": {
            "detected_insufficient": ratio(len(detected), insufficient_pool),
            "first_round_insufficient": ratio(len(first_detected), insufficient_pool),
            "n": len(insufficient_pool),
            "detected_question_ids": [r["id"] for r in detected],
        },
        "evidence_rescue": {
            "insufficient_first": len(insufficient_first_pool),
            "triggered": len(triggered),
            "recovered": len(recovered),
            "recovery_rate": ratio(len(recovered), triggered),
            "recovered_question_ids": [r["id"] for r in recovered],
            "still_insufficient_ids": [
                r["id"] for r in triggered if r["final_evidence_status"] == STATUS_INSUFFICIENT
            ],
            "extra_judge_calls": len(triggered),
            "rescue_latency_ms_mean": round(sum(rescue_latency) / len(rescue_latency)) if rescue_latency else None,
        },
        "personal_context_effect": {
            "n": len(pc_pool),
            "n_valid": len(pc_judged),
            "available": bool(pc_pool and pc_pool[0].get("personal_context_available")),
            "status_first_round": pc_status_first,
            "status_final": pc_status_final,
            "rescued_from_insufficient": ratio(len(pc_rescued), pc_pool),
            "rescued_question_ids": [r["id"] for r in pc_rescued],
        },
    }


def _print_report(report: dict) -> None:
    metrics = report["metrics"]

    errors = metrics.get("judge_errors")
    if errors and errors["excluded"]:
        print("========== 0. judge_errors（评审失败按 sufficient 兜底，已从下面所有指标剔除） ==========")
        print(f"  剔除题数           : {errors['excluded']}  (题号 {errors['question_ids']})")
        print("  这些题的结论不是 Judge 的判断，读数时先看这一节。\n")

    print("\n========== 1. router_domain_accuracy（路由域与期望域一致） ==========")
    print(f"  严格一致（集合相等）   : {metrics['router_domain_accuracy']['strict_match']}")
    print(f"  宽松（期望 ⊆ 路由结果）: {metrics['router_domain_accuracy']['expected_subset_of_routed']}")

    print("\n========== 2. forbidden_domain_intrusion（禁入域混入，越低越好） ==========")
    print(f"  检索结果混入禁入域   : {metrics['forbidden_domain_intrusion']['intruded_in_retrieval']}"
          f"  (题号 {metrics['forbidden_domain_intrusion']['intruded_question_ids']})")
    print(f"  路由直接选了禁入域   : {metrics['forbidden_domain_intrusion']['routed_into_forbidden']}")

    print("\n========== 3. cross_domain_coverage（跨域两域都进 Top-K） ==========")
    print(f"  覆盖率               : {metrics['cross_domain_coverage']['both_domains_in_topk']}"
          f"  (覆盖题号 {metrics['cross_domain_coverage']['covered_question_ids']})")

    print("\n========== 4. evidence_status_by_category（各题类的评审结论分布） ==========")
    for category, counts in metrics["evidence_status_by_category"].items():
        print(f"  {category:<22} {counts}")

    print("\n========== 5. insufficient_evidence_detection（无答案题被拒答的比例） ==========")
    m5 = metrics["insufficient_evidence_detection"]
    print(f"  最终检出率           : {m5['detected_insufficient']}  (检出题号 {m5['detected_question_ids']})")
    print(f"  首轮就检出           : {m5['first_round_insufficient']}")

    print("\n========== 6. evidence_rescue（判不足后扩大候选池的救援效果） ==========")
    m6 = metrics["evidence_rescue"]
    if m6["insufficient_first"] == 0:
        print("  没有第一次判 insufficient 的题，救援未参与（未开启 --rescue，或 Gate 没开）。")
    else:
        print(f"  第一次判不足         : {m6['insufficient_first']} 题")
        print(f"  救援触发（池子带出新资料）: {m6['triggered']} 题，多花 {m6['extra_judge_calls']} 次 LLM 调用")
        print(f"  救援后翻回 partial/sufficient: {m6['recovered']} 题，"
              f"救援成功率 {m6['recovery_rate']}  (题号 {m6['recovered_question_ids']})")
        print(f"  救援后仍不足         : {m6['still_insufficient_ids']}  (库真没有，不能被硬判 sufficient)")
        print(f"  救援额外耗时均值     : {m6['rescue_latency_ms_mean']} ms")

    print("\n========== 7. personal_context_effect（军师题在本轮内的结论分布） ==========")
    m7 = metrics["personal_context_effect"]
    if m7["n"] == 0:
        print("  题目里没有标记 requires_personal_context 的题。")
    else:
        valid_note = "" if m7["n_valid"] == m7["n"] else (
            f"（其中 {m7['n'] - m7['n_valid']} 道评审失败已剔除）"
        )
        if not m7["available"]:
            print(f"  {m7['n']} 道军师题，但 Personal Context 无内容 / 未启用"
                  f"（首轮结论分布 {m7['status_first_round']}）{valid_note}")
        else:
            print(f"  军师题              : {m7['n']} 道{valid_note}")
        print(f"  首轮结论分布        : {m7['status_first_round']}")
        print(f"  最终结论分布        : {m7['status_final']}")
        print(f"  救援后从不足翻回    : {m7['rescued_from_insufficient']}"
              f"  (题号 {m7['rescued_question_ids']})")

    ab = metrics.get("ab_vs_baseline")
    if ab is None:
        return
    print("\n========== 8. ab_vs_baseline（与另一份配置的逐题结论对比） ==========")
    print(f"  对照报告            : {ab['baseline']}  ({ab['baseline_generated_at']})")
    if ab["improved_ids"] or ab["regressed_ids"]:
        print(f"  改善（结论升高）: 题号 {ab['improved_ids']}")
        for entry in ab["improved"]:
            print(f"      [{entry['id']}] {entry['category']}: {entry['question']}  "
                  f"{entry['before']} → {entry['after']}")
        print(f"  退化（原本能给结论现在被拒答）: 题号 {ab['regressed_ids']}")
        for entry in ab["regressed"]:
            print(f"      [{entry['id']}] {entry['category']}: {entry['question']}  "
                  f"{entry['before']} → {entry['after']}")
    else:
        print("  没有题的结论发生变化。")
    if ab.get("unreliable_ids"):
        print(f"  不可比（任一侧是评审失败兜底，不算真变化）: 题号 {ab['unreliable_ids']}")
        for entry in ab["unreliable"]:
            print(f"      [{entry['id']}] {entry['category']}: {entry['question']}  "
                  f"{entry['before']} → {entry['after']}")


def _load_questions(path: Path, limit: int | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"]
    return questions if limit is None else questions[:limit]


if __name__ == "__main__":
    main()
