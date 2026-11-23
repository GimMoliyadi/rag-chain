"""RAG 回答质量评估。

和 eval_retrieval.py 分工：eval_retrieval 只问"检索到了没有"，本脚本问"回答得对不对"。
检索对了但回答编造引用，等于白搭。

检查项：
    1. answer 必须非空
    2. answer 里出现的 [N] 必须都能对上 sources 里的真实编号
    3. sources 每一条都必须是索引里真实存在的 chunk（防代码或模型拼出假来源）
    4. 检索为空时不得调用 LLM 自由发挥
    5. 知识库确实没有的内容，必须明确说明"信息不足"而不是硬答
    6. answer 的主要事实能否在引用到的 context 里找到字面支撑

第 6 项用 n-gram 覆盖率这种简单规则，不用 LLM judge：
    - 不为了打分再拉一个新的大模型（那是引入第二个不可控的裁判）
    - 规则只能算"字面能对上"，改写了表述的正确回答会被打成 REVIEW
    - 所以本脚本的 REVIEW 不等于"错"，它的意思是"需要你人眼看一眼"

用法：
    python eval_rag.py --all                 # 用真实 LLM 跑全部题
    python eval_rag.py --all --limit 3       # 只跑前 3 题，调试脚本本身
    python eval_rag.py --all --mock          # 用 MockLLM 只验结构，不花钱
    python eval_rag.py --mode raw --all      # 复跑旧版无门控基线
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path

from app.citations import check_citations
from app.config import Settings
from app.database import connect
from app.embedding import get_embedding_client
from app.engine import build_engine
from app.evidence import STATUS_INSUFFICIENT
from app.llm import BaseLLM, MockLLM, get_llm_client
from app.rag import EMPTY_KNOWLEDGE_BASE_ANSWER, RAGAnswer, RAGEngine
from app.retriever import Retriever
from app.vectorstore import ChromaStore

PASS = "PASS"
REVIEW = "REVIEW"
FAIL = "FAIL"
_SEVERITY = {PASS: 0, REVIEW: 1, FAIL: 2}
_VERDICTS = (PASS, REVIEW, FAIL)

# 答案里的引用标记：只认方括号内是纯数字（可含分隔符/连字符）的写法，
# 这样 [文本](url) 和 [[wikilink]] 天然被排除在外。
_BRACKET_RE = re.compile(r"\[([^\]]{0,40})\]")
_CITATION_ONLY_RE = re.compile(r"^[\d\s,、，\-–~]+$")
_NUMBER_RE = re.compile(r"\d+")

# 切句：中文用句末标点，英文兜底句号
_SENTENCE_RE = re.compile(r"[。！？；\n!?;]+")

# 事实支撑检查的参数
CLAIM_MIN_CHARS = 8   # 短于此长度的句子（"是的""不是"）不当事实主张
NGRAM = 8             # 连续字符窗口长度，中文 8 字已经能锚定一个具体说法
SUPPORT_RATIO = 0.30  # 该句 n-gram 命中 context 的比例低于这个值视为未被支撑

# SYSTEM_PROMPT 里要求"知识不足就说不足"，这里认的就是这套措辞
INSUFFICIENT_HINTS = (
    "没有足够的信息",
    "信息不足",
    "找不到",
    "无法回答",
    "知识库中目前",
)

# 纯过渡句，不含事实主张，不进支撑率的分母
_META_RE = re.compile(r"^(根据|依据|如下|以下|综上|总之|总的来说|简单来说|希望|如需|参考[资料文献]|本回答)")

CATEGORIES = ("direct", "paraphrase", "multi_chunk", "hard_negative", "unknown")


class RecordingLLM(BaseLLM):
    """包住真实 LLM，只记录调用次数。

    验证"检索为空就不该调用模型"这条不变式，必须能看到调用有没有发生，
    光看最终文案判断不了。
    """

    def __init__(self, inner: BaseLLM) -> None:
        self._inner = inner
        self.calls = 0

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def chat(self, messages: list[dict]) -> str:
        self.calls += 1
        return self._inner.chat(messages)


class EmptyRetriever:
    """检索永远返回空的替身，用来白盒验证短路分支。

    只实现 RAGEngine 真正用到的 search 方法；类型标注故意不写 Retriever，
    因为它是鸭子类型，强类型反而说不清。
    """

    default_top_k = 5

    def search(
        self, question: str, top_k: int | None = None, domains: list[str] | None = None,
    ) -> list:
        return []


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list):
        raise ValueError(f"题目文件顶层必须是数组：{path}")
    return questions


def load_chunk_index(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """chunk 原文 -> 该 chunk 所属文档的相对路径集合。

    用来验证 sources 不是编出来的：source 必须能在索引里原样找到，
    而且文件路径对得上。14303 条 chunk 约 11 MB，一次性读进来即可。
    """
    rows = conn.execute(
        "SELECT c.content, d.file_path FROM chunks c JOIN documents d ON d.id = c.document_id"
    )
    index: dict[str, set[str]] = {}
    for content, file_path in rows:
        index.setdefault(content, set()).add(file_path)
    return index


def cited_indexes(answer: str) -> list[int]:
    """从答案里提取所有被引用的编号（去重、升序，兼容 [1][2] 和 [1,2] 写法）。"""
    found: set[int] = set()
    for match in _BRACKET_RE.finditer(answer):
        inner = match.group(1).strip()
        if _CITATION_ONLY_RE.match(inner):
            for number in _NUMBER_RE.findall(inner):
                found.add(int(number))
    return sorted(found)


def _extract_claims(answer: str) -> list[str]:
    """把答案切成事实主张句：去掉引用标记，过滤过渡句和过短的句子。"""
    stripped = _BRACKET_RE.sub("", answer)
    claims: list[str] = []
    for piece in _SENTENCE_RE.split(stripped):
        text = piece.strip()
        if len(text) < CLAIM_MIN_CHARS or _META_RE.match(text):
            continue
        claims.append(text)
    return claims


def support_ratio(claims: list[str], context: str) -> tuple[float, list[dict]]:
    """算每句主张在引用 context 里的字面支撑度，返回 (平均支撑率, 存疑句明细)。"""
    if not claims:
        return 1.0, []
    ratios: list[float] = []
    unsupported: list[dict] = []
    for claim in claims:
        grams = {claim} if len(claim) <= NGRAM else {claim[i : i + NGRAM] for i in range(len(claim) - NGRAM + 1)}
        hit = sum(1 for gram in grams if gram in context)
        ratio = hit / len(grams) if grams else 1.0
        ratios.append(ratio)
        if ratio < SUPPORT_RATIO:
            unsupported.append({"claim": claim[:120], "ratio": round(ratio, 3)})
    return sum(ratios) / len(ratios), unsupported


def _verdict(findings: list[tuple[str, str]]) -> str:
    severity = max((_SEVERITY[level] for level, _ in findings), default=0)
    return _VERDICTS[severity]


def _brief(exc: Exception) -> str:
    """异常的一行摘要：类型 + 首行消息。完整 traceback 太长，不适合塞进结果行。"""
    message = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {message[0][:200]}" if message else type(exc).__name__


def _make_row(
    question: dict,
    *,
    verdict: str,
    answer: str,
    findings: list[tuple[str, str]],
    used_chunks: int,
    sources: list,
    refs: list[int],
    llm_calls: int | None,
    latency_ms: float,
    context_filter: bool,
    support: dict | None,
) -> dict:
    """把一题的检查结论组装成结果行。正常路径和异常路径共用，避免两处各写一遍字段。"""
    return {
        "id": question.get("id"),
        "question": question["question"],
        "category": question.get("category"),
        "expected_documents": question.get("expected_documents", []),
        "verdict": verdict,
        "answer": answer,
        "used_chunks": used_chunks,
        "source_count": len(sources),
        "citations": refs,
        "llm_calls": llm_calls,
        "latency_ms": round(latency_ms, 1),
        "context_filter": context_filter,
        "support": support,
        "findings": [{"level": level, "message": message} for level, message in findings],
        "sources": [
            {"index": s.index, "title": s.title, "file_path": s.file_path, "heading": s.heading}
            for s in sources
        ],
    }


def _source_findings(result: RAGAnswer, index: dict[str, set[str]], refs: list[int]) -> list[tuple[str, str]]:
    findings = []
    if not result.answer.strip():
        findings.append((FAIL, "answer 为空"))
    valid = {source.index for source in result.sources}
    dangling = [ref for ref in refs if ref not in valid]
    if dangling:
        findings.append((FAIL, f"引用了不存在的编号 {dangling}"))
    fake = [s.index for s in result.sources if s.file_path not in index.get(s.content, ())]
    if fake:
        findings.append((FAIL, f"sources 第 {fake} 条在索引里找不到对应 chunk"))
    return findings


def check_one(
    question: dict,
    retriever: Retriever,
    llm: BaseLLM,
    index: dict[str, set[str]],
    *,
    top_k: int,
    vault_name: str,
    context_filter: bool,
    filter_retrieve_top_k: int,
    llm_mock: bool,
) -> dict:
    """对一题跑完整 RAG 并给出 PASS / REVIEW / FAIL。

    每题单独 new 一个 RAGEngine，只为把 LLM 换成 RecordingLLM 以便数调用次数。
    engine 构造本身不加载任何模型（retriever 直接传入），所以这没有额外开销。
    """
    recorder = RecordingLLM(llm)
    engine = RAGEngine(
        retriever=retriever,
        llm=recorder,
        top_k=top_k,
        vault_name=vault_name,
        context_filter_enabled=context_filter,
        filter_retrieve_top_k=filter_retrieve_top_k,
    )
    started = time.perf_counter()
    try:
        result = engine.answer(question["question"])
    except Exception as exc:
        # 一题网络抖动不该废掉整轮评估：记为该题 FAIL 继续往下跑。
        # 不处理的话进程直接崩，退出码 1 还会被误读成"有题目没通过检查"。
        print(f"  [{question.get('id')}] FAIL  {_brief(exc)}")
        return _make_row(
            question,
            verdict=FAIL,
            answer="",
            findings=[(FAIL, f"链路抛异常，没有回答：{_brief(exc)}")],
            used_chunks=0,
            sources=[],
            refs=[],
            llm_calls=recorder.calls,
            latency_ms=(time.perf_counter() - started) * 1000,
            context_filter=context_filter,
            support=None,
        )
    latency_ms = (time.perf_counter() - started) * 1000

    answer = result.answer
    sources = result.sources
    refs = cited_indexes(answer)
    findings = _source_findings(result, index, refs)

    # 4. 检索为空不得调用模型；有结果恰好一次
    if result.used_chunks == 0 and recorder.calls != 0:
        findings.append((FAIL, "检索为空却仍然调用了 LLM"))
    elif result.used_chunks > 0 and recorder.calls != 1:
        findings.append((FAIL, f"检索有 {result.used_chunks} 条但 LLM 调用了 {recorder.calls} 次"))

    # 5. 知识不足时必须说不足
    is_unknown = not question.get("expected_documents")
    has_disclaimer = answer == EMPTY_KNOWLEDGE_BASE_ANSWER or any(hint in answer for hint in INSUFFICIENT_HINTS)
    if result.used_chunks == 0:
        if not has_disclaimer:
            findings.append((FAIL, "检索为空且回答没有说明知识不足"))
    elif is_unknown:
        findings.append(
            (REVIEW, f"知识库没有这条内容，但检索仍抓到 {result.used_chunks} 条并作答，需人工确认是否答非所问")
        )
    elif not refs:
        findings.append((REVIEW, "有检索结果但回答里没有任何引用编号"))

    # 6. 事实支撑。MockLLM 是回显，天然全命中，这一项在 mock 下没有意义。
    support: dict | None = None
    if llm_mock:
        findings.append((REVIEW, "mock 模式：LLM 是回显，事实支撑检查跳过"))
    elif result.used_chunks > 0 and not has_disclaimer:
        claimed = _extract_claims(answer)
        cited = {s.index for s in sources if s.index in refs} or {s.index for s in sources}
        context = "\n".join(s.content for s in sources if s.index in cited)
        ratio, unsupported = support_ratio(claimed, context)
        support = {"ratio": round(ratio, 3), "claims": len(claimed), "unsupported": unsupported}
        if ratio < SUPPORT_RATIO:
            findings.append((REVIEW, f"仅 {ratio:.0%} 的主张能在引用内容里找到字面支撑，{len(unsupported)} 句存疑"))

    return _make_row(
        question,
        verdict=_verdict(findings),
        answer=answer,
        findings=findings,
        used_chunks=result.used_chunks,
        sources=sources,
        refs=refs,
        llm_calls=recorder.calls,
        latency_ms=latency_ms,
        context_filter=context_filter,
        support=support,
    )


def _production_outcome(result: RAGAnswer) -> str:
    if result.used_chunks == 0:
        return "no_hits"
    if result.evidence and result.evidence.debug.get("judge_error"):
        return "gate_error"
    if result.evidence and result.evidence.status == STATUS_INSUFFICIENT:
        return "insufficient"
    if result.answer_verification and not result.answer_verification.passed:
        if result.answer_verification.debug.get("judge_error"):
            return "verification_error"
        return "verification_blocked"
    return "answered"


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").casefold()


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _expected_evidence_findings(
    question: dict, result: RAGAnswer, refs: list[int], severity: str,
) -> list[tuple[str, str]]:
    findings = []
    cited = [s for s in result.sources if s.index in refs]
    for expected in question.get("expected_evidence", []):
        expected_path = _normalize_path(expected["file_path"])
        quote = _normalize_text(expected["quote"])
        if not any(
            quote and _normalize_path(source.file_path) == expected_path
            and quote in _normalize_text(source.content)
            for source in cited
        ):
            findings.append((severity, f"未引用预期证据片段：{expected['file_path']}"))
    cited_paths = {_normalize_path(s.file_path) for s in cited}
    missing_documents = [
        path for path in question.get("expected_documents", [])
        if _normalize_path(path) not in cited_paths
    ]
    if missing_documents:
        findings.append((REVIEW, f"未引用 {len(missing_documents)} 篇预期文档"))
    answer = _normalize_text(result.answer).casefold()
    missing_terms = [
        term for term in question.get("expected_answer_terms", [])
        if _normalize_text(term).casefold() not in answer
    ]
    if missing_terms:
        findings.append((severity, f"答案缺少 {len(missing_terms)} 个预期要点"))
    return findings


def _production_findings(
    question: dict, result: RAGAnswer, refs: list[int], outcome: str, llm_mock: bool,
) -> list[tuple[str, str]]:
    if llm_mock:
        return [(REVIEW, "mock 模式只验证装配和响应结构")]
    if outcome in {"gate_error", "verification_error"}:
        return [(FAIL, f"证据评审或核验执行失败（{outcome}）")]
    verified = question.get("verified") is True
    severity = FAIL if verified else REVIEW
    findings = [] if verified else [(REVIEW, "题目标注尚未确认")]
    if question["category"] == "unknown":
        if outcome == "answered":
            level = REVIEW if not refs and any(hint in result.answer for hint in INSUFFICIENT_HINTS) else severity
            findings.append((level, "系统未在检索、证据门控或核验阶段拒答"))
        return findings
    if outcome != "answered":
        return findings + [(severity, f"预期可答但未生成可用答案（{outcome}）")]
    if not refs:
        findings.append((severity, "答案缺少有效引用"))
    if not question.get("expected_evidence") or not question.get("expected_answer_terms"):
        findings.append((REVIEW, "尚未标注证据片段和答案要点，需人工复核"))
    findings += _expected_evidence_findings(question, result, refs, severity)
    return findings


def check_production(
    question: dict, engine: RAGEngine, index: dict[str, set[str]], *,
    context_filter: bool, llm_mock: bool,
) -> dict:
    started = time.perf_counter()
    try:
        result = engine.answer(question["question"])
    except Exception as exc:
        row = _make_row(
            question, verdict=FAIL, answer="",
            findings=[(FAIL, f"链路抛异常，没有回答：{_brief(exc)}")],
            used_chunks=0, sources=[], refs=[], llm_calls=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            context_filter=context_filter, support=None,
        )
        return {**row, "outcome": "error", "route_domains": [], "evidence": None,
                "answer_verification": None, "timings": {}}
    latency_ms = (time.perf_counter() - started) * 1000
    refs = check_citations(result.answer, len(result.sources))["cited_indexes"]
    outcome = _production_outcome(result)
    findings = _source_findings(result, index, refs)
    findings += _production_findings(question, result, refs, outcome, llm_mock)
    row = _make_row(
        question, verdict=_verdict(findings), answer=result.answer, findings=findings,
        used_chunks=result.used_chunks, sources=result.sources, refs=refs, llm_calls=None,
        latency_ms=latency_ms, context_filter=context_filter, support=None,
    )
    return {
        **row, "outcome": outcome,
        "route_domains": result.route.domains if result.route else [],
        "evidence": result.evidence.to_dict() if result.evidence else None,
        "answer_verification": result.answer_verification.to_dict() if result.answer_verification else None,
        "timings": result.timings,
    }


def check_empty_retrieval_guard(llm: BaseLLM) -> dict:
    """白盒验证短路分支：检索恒为空时不得调用 LLM，且必须说明知识不足。

    用假的空检索器而不是凑一个"一定检索不到"的问题——后者依赖语料，不稳定。
    """
    recorder = RecordingLLM(llm)
    engine = RAGEngine(retriever=EmptyRetriever(), llm=recorder, top_k=5)
    try:
        answer = engine.answer("这道题的知识库里完全没有对应内容").answer
    except Exception as exc:
        return {
            "check": "empty_retrieval_guard",
            "verdict": FAIL,
            "llm_calls": recorder.calls,
            "findings": [{"level": FAIL, "message": f"短路分支测试抛异常：{_brief(exc)}"}],
        }

    findings: list[tuple[str, str]] = []
    if recorder.calls != 0:
        findings.append((FAIL, f"检索为空仍然调用了 LLM（{recorder.calls} 次）"))
    if answer != EMPTY_KNOWLEDGE_BASE_ANSWER:
        findings.append((FAIL, "检索为空的回答没有明确说明知识不足"))

    return {
        "check": "empty_retrieval_guard",
        "verdict": _verdict(findings),
        "llm_calls": recorder.calls,
        "findings": [{"level": level, "message": message} for level, message in findings],
    }


def _summary(rows: list[dict]) -> dict:
    counts = Counter(r["verdict"] for r in rows)
    latencies = [r["latency_ms"] for r in rows]
    ordered = sorted(latencies)
    return {
        "count": len(rows),
        "pass": counts[PASS],
        "review": counts[REVIEW],
        "fail": counts[FAIL],
        "latency_ms_mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "latency_ms_p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1) if ordered else None,
    }


def _print_banner(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def _print_table(rows: list[dict]) -> None:
    # 中文表头按"一个汉字占两列"补齐空格，让表头和下面的数据列对得上
    print("  ID   类别          判定     chunk      引用  支撑率   耗时ms  题目")
    print("  " + "-" * 63)
    for row in rows:
        support = f"{row['support']['ratio']:.2f}" if row.get("support") else "  -"
        citations = ",".join(str(n) for n in row["citations"]) or "-"
        print(
            f"  {str(row['id']):>3}  {row['category']:<14}{row['verdict']:<9}"
            f"{row['used_chunks']:>6}{str(citations):>10}{support:>8}{row['latency_ms']:>9}  {row['question'][:24]}"
        )


def _print_detail(rows: list[dict], level: str) -> None:
    shown = [r for r in rows if r["verdict"] == level]
    _print_banner(f"需要看的条目（{level}）：{len(shown)} / {len(rows)} 题")
    for row in shown:
        print()
        print(f"  [{row['id']}] {row['verdict']}  category={row['category']}")
        print(f"      Q: {row['question']}")
        if row["findings"]:
            for finding in row["findings"]:
                print(f"      - [{finding['level']}] {finding['message']}")
        else:
            print("      - （自动检查未发现问题）")
        if row.get("support") and row["support"]["unsupported"]:
            print(f"      支撑率 {row['support']['ratio']:.2f}，未被字面支撑的句子：")
            for item in row["support"]["unsupported"]:
                print(f"        · {item['claim']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 回答质量评估：PASS / REVIEW / FAIL")
    parser.add_argument("--questions", default="eval/retrieval_questions.json", help="题目 JSON 路径，复用检索评估的题集")
    parser.add_argument(
        "--mode", choices=("production", "raw", "filter"), default="production",
        help="production 复用正式问答链路；raw、filter 保留历史基线",
    )
    parser.add_argument("--mock", dest="llm_mock", action="store_true", help="用 MockLLM 只验结构和不变式，不调用真实模型")
    parser.add_argument("--all", dest="include_unverified", action="store_true", help="纳入 verified=false 的候选题")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    parser.add_argument("--out", default="", help="结果 JSON 输出路径")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    questions = [
        q
        for q in load_questions(Path(args.questions))
        if q.get("category") in CATEGORIES and (args.include_unverified or q.get("verified", False))
    ]
    if args.limit:
        questions = questions[: args.limit]

    _print_banner(f"题目加载：{Path(args.questions)}")
    print(f"  参与评估 : {len(questions)}")
    print(f"  模式     : {args.mode}")
    if not questions:
        print("  没有题目可评估。人工确认题目（把 verified 改成 true）或加 --all 后再跑。")
        return 2

    settings = Settings(llm_provider="mock") if args.llm_mock else Settings()
    settings.validate_providers()
    llm = get_llm_client(settings)
    llm_mock = isinstance(llm, MockLLM)
    production = args.mode == "production"
    context_filter = settings.context_filter_enabled if production else args.mode == "filter"
    with closing(connect(settings.db_path_resolved)) as conn:
        chunk_index = load_chunk_index(conn)
    with closing(ChromaStore(settings.chroma_path_resolved)) as store:
        if production:
            engine, _ = build_engine(settings, store=store, llm=llm)
            retriever = engine.retriever
        else:
            retriever = Retriever(
                embedding=get_embedding_client(settings), store=store, default_top_k=settings.top_k,
            )
        chunk_count = store.count()
        print(f"  向量库   : {chunk_count} 条 chunk / 索引内 chunk 原文 {len(chunk_index)} 条")
        print(f"  LLM      : {llm.model_name}   context_filter={context_filter}")
        retriever.search("预热查询，不计入耗时统计", top_k=settings.top_k)

        _print_banner("逐题评估")
        if production:
            rows = [
                check_production(q, engine, chunk_index, context_filter=context_filter, llm_mock=llm_mock)
                for q in questions
            ]
        else:
            rows = [
                check_one(
                    q, retriever, llm, chunk_index, top_k=settings.top_k,
                    vault_name=settings.vault_name, context_filter=context_filter,
                    filter_retrieve_top_k=settings.filter_retrieve_top_k, llm_mock=llm_mock,
                )
                for q in questions
            ]

    _print_banner("汇总表")
    summary = _summary(rows)
    _print_table(rows)
    print()
    print(f"  题目数={summary['count']}  PASS={summary['pass']}  REVIEW={summary['review']}  FAIL={summary['fail']}")
    print(f"  端到端耗时（含 LLM）  均值={summary['latency_ms_mean']}ms  P95={summary['latency_ms_p95']}ms")

    _print_detail(rows, FAIL)
    _print_detail(rows, REVIEW)

    _print_banner("白盒不变式：检索恒为空时的行为")
    guard = check_empty_retrieval_guard(llm)
    print(f"  判定 = {guard['verdict']}   LLM 调用次数 = {guard['llm_calls']}")
    for finding in guard["findings"]:
        print(f"  - [{finding['level']}] {finding['message']}")
    if not guard["findings"]:
        print("  - 检索为空时 LLM 调用 0 次，且回答明确说明了知识不足。")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "script": "eval_rag.py",
                        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "questions_file": str(Path(args.questions)),
                        "mode": args.mode,
                        "llm_mock": llm_mock,
                        "llm_model": llm.model_name,
                        "top_k": settings.top_k,
                        "context_filter": context_filter,
                        "reranker_enabled": production and settings.reranker_enabled,
                        "evidence_gate_enabled": production and settings.evidence_gate_enabled,
                        "strict_evidence_enabled": production and settings.strict_evidence_enabled,
                        "chunks_in_index": chunk_count,
                    },
                    "summary": summary,
                    "empty_retrieval_guard": guard,
                    "results": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n结果已写入 {out_path}")

    return 1 if guard["verdict"] == FAIL or any(r["verdict"] == FAIL for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
