"""Evidence Gate：判断检索结果是否足以支撑回答用户的问题。

为什么需要这一层（实测结论，RETRIEVAL_BASELINE.md）：
有答案查询和无答案查询的 cosine 分数区间高度重叠（无答案的 #31 甚至能拿
0.5204），任何形如 `if score < 0.6: insufficient` 的固定阈值都会大面积
误判——所以这里的判定完全不看分数，只让 LLM 读「问题 + 最终 Top-K chunk」，
回答一个问题：这些资料够不够回答？

职责边界（Judge Prompt 的核心约束）：
    - Judge 不回答用户的问题，只评审证据充分性；
    - Judge 不允许用自己记忆里的知识补充事实——它不知道知识库里有什么，
      用自身知识判断会把「资料没有」错判成「资料够了」；
    - 允许 partial：资料覆盖了问题的部分方面，其余没有依据。

失败策略：Judge 调用失败或输出解析失败时，不阻断问答链路——
默认按 sufficient 处理（回到现有 RAG 行为），并在 debug 里留下
judge_error 标记。宁可多答一次不可靠的答案，不可让整个系统挂掉。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import BaseLLM
from .retriever import SearchResult

# 评审结论的三种取值。sufficient=资料足够正常回答；partial=只够答一部分；
# insufficient=资料只是语义相近，没有实质依据，应当拒答。
STATUS_SUFFICIENT = "sufficient"
STATUS_PARTIAL = "partial"
STATUS_INSUFFICIENT = "insufficient"
VALID_STATUSES = frozenset({STATUS_SUFFICIENT, STATUS_PARTIAL, STATUS_INSUFFICIENT})

# LLM 输出最常见的脏格式是 markdown 代码围栏，先剥掉再解析
JSON_FENCE_PREFIX = "```json"
JSON_FENCE_SUFFIX = "```"


@dataclass
class EvidenceVerdict:
    """一次证据评审的结论。supported/unsupported_aspects 来自 Judge，
    用于 partial 回答时明确区分「能答的」和「缺依据的」部分。

    knowledge_support / personal_context_support 只在 Personal Context 参与评审时
    才有值：Judge 把「知识库资料够不够」和「用户自己填写的当前状态够不够」分开说，
    这样「知识库没有直接答案但个人状态够用」的问题不会被简单判成 insufficient。
    没有个人状态时这两个字段保持 None，而不是 false——None 表示「没参与判断」。
    """

    status: str
    confidence: float = 0.0
    reason: str = ""
    supported_aspects: list[str] = field(default_factory=list)
    unsupported_aspects: list[str] = field(default_factory=list)
    knowledge_support: bool | None = None
    personal_context_support: bool | None = None
    personal_context_available: bool | None = None
    # 过程信息只挂旁边不改契约：judge_error=True 表示这次结论是兜底产生的，
    # 不是 Judge 真实评审出来的，统计指标时应把这类样本剔除。
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转成 API 响应 / eval 落盘用的普通 dict（dataclass.asdict 会带上
        debug 的全部内部字段，这里显式挑选对外字段，避免契约被内部细节牵连）。"""
        return {
            "status": self.status,
            "confidence": self.confidence,
            "reason": self.reason,
            "supported_aspects": self.supported_aspects,
            "unsupported_aspects": self.unsupported_aspects,
            "knowledge_support": self.knowledge_support,
            "personal_context_support": self.personal_context_support,
            "personal_context_available": self.personal_context_available,
            "judge_error": bool(self.debug.get("judge_error")),
            "status_adjusted": bool(self.debug.get("status_adjusted")),
            # 这条结论是不是扩大候选池之后才得到的。统计时要用它把「Judge 本身的能力」
            # 和「候选池太小造成的假不足」分开，否则救援的收益会混进 Judge 的准确率里。
            "rescue_used": bool(self.debug.get("rescue_used")),
        }


# Judge 的 system prompt。每一条约束都对应一种已知的失败方式：
# 「不回答问题本身」防止 Judge 抢答；「不依赖自身记忆」防止它用训练语料
# 里的知识把知识库没有的东西判成 sufficient；「允许 partial」防止它
# 二选一地把半覆盖的资料硬判成 sufficient 或 insufficient。
JUDGE_SYSTEM_PROMPT = """你是知识库问答系统的证据充分性评审员。你的唯一任务：判断给定的参考资料是否足以支撑回答用户的问题。你不回答问题本身。

评审规则：
1. 只依据给出的参考资料判断，不要使用你自己记忆中的任何知识补充事实。
2. sufficient：资料包含回答问题所需的核心事实、方法或结论。
3. partial：资料只覆盖问题的部分方面，其余方面在资料中找不到依据。
4. insufficient：资料与问题只是语义上相近，没有回答问题所需的实质内容。
5. insufficient 时必须在 unsupported_aspects 里明确指出缺少什么信息。
6. 不要因为资料看起来「相关」就判 sufficient——相关不等于足够。

只输出一个 JSON 对象，不要输出任何其它文字。格式：
{"status": "sufficient 或 partial 或 insufficient", "confidence": 0.0到1.0的小数, "reason": "一句话理由", "supported_aspects": ["资料能支撑的方面"], "unsupported_aspects": ["资料缺依据的方面"]}"""

JUDGE_USER_TEMPLATE = """【用户问题】
{question}

{knowledge_block}

请判断这些资料是否足以支撑回答【用户问题】，只输出 JSON。"""

# 有个人状态时的用户消息。顺序刻意是「个人情况 → 问题 → 知识库资料」，
# 和 RAG 生成 Prompt 的三段结构保持一致，Judge 和被评审的生成链路看到的是
# 同一套材料划分，两边对「什么算知识库事实」的理解不会分叉。
JUDGE_USER_TEMPLATE_WITH_PERSONAL_CONTEXT = """【用户个人当前情况】
{personal_context}

【用户问题】
{question}

{knowledge_block}

请判断这些材料是否足以支撑回答【用户问题】，只输出 JSON。"""

# 有 Personal Context 时的附加条款。刻意做成「追加段」而不是改写原 Prompt：
# 没有个人状态的普通问答必须走完全一致的 Judge Prompt，A/B 才能对齐。
# 核心是第 8 条：知识库没有直接答案不等于必须拒答——用户自己填写的当前状态
# （在做什么项目、近期目标）同样可以支撑一次有依据的分析。
PERSONAL_CONTEXT_JUDGE_ADDENDUM = """
7. 除了知识库资料，还给了「用户个人当前情况」——这是用户亲手填写的当前状态，
   是用户亲述的事实，不是知识库内容。两者必须分开判断，不要混为一谈。
8. 知识库资料没有直接答案、但个人当前情况足以支撑分析时，status 判 partial，
   并在 reason 里说明「依据来自用户当前情况」。不要因为知识库缺答案就一律判 insufficient。
9. 个人当前情况里没有提到的信息，视同不知道，不要推断或补全用户的情况。

输出 JSON 需额外包含两个布尔字段（个人当前情况是否存在才判断）：
{"status": "sufficient 或 partial 或 insufficient", "confidence": 0.0到1.0的小数, "reason": "一句话理由", "supported_aspects": ["资料能支撑的方面"], "unsupported_aspects": ["资料缺依据的方面"], "knowledge_support": true或false, "personal_context_support": true或false}"""

# 知识库返回空结果时的占位。Judge 必须看到「检索过但一无所获」这个事实，
# 而不是看着一块空的资料区猜——那样它可能顺着个人情况把 status 判成 sufficient。
EMPTY_KNOWLEDGE_BASE_NOTE = "（本次检索没有返回任何资料）"


class EvidenceGate:
    """用当前真实 LLM 做证据评审。构造时注入 llm，方便测试用 MockLLM。"""

    def __init__(self, llm: BaseLLM, chunk_chars: int = 600):
        if chunk_chars < 1:
            raise ValueError(f"chunk_chars 必须是正整数，当前：{chunk_chars}")
        self._llm = llm
        self._chunk_chars = chunk_chars

    def judge(
        self,
        question: str,
        hits: list[SearchResult],
        personal_context: str | None = None,
    ) -> EvidenceVerdict:
        """评审检索结果。任何失败都降级为「按 sufficient 继续」而不是抛异常，
        这是刻意设计的：门控是为了提升回答质量，不能反过来成为新的单点故障。

        personal_context 是 Personal Context 渲染后的正文（不是结构化对象），
        传 None 表示这次评审没有个人状态，Prompt 与历史版本完全一致。
        """
        if not hits and personal_context is None:
            # 正常流程里 RAG 层在 hits 为空时根本不会走到 Judge（有固定兜底文案）；
            # 这个分支只防调用方直接把空列表塞进来，语义上「没有任何资料」
            # 就是 insufficient，且不需要为此浪费一次 LLM 调用。
            # 有个人状态时不走这个捷径：知识库空 + 个人状态可能正好够用，
            # 必须让 Judge 看到「检索过但一无所获」再判断。
            return EvidenceVerdict(
                status=STATUS_INSUFFICIENT,
                reason="没有检索到任何资料",
                debug={"judge_error": False, "empty_hits": True},
            )

        raw = self._call_judge(question, hits, personal_context)
        if raw is None:
            return self._fallback_verdict("Judge 调用失败")
        verdict = self._parse_verdict(raw, personal_context is not None)
        if verdict is None:
            return self._fallback_verdict("Judge 输出解析失败")
        return verdict

    # ---- 内部实现 ----

    def _call_judge(
        self, question: str, hits: list[SearchResult], personal_context: str | None = None,
    ) -> str | None:
        """调 LLM。网络 / 服务端异常一律吞掉返回 None，由上层统一兜底。"""
        context = _build_judge_context(hits, self._chunk_chars)
        if personal_context is None:
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                    question=question, knowledge_block=_knowledge_block(hits, context),
                )},
            ]
        else:
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT + PERSONAL_CONTEXT_JUDGE_ADDENDUM},
                {"role": "user", "content": JUDGE_USER_TEMPLATE_WITH_PERSONAL_CONTEXT.format(
                    personal_context=personal_context,
                    question=question,
                    knowledge_block=_knowledge_block(hits, context),
                )},
            ]
        try:
            return self._llm.chat(messages)
        except Exception:
            return None

    def _parse_verdict(self, raw: str, personal_context_available: bool) -> EvidenceVerdict | None:
        """从 LLM 输出里抠出 JSON 并校验字段。任何一步不合法都返回 None。"""
        try:
            obj = _extract_json(raw)
        except (ValueError, TypeError):
            return None

        status = str(obj.get("status", "")).strip().lower()
        if status not in VALID_STATUSES:
            return None

        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        # Judge 偶尔会输出 0~100 的分数或负数，统一夹回 0~1，
        # 别让一个越界数字污染下游的统计口径。
        confidence = max(0.0, min(1.0, confidence))

        # 两个 support 字段只在有个人状态时读取；没有个人状态时保持 None，
        # 不读它自己瞎填的值，否则统计里会把「没参与判断」误算成「不支持」。
        knowledge_support = _as_bool(obj.get("knowledge_support")) if personal_context_available else None
        personal_context_support = (
            _as_bool(obj.get("personal_context_support")) if personal_context_available else None
        )

        verdict = EvidenceVerdict(
            status=status,
            confidence=confidence,
            reason=str(obj.get("reason", "")),
            supported_aspects=_as_str_list(obj.get("supported_aspects")),
            unsupported_aspects=_as_str_list(obj.get("unsupported_aspects")),
            knowledge_support=knowledge_support,
            personal_context_support=personal_context_support,
            personal_context_available=personal_context_available,
            debug={"judge_error": False, "raw_chars": len(raw)},
        )
        return _adjust_status_with_personal_context(verdict)

    @staticmethod
    def _fallback_verdict(reason: str) -> EvidenceVerdict:
        """兜底结论：按 sufficient 继续（等于门控这次没拦任何东西），
        现有 RAG 行为原样保留，debug 标记让调用方知道这不是真实评审。"""
        return EvidenceVerdict(
            status=STATUS_SUFFICIENT,
            confidence=0.0,
            reason=reason,
            debug={"judge_error": True, "fallback_reason": reason},
        )


def _knowledge_block(hits: list[SearchResult], context: str) -> str:
    """Judge 消息里的知识库资料段。空结果必须明说「检索过但没有结果」，
    留一片空白的话 Judge 可能顺着其它材料把 status 判成 sufficient。"""
    if not hits:
        return f"【检索到的参考资料】\n{EMPTY_KNOWLEDGE_BASE_NOTE}"
    return f"【检索到的参考资料】\n{context}"


def _adjust_status_with_personal_context(verdict: EvidenceVerdict) -> EvidenceVerdict:
    """一致性修正：知识库没有依据、但个人当前状态足以支撑分析时，insufficient 降为 partial。

    这是确定性规则，不依赖 Prompt 措辞被模型严格遵守——直接拒答会把
    用户亲手填写的当前状态整段扔掉，而那种情况下系统本来是有材料可依的。
    只在这个明确条件成立时改判，并在 debug 里留 status_adjusted 标记，
    统计时能看清有多少次是被规则修正的。
    """
    if (
        verdict.status != STATUS_INSUFFICIENT
        or verdict.knowledge_support is None
        or verdict.personal_context_support is None
    ):
        return verdict
    if verdict.knowledge_support or not verdict.personal_context_support:
        return verdict
    verdict.status = STATUS_PARTIAL
    verdict.debug["status_adjusted"] = True
    return verdict


def _as_bool(value: object) -> bool | None:
    """把 Judge 给的 support 字段规整成布尔。

    LLM 常写成 "true" / "yes" / 1；读不懂时返回 None 而不是猜——
    猜一个布尔值比留空更危险，下游统计会把它当成一个真实判断。
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "支持"}:
        return True
    if text in {"false", "no", "n", "0", "不支持"}:
        return False
    return None


def _build_judge_context(hits: list[SearchResult], chunk_chars: int) -> str:

    """把检索结果拼成 Judge 可读的编号资料块。

    每条 chunk 截到 chunk_chars：Judge 判断的是「有没有回答问题所需的
    实质内容」， chunk 开头的几百字足以说明这一段在讲什么，
    全文塞进去只会让 Judge 请求变慢变贵，判断质量并没有变好。
    """
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        content = hit.content if len(hit.content) <= chunk_chars else hit.content[:chunk_chars] + "…"
        heading = f" / {hit.heading}" if hit.heading else ""
        blocks.append(
            f"[{i}] 文档：{hit.title}{heading}（领域：{hit.domain or 'unknown'}）\n"
            f"内容：{content}"
        )
    return "\n\n".join(blocks)


def _extract_json(raw: str) -> dict:
    """从 LLM 输出里提取 JSON 对象。

    容忍三种已知脏格式：markdown 围栏、前后多余说明文字、
    JSON 后面跟的解释。策略是剥围栏后取首尾大括号之间的部分。
    """
    text = raw.strip()
    lowered = text.lower()
    if lowered.startswith(JSON_FENCE_PREFIX):
        text = text[len(JSON_FENCE_PREFIX):]
    if text.rstrip().lower().endswith(JSON_FENCE_SUFFIX):
        text = text.rstrip()[: -len(JSON_FENCE_SUFFIX)]

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("输出里没有 JSON 对象")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON 顶层不是对象")
    return obj


def _as_str_list(value: object) -> list[str]:
    """把 Judge 给的 aspects 字段规整成字符串列表。

    LLM 有时会把单项写成字符串而不是列表，有时混进非字符串元素——
    这类小毛病不值得让整次评审作废，能修就修。
    """
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
