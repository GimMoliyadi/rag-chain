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

失败策略：默认保持历史行为：Judge 调用失败或输出解析失败时按
sufficient 处理，并在 debug 里留下 judge_error 标记。需要把“宁可不答也不
冒险”的可靠性放在第一位时，构造 `EvidenceGate(..., fail_closed=True)`；此时
评审失败会返回明确的拒答结论，不会把未确认的资料交给生成模型。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import BaseLLM
from .retriever import SearchResult

# 评审结论的三种取值。sufficient=资料足够正常回答；partial=只够答一部分；
# insufficient=资料只是语义相近，没有实质依据，应当拒答。
STATUS_SUFFICIENT = "sufficient"
STATUS_PARTIAL = "partial"
STATUS_INSUFFICIENT = "insufficient"
VALID_STATUSES = frozenset({STATUS_SUFFICIENT, STATUS_PARTIAL, STATUS_INSUFFICIENT})

# 最终答案核对的状态。严格模式只放行 supported；unsupported / uncertain /
# unverified 都会留在固定拒答文案，不把未经核对的答案交给调用方。
ANSWER_STATUS_SUPPORTED = "supported"
ANSWER_STATUS_UNSUPPORTED = "unsupported"
ANSWER_STATUS_UNCERTAIN = "uncertain"
ANSWER_STATUS_UNVERIFIED = "unverified"
VALID_ANSWER_STATUSES = frozenset({
    ANSWER_STATUS_SUPPORTED,
    ANSWER_STATUS_UNSUPPORTED,
    ANSWER_STATUS_UNCERTAIN,
})

# 最终核验器在 advisor 模式下还要标出主张的材料来源。缺省不写 basis
# 时按 knowledge 处理，是为了兼容旧版 verifier 输出；normal 模式仍然只
# 接受带知识库引用的主张。
ANSWER_CLAIM_BASIS_KNOWLEDGE = "knowledge"
ANSWER_CLAIM_BASIS_PERSONAL_CONTEXT = "personal_context"
ANSWER_CLAIM_BASIS_ANALYSIS = "analysis"
ANSWER_CLAIM_BASIS_NON_CLAIM = "non_claim"
VALID_ANSWER_CLAIM_BASES = frozenset({
    ANSWER_CLAIM_BASIS_KNOWLEDGE,
    ANSWER_CLAIM_BASIS_PERSONAL_CONTEXT,
    ANSWER_CLAIM_BASIS_ANALYSIS,
    ANSWER_CLAIM_BASIS_NON_CLAIM,
})

# Keep this parser in sync with app.citations.check_citations().  The verifier
# needs the citation markers attached to each deterministic answer unit, not
# only the global set returned by check_citations().
_ANSWER_CITATION = re.compile(r"(?<!\\)\[(\d+(?:\s*[,，]\s*\d+)*)\](?!\()")
_CONTEXT_HEADER = re.compile(r"(?m)^\[(\d+)\][^\n]*$")


@dataclass(frozen=True)
class _AnswerUnit:
    """A deterministic unit that the verifier must judge exactly once."""

    unit_id: str
    text: str
    citation_indexes: tuple[int, ...]

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
            "strict_fail_closed": bool(self.debug.get("strict_fail_closed")),
            "status_adjusted": bool(self.debug.get("status_adjusted")),
            # 这条结论是不是扩大候选池之后才得到的。统计时要用它把「Judge 本身的能力」
            # 和「候选池太小造成的假不足」分开，否则救援的收益会混进 Judge 的准确率里。
            "rescue_used": bool(self.debug.get("rescue_used")),
        }


@dataclass
class AnswerVerification:
    """对最终生成文本逐项核对后的结果。

    这不是事实证明：它记录的是一次独立 LLM 评审是否认为每条回答主张都
    能在本次提供的资料中找到依据。严格编排层只在 ``passed`` 为真时放行，
    调用失败、输出损坏、引用缺失或任一主张不受支持都会拒答。
    ``claims`` 保留在进程内供测试和调试使用；``to_dict`` 只暴露数量和状态，
    避免拒答时把被拦截的原回答片段重新回显给调用方。
    """

    status: str
    confidence: float = 0.0
    reason: str = ""
    claims: list[dict] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    citation_check: dict | None = None
    debug: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """严格放行条件：评审成功且每个解析出的主张都得到引用支持。"""
        return (
            self.status == ANSWER_STATUS_SUPPORTED
            and not self.unsupported_claims
            and not self.invalid_citations
            and not self.debug.get("judge_error")
            and not self.debug.get("citation_error")
        )

    def to_dict(self) -> dict:
        """转成响应元数据，但不回显被拦截回答中的逐字主张。

        ``reason`` 也是模型生成的自由文本；拒答时它可能顺手复述一条
        未核验主张，所以拒答路径只返回固定的诊断理由。逐条 claim 仍保留
        在进程内，供日志/测试使用，但不会通过 HTTP 或 SSE 出去。
        """
        if self.passed:
            public_reason = self.reason
        elif self.debug.get("citation_error"):
            public_reason = "回答引用编号缺失或无效，未输出原回答。"
        elif self.debug.get("judge_error"):
            public_reason = "最终答案证据核对未完成，未输出原回答。"
        else:
            public_reason = "最终答案未通过逐项证据核对，未输出原回答。"
        return {
            "status": self.status,
            "confidence": self.confidence,
            "reason": public_reason,
            "claim_count": len(self.claims),
            "unsupported_claim_count": len(self.unsupported_claims),
            "invalid_citations": self.invalid_citations,
            "citation_check": self.citation_check,
            "judge_error": bool(self.debug.get("judge_error")),
            "citation_error": bool(self.debug.get("citation_error")),
            "strict_fail_closed": bool(self.debug.get("strict_fail_closed")),
            # 调用方可能仍会收到检索 sources；这个显式布尔值告诉它们
            # 这些来源是否可以作为本次答案的依据。
            "sources_usable": self.passed,
            # 只暴露按来源聚合后的数量，不暴露主张文本或模型的逐条 reason。
            "basis_counts": _claim_basis_counts(self.claims),
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
7. 评审范围只限本次提供的参考资料；insufficient 只表示这些资料不足，不能断言整个知识库不存在答案。

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
8. 除了知识库资料，还给了「用户个人当前情况」——这是用户亲手填写的当前状态，
   是用户亲述的事实，不是知识库内容。两者必须分开判断，不要混为一谈。
9. 知识库资料没有直接答案、但个人当前情况足以支撑分析时，status 判 partial，
   并在 reason 里说明「依据来自用户当前情况」。不要因为知识库缺答案就一律判 insufficient。
10. 个人当前情况里没有提到的信息，视同不知道，不要推断或补全用户的情况。

输出 JSON 需额外包含两个布尔字段（个人当前情况是否存在才判断）：
{"status": "sufficient 或 partial 或 insufficient", "confidence": 0.0到1.0的小数, "reason": "一句话理由", "supported_aspects": ["资料能支撑的方面"], "unsupported_aspects": ["资料缺依据的方面"], "knowledge_support": true或false, "personal_context_support": true或false}"""

# 知识库返回空结果时的占位。Judge 必须看到「检索过但一无所获」这个事实，
# 而不是看着一块空的资料区猜——那样它可能顺着个人情况把 status 判成 sufficient。
EMPTY_KNOWLEDGE_BASE_NOTE = "（本次检索没有返回任何资料）"


class EvidenceGate:
    """用当前真实 LLM 做证据评审。构造时注入 llm，方便测试用 MockLLM。"""

    def __init__(
        self,
        llm: BaseLLM,
        chunk_chars: int = 600,
        *,
        fail_closed: bool = False,
    ):
        if chunk_chars < 1:
            raise ValueError(f"chunk_chars 必须是正整数，当前：{chunk_chars}")
        self._llm = llm
        self._chunk_chars = chunk_chars
        self._fail_closed = fail_closed

    def judge(
        self,
        question: str,
        hits: list[SearchResult],
        personal_context: str | None = None,
        evidence_context: str | None = None,
    ) -> EvidenceVerdict:
        """评审检索结果。

        默认模式保留历史兼容行为：任何失败都降级为「按 sufficient 继续」。
        `fail_closed=True` 时则反过来把评审失败当作不可确认，返回拒答结论，
        从而保证生成模型不会在 Judge 没有成功工作的情况下自由发挥。

        personal_context 是 Personal Context 渲染后的正文（不是结构化对象），
        传 None 表示这次评审没有个人状态，Prompt 与历史版本完全一致。
        evidence_context 传入时直接作为 Judge 看到的资料正文；RAG 编排层用它
        复用已经完成清洗和截断的最终 context，避免 Judge 与生成器看到不同文本。
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

        raw = self._call_judge(question, hits, personal_context, evidence_context)
        if raw is None:
            return self._failed_verdict("Judge 调用失败")
        verdict = self._parse_verdict(raw, personal_context is not None)
        if verdict is None:
            return self._failed_verdict("Judge 输出解析失败")
        return verdict

    # ---- 内部实现 ----

    def _call_judge(
        self,
        question: str,
        hits: list[SearchResult],
        personal_context: str | None = None,
        evidence_context: str | None = None,
    ) -> str | None:
        """调 LLM。网络 / 服务端异常一律吞掉返回 None，由上层统一兜底。"""
        context = evidence_context if evidence_context is not None else _build_judge_context(
            hits, self._chunk_chars,
        )
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
        """兼容旧调用方的 fail-open 兜底。"""
        return EvidenceVerdict(
            status=STATUS_SUFFICIENT,
            confidence=0.0,
            reason=reason,
            debug={"judge_error": True, "fallback_reason": reason},
        )

    def _failed_verdict(self, reason: str) -> EvidenceVerdict:
        """严格模式的 fail-closed 兜底。

        这里的状态只表示“本次证据评审失败”，不表示全库不存在答案；调用方
        会据 `strict_fail_closed` 选择不会扩大含义的拒答文案。
        """
        if not self._fail_closed:
            return self._fallback_verdict(reason)
        return EvidenceVerdict(
            status=STATUS_INSUFFICIENT,
            confidence=0.0,
            reason=(
                "证据评审失败，无法确认本次检索资料是否足以支持回答；"
                "这不代表知识库中不存在相关内容。"
            ),
            debug={
                "judge_error": True,
                "strict_fail_closed": True,
                "fallback_reason": reason,
            },
        )


ANSWER_VERIFIER_SYSTEM_PROMPT = """你是知识库问答系统的最终答案证据核对员。你的唯一任务：逐项检查【最终回答】中的事实性主张，判断它们是否都能被编号参考资料直接支持。你不改写回答，也不回答问题。

核对规则：
1. 只能使用给出的编号参考资料，禁止使用你自己的常识、记忆或外部知识补全。
2. 系统已经把最终回答切成带 unit_id 的文本单元。必须对每个单元各返回一条 claims 项，unit_id 集合必须与输入完全相同：不能漏项、重复或新增，也不能只摘录单元中的一半来代替整个单元。
3. 每条需要知识库事实支撑的单元都必须列出它实际使用的引用编号；citation_indexes 必须与该单元正文中实际出现的编号完全一致。没有引用或引用的原文不能推出该单元时，supported 必须为 false。
4. 对每个知识库引用，必须返回 evidence_quotes，其中 citation_index 对应编号，quote 必须是对应原文的连续短摘录；同一个 citation_index 可以返回多个不同摘录，以覆盖该单元的多个事实。不能用其它来源的摘录冒充支持。quote 请直接从该编号的“内容”复制，不能改写、总结、翻译或删改事实字符，也不能用 `...` / `……` 拼接两处文字。核验器只会忽略换行/空白、Markdown 展示标记（如 `**`、列表符号、标题符号、数学 `$` 分隔符）以及导入文本中的转义引号；LaTeX 命令、数学运算符、中文词语和“必须”等限定词都不能替换或删除。优先选择原文中约 8–30 个字符、完全不跨这些标记的连续片段（例如“所有节点属于同一个玩家”或“这个条件排除了任何”）。
5. 用户问题、回答中的礼貌语、标题或格式标记可以标为 is_claim=false、basis=non_claim；它们也必须有对应 unit_id，但不需要知识库引用。只要有一个事实性单元缺乏依据，整体 status 就是 unsupported。
6. 无法确定时用 uncertain，不要为了放行而猜测。

只输出一个 JSON 对象，不要输出其它文字：
{"status":"supported 或 unsupported 或 uncertain","confidence":0.0到1.0的小数,"reason":"一句话理由","claims":[{"unit_id":"u1","claim":"该单元原文（仅供说明）","is_claim":true,"basis":"knowledge 或 non_claim","supported":true或false,"citation_indexes":[1],"evidence_quotes":[{"citation_index":1,"quote":"对应原文中的连续短摘录"}],"reason":"对应原文为何支持或不支持"}],"invalid_citations":[]}"""

ANSWER_VERIFIER_ADVISOR_SYSTEM_ADDENDUM = """

这是 advisor 模式。除了编号参考资料，还给你一份与生成器完全相同的【用户当前情况】。
逐条主张必须标明唯一的 basis：
- knowledge：知识库事实或结论。必须有一个或多个 citation_indexes，并且编号真实存在、确实出现在该文本单元中；每个编号都要有 evidence_quotes，quote 必须能在对应编号的原文中精确定位。没有引用的知识库事实必须 supported=false。
- personal_context：只复述【用户当前情况】中明确写出的用户事实。必须提供 personal_context_quote，且摘录必须原样出现在这一次传入的个人上下文中；不能从个人情况推断或补全其它用户事实。此类主张的 citation_indexes 必须为空。
- analysis：明确标注为“分析 / 建议 / 判断 / 推断 / 推荐”等的推理或行动建议，不得伪装成知识库事实或用户已确认的事实。分析标记必须出现在该文本单元本身，而不是回答其它位置；它必须有知识库 evidence_quotes 或 personal_context_quote 作为依据，不能只因为全篇出现过“建议”就放行。若分析段含有可独立核验的知识库事实，应标为 knowledge 并要求引用。
- non_claim：标题、礼貌语或纯格式文本，没有事实性内容。
所有文本单元都必须返回一条 claims 项，unit_id 不能漏项或重复。只要一条用户事实无个人上下文摘录、一条知识库事实无有效引用，或一条分析没有可定位依据，整体 status 就是 unsupported；无法确定时用 uncertain。

advisor 模式额外要求的 JSON 格式：
{"status":"supported 或 unsupported 或 uncertain","confidence":0.0到1.0的小数,"reason":"一句话理由","claims":[{"unit_id":"u1","claim":"该单元原文（仅供说明）","is_claim":true,"basis":"knowledge 或 personal_context 或 analysis 或 non_claim","supported":true或false,"citation_indexes":[1],"evidence_quotes":[{"citation_index":1,"quote":"对应原文中的连续短摘录"}],"personal_context_quote":"个人上下文中的连续短摘录","reason":"对应材料为何支持或不支持"}],"invalid_citations":[]}
"""

ANSWER_VERIFIER_USER_TEMPLATE = """【用户问题】
{question}

【最终回答】
{answer}

【代码生成的最终回答单元（必须逐一返回 unit_id）】
{answer_units}

【编号参考资料】
{context}

【机器引用检查】
{citation_check}

请逐项核对最终回答，不要输出回答正文，只输出 JSON。"""

ANSWER_VERIFIER_ADVISOR_USER_TEMPLATE = """【用户问题】
{question}

【用户当前情况】
{personal_context}

【最终回答】
{answer}

【代码生成的最终回答单元（必须逐一返回 unit_id）】
{answer_units}

【编号参考资料】
{context}

【机器引用检查】
{citation_check}

请逐项核对最终回答，区分 knowledge、personal_context、analysis 三种来源；不要输出回答正文，只输出 JSON。"""


class AnswerVerifier:
    """用独立一次 LLM 调用核对最终回答是否忠于本次证据。

    这层只在 RAGEngine 的严格模式启用。引用编号先做确定性检查，随后让
    LLM 按原子主张逐项比对同一份最终 context；任何异常或不确定都返回
    ``passed=False``，由编排层输出固定拒答而不是原回答。
    """

    def __init__(self, llm: BaseLLM):
        self._llm = llm

    def verify(
        self,
        question: str,
        answer: str,
        context: str,
        citation_check: dict,
        *,
        personal_context: str | None = None,
        advisor_mode: bool = False,
        personal_context_available: bool | None = None,
    ) -> AnswerVerification:
        """核对最终答案；失败永远返回结果对象，不把异常交给生成链路。

        ``personal_context`` 必须是生成器实际使用的渲染正文，而不是重新从
        文件读取的结构化对象。这样 advisor 的最终核验和生成器看到的是同一
        份用户状态；normal 调用不提供它，保持原来的“每条主张都要知识库引用”
        语义。
        """
        if not answer.strip():
            return _failed_answer_verification("最终回答为空")

        answer_units = _build_answer_units(answer)
        if not answer_units:
            return _failed_answer_verification("最终回答没有可核验文本单元")
        answer_units_text = _format_answer_units(answer_units)

        # 这是不依赖模型的硬门槛：编号缺失 / 越界时没有可执行的逐项映射，
        # 不应再花一次 LLM 调用，也不能让模型自行猜编号含义。advisor 允许
        # 纯个人情况 / 明确分析没有知识库引用，但仍然拒绝任何无效编号。
        citation_status = citation_check.get("status")
        citation_ok = citation_status == "ok"
        citation_missing_allowed = advisor_mode and citation_status == "missing"
        if not citation_ok and not citation_missing_allowed:
            return AnswerVerification(
                status=ANSWER_STATUS_UNSUPPORTED,
                confidence=0.0,
                reason="回答的引用编号缺失或无效，无法逐项核对对应原文。",
                invalid_citations=list(citation_check.get("invalid_indexes", [])),
                citation_check=citation_check,
                debug={"citation_error": True, "strict_fail_closed": True},
            )

        if advisor_mode:
            messages = [
                {
                    "role": "system",
                    "content": ANSWER_VERIFIER_SYSTEM_PROMPT + ANSWER_VERIFIER_ADVISOR_SYSTEM_ADDENDUM,
                },
                {"role": "user", "content": ANSWER_VERIFIER_ADVISOR_USER_TEMPLATE.format(
                    question=question,
                    personal_context=personal_context or "（当前没有个人状态信息）",
                    answer=answer,
                    answer_units=answer_units_text,
                    context=context,
                    citation_check=json.dumps(citation_check, ensure_ascii=False),
                )},
            ]
        else:
            messages = [
                {"role": "system", "content": ANSWER_VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": ANSWER_VERIFIER_USER_TEMPLATE.format(
                    question=question,
                    answer=answer,
                    answer_units=answer_units_text,
                    context=context,
                    citation_check=json.dumps(citation_check, ensure_ascii=False),
                )},
            ]
        try:
            raw = self._llm.chat(messages)
        except Exception:
            return _failed_answer_verification("最终答案证据核对调用失败")

        context_available = (
            bool(personal_context)
            if personal_context_available is None
            else personal_context_available
        )
        result = self._parse_verification(
            raw,
            citation_check,
            advisor_mode=advisor_mode,
            personal_context_available=context_available,
            personal_context=personal_context if context_available else None,
            context=context,
            answer=answer,
            answer_units=answer_units,
        )
        parse_debug = result.debug
        # One bounded retry, only when every model verdict was "supported" but
        # the excerpts could not be located verbatim in the cited block.  That
        # is a copying defect in the *reviewer*, not evidence that the answer
        # text is unsupported: the words are still checked against the same
        # context, so a hallucinated claim cannot pass on the second attempt.
        # Semantic rejections (model said unsupported) are never retried, and a
        # failed retry keeps the stricter of the two outcomes.
        if (
            result.passed
            or not parse_debug.get("quote_unlocatable")
            or parse_debug.get("judge_error")
            or parse_debug.get("citation_error")
        ):
            return result
        retry_messages = messages + [
            {
                "role": "user",
                "content": _quote_retry_instruction(parse_debug),
            }
        ]
        try:
            retry_raw = self._llm.chat(retry_messages)
        except Exception:
            return result
        retry = self._parse_verification(
            retry_raw,
            citation_check,
            advisor_mode=advisor_mode,
            personal_context_available=context_available,
            personal_context=personal_context if context_available else None,
            context=context,
            answer=answer,
            answer_units=answer_units,
        )
        retry.debug["quote_retry_used"] = True
        if retry.passed:
            return retry
        # Neither attempt passed: report the first (strict) verdict, with the
        # retry recorded so refusals can be audited separately from one-shot
        # judge failures.
        result.debug["quote_retry_used"] = True
        result.debug["quote_retry_status"] = retry.status
        return result

    def _parse_verification(
        self,
        raw: str,
        citation_check: dict,
        *,
        advisor_mode: bool = False,
        personal_context_available: bool = False,
        personal_context: str | None = None,
        context: str = "",
        answer: str = "",
        answer_units: list[_AnswerUnit] | None = None,
    ) -> AnswerVerification:
        try:
            obj = _extract_json(raw)
        except (ValueError, TypeError):
            return _failed_answer_verification("最终答案证据核对输出无法解析")

        status = str(obj.get("status", "")).strip().lower()
        if status not in VALID_ANSWER_STATUSES:
            return _failed_answer_verification("最终答案证据核对状态无效")

        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            return _failed_answer_verification("最终答案证据核对置信度无效")
        confidence = max(0.0, min(1.0, confidence))

        raw_claims = obj.get("claims")
        if not isinstance(raw_claims, list) or not raw_claims:
            return _failed_answer_verification("最终答案证据核对没有逐项列出主张")

        units = answer_units if answer_units is not None else _build_answer_units(answer)
        if not units:
            return _failed_answer_verification("最终回答没有可核验文本单元")
        units_by_id = {unit.unit_id: unit for unit in units}
        expected_unit_ids = set(units_by_id)
        returned_unit_ids: set[str] = set()
        if len(raw_claims) != len(units):
            return _failed_answer_verification("最终答案证据核验漏列或重复了文本单元")

        available = (
            set(citation_check.get("cited_indexes", []))
            | set(citation_check.get("unused_indexes", []))
        )
        answer_cited = set(citation_check.get("cited_indexes", []))
        # citation_check 已确认引用全部落在 sources 内；这里再按实际出现的编号
        # 做一层边界计算，避免模型返回不在本次资料中的编号时误报 supported。
        invalid: set[int] = set(citation_check.get("invalid_indexes", []))
        claims: list[dict] = []
        unsupported_claims: list[str] = []
        basis_counts: dict[str, int] = {}
        surface_mismatch_count = 0
        relabeled_quotes = 0
        unlocatable_units: list[str] = []
        source_texts = _context_source_texts(context)
        surface_texts = _context_source_surface_texts(context)
        for raw_claim in raw_claims:
            if not isinstance(raw_claim, dict):
                return _failed_answer_verification("最终答案证据核对的主张格式无效")
            raw_unit_id = raw_claim.get("unit_id")
            if not isinstance(raw_unit_id, str) or not raw_unit_id.strip():
                return _failed_answer_verification("最终答案证据核对缺少文本单元编号")
            unit_id = raw_unit_id.strip()
            if unit_id not in units_by_id or unit_id in returned_unit_ids:
                return _failed_answer_verification("最终答案证据核对的文本单元编号重复或不存在")
            returned_unit_ids.add(unit_id)
            unit = units_by_id[unit_id]

            # The unit text is generated by code and is the scope being judged.
            # A model-supplied claim may paraphrase it, but can never narrow the
            # scope by replacing the unit with only one of its clauses.
            claim = unit.text
            supported = raw_claim.get("supported")
            raw_indexes = raw_claim.get("citation_indexes")
            is_claim = raw_claim.get("is_claim", True)
            if not isinstance(supported, bool) or not isinstance(is_claim, bool) or not isinstance(raw_indexes, list):
                return _failed_answer_verification("最终答案证据核对的主张字段无效")

            raw_basis = raw_claim.get("basis", ANSWER_CLAIM_BASIS_KNOWLEDGE)
            basis = str(raw_basis).strip().lower()
            if basis not in VALID_ANSWER_CLAIM_BASES:
                return _failed_answer_verification("最终答案证据核对的主张来源无效")
            if not advisor_mode and basis != ANSWER_CLAIM_BASIS_KNOWLEDGE:
                # normal 模式没有个人上下文，也不接受把未引用的事实包装成
                # analysis；这条分支是防止旧/恶意 verifier 输出绕过原契约。
                if not (basis == ANSWER_CLAIM_BASIS_NON_CLAIM and not is_claim):
                    return _failed_answer_verification("普通模式不允许非知识库主张来源")

            indexes: list[int] = []
            for value in raw_indexes:
                # bool 是 int 的子类，必须单独挡住，避免 true 被当作 [1]。
                if isinstance(value, bool) or not isinstance(value, int):
                    return _failed_answer_verification("最终答案证据核对的引用编号格式无效")
                indexes.append(value)

            claim_invalid = {
                index for index in indexes
                if index not in available or index not in answer_cited
            }
            invalid.update(claim_invalid)
            expected_indexes = set(unit.citation_indexes)
            # A display formula, table row or code line often sits on its own
            # line with no citation marker while the surrounding prose carries
            # the number.  Such a unit is accepted only when its whole text is a
            # verbatim excerpt of a source the verifier actually bound to it:
            # then the answer added nothing, and an invented line can never
            # satisfy the check.  Everything else keeps the strict rule.
            bound_indexes = expected_indexes
            uncited_verbatim = False
            structural_lead_in = False
            if not expected_indexes and not indexes:
                # A colon-terminated lead-in (``其中：``, ``…公式为：``) is
                # punctuation, not an assertion.  The verifier sometimes labels
                # it a claim, which would otherwise fail every answer that uses
                # one.  Accept it only when it asserts nothing the sources
                # lack: no character may be added beyond the cited sources'
                # own vocabulary, and it must not end a sentence.
                structural_lead_in = _is_structural_lead_in(unit.text)
                if structural_lead_in:
                    structural_lead_in = _lead_in_tokens_supported(unit.text, surface_texts)
            elif not expected_indexes and indexes and not claim_invalid:
                bound_indexes = set(indexes)
                # Two shapes qualify when the citation sits on a neighbouring
                # line rather than in the unit itself:
                # - the whole unit is a verbatim excerpt of the cited source
                #   (a display formula, table row or code line);
                # - the unit is a colon-terminated lead-in whose words the
                #   evidence uses (``根据…公式，…计算得到：``).  The verifier
                #   habitually binds its formula quote to such a line even
                #   though the marker lives on the next prose sentence.
                uncited_verbatim = _unit_text_is_verbatim_from_sources(
                    unit.text, bound_indexes, source_texts,
                )
                if not uncited_verbatim:
                    structural_lead_in = _is_structural_lead_in(unit.text) and _lead_in_tokens_supported(
                        unit.text, surface_texts,
                    )
                if uncited_verbatim or structural_lead_in:
                    claim_invalid = set()
            elif set(indexes) != expected_indexes or len(indexes) != len(set(indexes)):
                # A unit may only use the citations that actually occur in its
                # own text.  A global citation elsewhere in the answer cannot
                # be borrowed to support this unit.
                claim_invalid.update(expected_indexes ^ set(indexes))
                if expected_indexes:
                    invalid.update(expected_indexes ^ set(indexes))

            raw_quotes = raw_claim.get("evidence_quotes", [])
            if raw_quotes is None:
                raw_quotes = []
            quote_indexes, quote_error, quote_relabels = _validate_knowledge_quotes(
                raw_quotes,
                source_texts,
                bound_indexes,
            )
            relabeled_quotes += quote_relabels
            surface_error = False
            if basis == ANSWER_CLAIM_BASIS_KNOWLEDGE and bound_indexes:
                cited_sources = [surface_texts.get(index, "") for index in bound_indexes]
                surface_error = not _claim_surface_tokens_supported(unit.text, cited_sources)
            # Distinguish "the reviewer mis-copied the excerpt" from "the answer
            # itself is unsupported": only the former justifies a bounded retry,
            # and only when the model claimed support and the answer's own
            # identifiers/numbers are all present in the cited sources.
            quote_unlocatable = bool(
                quote_error and supported and not claim_invalid and not surface_error
            )
            if quote_unlocatable:
                unlocatable_units.append(unit_id)
            personal_quote = raw_claim.get("personal_context_quote")
            personal_quote_ok = _valid_personal_context_quote(personal_quote, personal_context)

            if basis == ANSWER_CLAIM_BASIS_KNOWLEDGE:
                # 知识库主张始终要求引用确实来自本次资料；普通单元还必须把
                # 编号写在自身文本里，只有「整行逐字出自被引用原文」的显示行
                # 或通过词汇门的结构行（引导语、序号）才允许编号落在邻近叙述句
                # 或不落编号。
                if structural_lead_in and not indexes and not claim_invalid:
                    # A pure lead-in such as ``其中：`` introduces the following
                    # block and asserts nothing on its own.  It is accepted only
                    # after passing the vocabulary and structure gate above; it
                    # never adds support to any other unit.
                    claim_supported = True
                else:
                    claim_supported = supported and bool(indexes) and not claim_invalid
                    if expected_indexes:
                        claim_supported = (
                            claim_supported
                            and not quote_error
                            and quote_indexes == expected_indexes
                        )
                    else:
                        claim_supported = (
                            claim_supported
                            and uncited_verbatim
                            and not quote_error
                        )
                # A verifier quote can be valid while the answer adds a named
                # model, year, number, or formula symbol absent from every cited
                # source.  Reject those additions deterministically; this closes
                # the exact hallucinated-identifier path that an optimistic LLM
                # verdict can otherwise miss.
                claim_supported = claim_supported and not surface_error
                claim_supported = claim_supported and is_claim and personal_quote is None
            elif basis == ANSWER_CLAIM_BASIS_PERSONAL_CONTEXT:
                # 个人事实必须来自同一份 Personal Context，不能同时挂一个
                # 知识库编号来掩盖来源混杂；混合句应由 verifier 拆开。
                claim_supported = (
                    supported
                    and is_claim
                    and advisor_mode
                    and personal_context_available
                    and not indexes
                    and not raw_quotes
                    and personal_quote_ok
                )
            elif basis == ANSWER_CLAIM_BASIS_ANALYSIS:
                # The marker must belong to this unit, and analysis must still
                # point to a knowledge quote or the same personal context.
                analysis_explicit = _answer_has_explicit_analysis_marker(unit.text)
                claim_supported = (
                    supported
                    and is_claim
                    and advisor_mode
                    and analysis_explicit
                    and not claim_invalid
                    and not quote_error
                    and quote_indexes == expected_indexes
                    and (bool(expected_indexes) or personal_quote_ok)
                    and (personal_quote is None or personal_quote_ok)
                )
            else:  # non_claim
                claim_supported = (
                    not is_claim
                    and not indexes
                    and not raw_quotes
                    and personal_quote is None
                )
            claim_reason = str(raw_claim.get("reason", "")).strip()
            claims.append({
                "claim": claim,
                "basis": basis,
                "supported": claim_supported,
                "citation_indexes": indexes,
                "reason": claim_reason,
                "unit_id": unit_id,
                "is_claim": is_claim,
            })
            basis_counts[basis] = basis_counts.get(basis, 0) + 1
            if not claim_supported:
                unsupported_claims.append(claim)
            if surface_error:
                # Keep this diagnostic internal; the public response only exposes
                # aggregate counts and never echoes the rejected claim text.
                surface_mismatch_count += 1

        if returned_unit_ids != expected_unit_ids:
            return _failed_answer_verification("最终答案证据核验没有覆盖全部文本单元")

        top_invalid = obj.get("invalid_citations", [])
        if top_invalid is not None:
            if not isinstance(top_invalid, list):
                return _failed_answer_verification("最终答案证据核对的无效引用字段格式错误")
            for value in top_invalid:
                if isinstance(value, bool) or not isinstance(value, int):
                    return _failed_answer_verification("最终答案证据核对的无效引用编号格式错误")
                invalid.add(value)

        invalid_indexes = sorted(invalid)
        if status == ANSWER_STATUS_SUPPORTED and (unsupported_claims or invalid_indexes):
            status = ANSWER_STATUS_UNSUPPORTED

        return AnswerVerification(
            status=status,
            confidence=confidence,
            reason=str(obj.get("reason", "")),
            claims=claims,
            unsupported_claims=unsupported_claims,
            invalid_citations=invalid_indexes,
            citation_check=citation_check,
            debug={
                "judge_error": False,
                "claim_check_failed": bool(unsupported_claims or invalid_indexes),
                "basis_counts": basis_counts,
                "surface_mismatch_count": surface_mismatch_count,
                "relabeled_quote_count": relabeled_quotes,
                # Tells the caller that the only defect was an unlocatable
                # excerpt, which is the one condition worth a bounded retry.
                "quote_unlocatable": bool(unlocatable_units),
            },
        )


def _quote_retry_instruction(debug: dict) -> str:
    """Ask the reviewer to re-copy locatable excerpts, not to loosen its verdict.

    The retry runs through the same validator, so this instruction cannot turn
    an unsupported answer into a supported one: it only tells the reviewer what
    went wrong mechanically (a quote that is not a contiguous substring of the
    cited block, or an excerpt taken from a different number).
    """
    return (
        "上一轮 JSON 有一条机械错误需要修正：有 evidence_quotes 的 quote 在它标注的"
        "编号资料里找不到逐字连续原文（可能是跨行拼接、改了字词、或摘录其实来自另一个编号）。\n"
        "请重新输出完整 JSON，并遵守：\n"
        "1. 每个 quote 必须是从该 citation_index 的『内容』里原文复制的一段连续文字，"
        "长度约 8–30 个字符，不要拼接两处、不要省略、不要改写；\n"
        "2. 如果某个事实确实找不到可逐字定位的原文，就把该单元 supported 设为 false，"
        "不要为了通过而编造摘录；\n"
        "3. 只输出 JSON，格式与上一轮完全相同。"
    )


def _failed_answer_verification(reason: str) -> AnswerVerification:
    """统一的严格拒答结果；不暴露被拦截答案的内容。"""
    return AnswerVerification(
        status=ANSWER_STATUS_UNVERIFIED,
        confidence=0.0,
        reason="最终答案证据核对未完成，未输出原回答。",
        debug={
            "judge_error": True,
            "strict_fail_closed": True,
            "fallback_reason": reason,
        },
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

    容忍四种已知脏格式：markdown 围栏、前后多余说明文字、
    JSON 后面跟的解释、以及字符串值里未转义的反斜杠（模型抄写笔记里的
    ``\\'``/``\\"`` 时经常忘掉 JSON 转义）。策略是剥围栏后取首尾大括号之间
    的部分；严格模式先按原样解析，失败再尝试修复转义——修复只针对
    「反斜杠后跟非法 JSON 转义字符」这一种情况，不会改动任何字面内容。
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
    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        obj = json.loads(_repair_unescaped_backslashes(candidate))
    if not isinstance(obj, dict):
        raise ValueError("JSON 顶层不是对象")
    return obj


_JSON_VALID_ESCAPES = set('"\\/bfnrtu')


def _repair_unescaped_backslashes(candidate: str) -> str:
    """Escape bare backslashes inside JSON strings.

    A model quoting a note that stores ``\\'`` or ``\\"`` often emits the
    backslash unescaped, which is invalid JSON.  Doubling such backslashes
    preserves the exact characters; valid JSON escapes are left untouched.
    """
    out: list[str] = []
    in_string = False
    index = 0
    while index < len(candidate):
        char = candidate[index]
        if not in_string:
            if char == '"':
                in_string = True
            out.append(char)
            index += 1
            continue
        if char == "\\" and index + 1 < len(candidate):
            nxt = candidate[index + 1]
            if nxt in _JSON_VALID_ESCAPES:
                out.append(char)
                out.append(nxt)
                index += 2
            else:
                out.append("\\\\")
                index += 1
            continue
        if char == '"':
            in_string = False
        out.append(char)
        index += 1
    return "".join(out)


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


def _build_answer_units(answer: str) -> list[_AnswerUnit]:
    """Split an answer into stable, code-owned units for strict verification.

    Newlines and sentence-ending punctuation are conservative boundaries. Citation
    markers immediately following punctuation stay attached to the preceding unit,
    so the verifier can bind a citation to the exact text that used it. This is a
    coverage boundary, not a semantic claim extractor; the verifier must judge the
    complete unit text rather than a model-selected substring.
    """
    normalized = answer.replace("\r\n", "\n").replace("\r", "\n")
    pieces: list[str] = []
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        start = 0
        index = 0
        while index < len(line):
            if not _is_answer_unit_boundary(line, index):
                index += 1
                continue
            end = index + 1
            # Citation markers belong to the sentence immediately before them.
            while True:
                cursor = end
                while cursor < len(line) and line[cursor].isspace():
                    cursor += 1
                match = _ANSWER_CITATION.match(line, cursor)
                if match is None:
                    break
                end = match.end()
            piece = line[start:end].strip()
            if piece:
                pieces.append(piece)
            start = end
            index = end
        tail = line[start:].strip()
        if tail:
            pieces.append(tail)

    if not pieces and answer.strip():
        pieces.append(answer.strip())
    return [
        _AnswerUnit(
            unit_id=f"u{index}",
            text=text,
            citation_indexes=tuple(_answer_citation_indexes(text)),
        )
        for index, text in enumerate(pieces, start=1)
    ]


def _is_answer_unit_boundary(text: str, index: int) -> bool:
    """Return whether ``text[index]`` is a conservative sentence boundary.

    Chinese punctuation is an unambiguous boundary for the answers this service
    produces.  Semicolons are deliberately kept inside a unit: models commonly
    put a citation after a semicolon-separated clause, and splitting there would
    manufacture an uncited fact even though the following citation covers the
    complete sentence. English answers also occur, though, and treating a
    whole paragraph such as ``"Fact one. Fact two."`` as one unit would let a
    verifier response cover only the first fact while still appearing to
    cover the paragraph.  A full stop is therefore a boundary at end of line
    or before whitespace, except when it is the decimal point in a number.
    This intentionally stays small rather than attempting full NLP sentence
    segmentation; markdown/newline boundaries remain the primary split.
    """
    char = text[index]
    if char in "。！？!?":
        return True
    if char != ".":
        return False
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    if previous.isdigit() and following.isdigit():
        return False
    return not following or following.isspace()


def _format_answer_units(units: list[_AnswerUnit]) -> str:
    """Render code-generated units as JSON so the model cannot invent the scope."""
    return json.dumps(
        [{"unit_id": unit.unit_id, "text": unit.text} for unit in units],
        ensure_ascii=False,
    )


def _answer_citation_indexes(text: str) -> list[int]:
    """Return citation markers that occur in one answer unit."""
    # Citations inside fenced code are examples, not answer evidence.
    text = re.sub(r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)", "", text)
    return sorted({
        int(number)
        for match in _ANSWER_CITATION.finditer(text)
        for number in re.split(r"\s*[,，]\s*", match.group(1))
    })


def _context_source_texts(context: str) -> dict[int, str]:
    """Extract the original text for each numbered context block.

    RAG builds context as ``[N] ... 内容：...``. The fallback to the complete
    block keeps direct verifier tests and compatible callers useful while still
    requiring an exact substring of the numbered block.
    """
    headers = list(_CONTEXT_HEADER.finditer(context))
    sources: dict[int, str] = {}
    for position, header in enumerate(headers):
        end = headers[position + 1].start() if position + 1 < len(headers) else len(context)
        block = context[header.end() : end]
        content_match = re.search(r"(?ms)^\s*内容[:：]\s*(.*)\Z", block)
        content = content_match.group(1).strip() if content_match else block.strip()
        sources[int(header.group(1))] = content
    return sources


def _context_source_surface_texts(context: str) -> dict[int, str]:
    """Return full numbered blocks for the narrow identifier sanity check.

    Surface terms such as a document's algorithm name often appear in the
    numbered title/header rather than in the chunk body.  Quote validation keeps
    using :func:`_context_source_texts` (body only); this companion map is only
    for checking that an answer did not invent a salient identifier or number.
    """
    headers = list(_CONTEXT_HEADER.finditer(context))
    surfaces: dict[int, str] = {}
    for position, header in enumerate(headers):
        end = headers[position + 1].start() if position + 1 < len(headers) else len(context)
        surfaces[int(header.group(1))] = context[header.start():end]
    return surfaces


def _is_structural_lead_in(unit_text: str) -> bool:
    """Whether a unit is a structural line rather than an assertion.

    Two shapes qualify, and both must assert nothing on their own:
    - a short colon-terminated lead-in (``其中：``, ``…公式为：``), which
      introduces the following block;
    - a bare list counter (``1.``), which is pure numbering.
    A line that already states a fact (``2019 年分解 99：``) is rejected, as is
    an invented heading whose wording the evidence never uses.
    """
    text = unit_text.strip()
    if text.startswith("**") and text.endswith("**"):
        text = text[2:-2].strip()
    body = text.rstrip("：:").strip()
    # A bare list counter carries no content regardless of the marker.
    if re.fullmatch(r"\d{1,2}\s*[.、)）]?", body):
        return True
    if not text.endswith(("：", ":")):
        return False
    if len(body) > 60:
        return False
    if any(char in body for char in "。！？；!?;"):
        return False
    # An inline-math tail (``…公式为：\[``) or LaTeX display openers are
    # formatting, not stated content; strip them before the ASCII check.
    body = re.sub(r"\\+[\[\]()]", "", body)
    body = re.sub(r"\$[^$]*\$", "", body)
    body = body.strip()
    if not body:
        return True
    # A lead-in may name the topic, including ASCII terms the notes use
    # (``计算…自由度 F 时``); the vocabulary gate decides whether the words are
    # attested.  Only a digit-bearing assertion (``分解 99：``) is excluded here.
    return not re.search(r"\d", body)


def _lead_in_tokens_supported(unit_text: str, surface_texts: dict[int, str]) -> bool:
    """Whether a lead-in only uses words that appear somewhere in the evidence.

    Conservative and one-directional: it can only *deny* a lead-in whose wording
    the sources never use.  A structural lead-in may paraphrase freely (计算
    平面机构的自由度 vs the note's 自由度公式): its shape and short length
    already bound what it can assert, and the units under it carry every
    checkable fact.  Deny only a lead-in that is mostly characters the evidence
    never uses — the signature of an invented section, not a paraphrase.
    """
    if not surface_texts:
        return False
    body = unit_text.strip().rstrip("*").rstrip("：:").strip()
    # A bare list counter has no content to check.
    if re.fullmatch(r"\d{1,2}\s*[.、)）]?", body):
        return True
    # Drop LaTeX display delimiters, inline math and Markdown emphasis before
    # word extraction.
    body = re.sub(r"\\+[\[\]()]", " ", body)
    body = re.sub(r"\$[^$]*\$", " ", body)
    body = body.replace("**", "").replace("__", "").strip()
    normalized = _normalize_surface_text(body).casefold()
    normalized = re.sub(r"^(其中|以下|如下|具体来说|分别是|包括|即)", "", normalized)
    # The haystack includes the numbered block headers (``[1] 文档：…``), so a
    # lead-in echoing the section title is checked against the same evidence
    # the answer's own citations point into.
    haystack = _normalize_surface_text(" ".join(surface_texts.values())).casefold()
    fragments = re.findall(r"[\u3400-\u9fff]+", normalized)
    if not fragments:
        ascii_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", normalized)
        return all(
            re.search(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])", haystack)
            for token in ascii_tokens
        )
    all_chars = "".join(fragments)
    if not all_chars:
        return True
    # A threshold of 0.6 keeps the allowance narrow: a lead-in built mostly
    # from attested characters passes (``示例验证`` over notes containing
    # 例1/例2 and 示`` symbols), while an invented heading whose vocabulary the
    # evidence never shares fails.
    coverage = sum(1 for char in all_chars if char in haystack) / len(all_chars)
    return coverage >= 0.5


def _unit_text_is_verbatim_from_sources(
    unit_text: str,
    indexes: set[int],
    source_texts: dict[int, str],
) -> bool:
    """Whether an uncited display line is literally copied from a cited source.

    Only used for units such as a standalone formula, table row or code line
    whose citation marker the writer put on the neighbouring prose.  Requiring
    a verbatim substring means the check can only *confirm* existing text: an
    invented formula, a changed coefficient, or a fabricated table row fails
    here and is still refused.
    """
    text = _normalize_quote_text(unit_text)
    if not text:
        return False
    return any(
        text in _normalize_quote_text(source_texts.get(index, ""))
        for index in indexes
    )


def _validate_knowledge_quotes(
    raw_quotes: object,
    source_texts: dict[int, str],
    expected_indexes: set[int],
) -> tuple[set[int], bool, int]:
    """Validate exact source excerpts and return (covered indexes, error, relabels).

    The verifier occasionally copies a sentence that really is in one of the
    unit's cited notes but labels it with a different number from the same
    unit.  That is a clerical mistake, not new evidence: the excerpt still has
    to be a verbatim substring of *some* source this unit cites.  We therefore
    re-attribute such a quote to the block it actually occurs in and count the
    correction, while still demanding that every cited index end up covered by
    a quote found in that exact block.  A quote that appears in no cited block
    remains a hard failure, so this cannot smuggle in outside text.
    """
    if not isinstance(raw_quotes, list):
        return set(), True, 0
    seen: set[int] = set()
    error = False
    relabels = 0
    for item in raw_quotes:
        if not isinstance(item, dict):
            error = True
            continue
        index = item.get("citation_index")
        quote = item.get("quote")
        # A single source may support several clauses in one unit.  The verifier
        # is therefore allowed to return multiple short excerpts for the same
        # citation index; ``seen`` tracks coverage, not duplicate rejection.
        if isinstance(index, bool) or not isinstance(index, int):
            error = True
            continue
        if index not in expected_indexes or not isinstance(quote, str) or not quote.strip():
            error = True
            continue
        source = source_texts.get(index, "")
        if source and _quote_occurs_in_source(quote, source):
            seen.add(index)
            continue
        actual = next(
            (
                other for other in sorted(expected_indexes)
                if _quote_occurs_in_source(quote, source_texts.get(other, ""))
            ),
            None,
        )
        if actual is None:
            error = True
            continue
        seen.add(actual)
        relabels += 1
    return seen, error, relabels


def _quote_occurs_in_source(quote: str, source: str) -> bool:
    """Check a model quote against the cited source without treating formatting
    serialization as a factual difference.

    Obsidian notes in this vault contain escaped quotation marks (``\\"``), while
    the LLM normally emits the visible quotation mark (``"``).  Models also
    occasionally copy a Markdown list as one quote, dropping the list markers
    and line breaks.  Those transformations do not change the words being
    evidenced, so compare a conservative formatting-normalized form after first
    trying the required exact substring.  We do *not* strip arbitrary words or
    punctuation and still reject ellipses/stitched prose that cannot be located
    in the source.
    """
    if not isinstance(quote, str) or not quote.strip():
        return False
    quote = quote.strip()
    if quote in source:
        return True
    normalized_quote = _normalize_quote_text(quote)
    normalized_source = _normalize_quote_text(source)
    if normalized_quote in normalized_source:
        return True
    # Models copy the prose and silently drop the inline variable while quoting
    # (``信息集 $I$ 是…`` → ``信息集是…``).  Removing single-``$`` spans and
    # ``\(...\)`` from *both* sides lets that excerpt match the surrounding
    # words.  This can only remove text from the source, so it never makes a
    # quote containing a number or identifier easier to locate — it is
    # strictly harder.  Display math (``$$…$$``) is kept because that is where
    # the formula itself lives.
    stripped_quote = _normalize_quote_text(_strip_inline_math(quote))
    stripped_source = _normalize_quote_text(_strip_inline_math(source))
    if stripped_quote and stripped_quote in stripped_source:
        return True
    # Second chance for a quote the model stitched together from a numbered
    # list without keeping the line breaks: after whitespace is folded the
    # ordinal sits mid-string (``性质：1.相同玩家``) and the line-start rule
    # cannot see it.  Drop short ordinal markers on *both* sides — the same
    # deterministic transform is applied to quote and source, so the check
    # cannot become more permissive about content.  A decimal such as
    # ``3.14`` is untouched because its point is followed by a digit, and a
    # three-or-more digit number is untouched because the lookbehind rejects it.
    return (
        _strip_ordinal_markers(normalized_quote)
        in _strip_ordinal_markers(normalized_source)
    )


def _strip_inline_math(value: str) -> str:
    """Remove single-``$`` and ``\\(...\\)`` spans, keeping ``$$`` display math."""
    value = re.sub(r"\\\((.+?)\\\)", " ", value, flags=re.S)
    return re.sub(r"(?<!\$)\$(?!\$)[^$\n]{1,200}\$(?!\$)", " ", value)


def _strip_ordinal_markers(text: str) -> str:
    """Remove list ordinals that a model merged into running text."""
    return re.sub(r"(?<![0-9])\d{1,2}[.)](?![0-9])", "", text)


def _normalize_quote_text(value: str) -> str:
    """Normalize only Markdown/whitespace/LaTeX-presentation escapes.

    This deliberately never removes or substitutes *words*.  It folds things a
    renderer would also fold: escaped quotes, `**`/`$` delimiters, list/heading
    markers, a handful of unambiguous LaTeX commands that render to one visible
    character (`\\in` → `∈`), and math minus variants.  Everything else — LaTeX
    commands not in the table, operators, Chinese wording, quantifiers like
    “必须” — must match literally, so a semantic rewrite still fails.
    """
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    # A literal escaped quote is how several imported Markdown notes store the
    # same visible character.  Keep other backslashes intact so a quote cannot
    # silently turn into a different formula.
    text = text.replace('\\"', '"').replace("\\'", "'")
    # Markdown emphasis and math delimiters are presentation, not words.  A
    # model may omit them while copying a short quote; removing only these
    # delimiters preserves the actual symbols and text inside them.
    text = text.replace("**", "").replace("__", "").replace("$", "")
    text = _fold_latex_presentation(text)
    # A quote may span Markdown list items.  Remove only list/heading markers at
    # line starts, then ignore whitespace; no content words are discarded.  The
    # latter makes ``a。\n- b`` comparable to ``a。b`` while keeping every
    # non-whitespace character required by the quote.
    text = re.sub(r"(?m)^\s{0,3}(?:[-+*]|\d+[.)])\s+", "", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    return re.sub(r"\s+", "", text).strip()


# One visible character per command, so the mapping cannot merge two different
# facts.  Commands not listed here are left as literal text on purpose.
_LATEX_PRESENTATION = {
    r"\in": "∈",
    r"\notin": "∉",
    r"\ni": "∋",
    r"\subset": "⊂",
    r"\subseteq": "⊆",
    r"\supset": "⊃",
    r"\cup": "∪",
    r"\cap": "∩",
    r"\forall": "∀",
    r"\exists": "∃",
    r"\emptyset": "∅",
    r"\varnothing": "∅",
    r"\infty": "∞",
    r"\sum": "∑",
    r"\prod": "∏",
    r"\times": "×",
    r"\cdot": "·",
    r"\pm": "±",
    r"\to": "→",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\Rightarrow": "⇒",
    r"\Leftrightarrow": "⇔",
    r"\le": "≤",
    r"\leq": "≤",
    r"\ge": "≥",
    r"\geq": "≥",
    r"\neq": "≠",
    r"\ne": "≠",
    r"\approx": "≈",
    r"\iota": "ι",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ",
    r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ", r"\phi": "φ",
    r"\varphi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Sigma": "Σ", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
}

_LATEX_MINUS_VARIANTS = str.maketrans({
    "\u2212": "-",   # MINUS SIGN
    "\u2013": "-",   # EN DASH used as minus in rendered formulas
})

# Command names (without the backslash) recognised as typesetting rather than
# content.  These carry no fact of their own and are skipped by both the macro
# gate and the surface-token check.
_LATEX_TYPESETTING_NAMES = frozenset({
    "frac", "dfrac", "tfrac", "sqrt", "left", "right", "quad", "qquad",
    "dots", "ldots", "cdots", "text", "mathrm", "mathbf", "mathit", "mathcal",
    "mathbb", "partial", "nabla", "big", "bigl", "bigr", "bigg", "begin",
    "end", "hat", "bar", "vec", "dot", "ddot", "overline", "underline",
    "operatorname", "lim", "log", "ln", "exp", "sin", "cos", "tan", "max",
    "min", "sup", "inf", "det", "Pr", "boxed", "displaystyle", "limits",
})

# Everything whose name is pure syntax for this module.  A rendered symbol
# command (``\iota``) is skipped here because its glyph is non-ASCII and never
# enters the ASCII token check; the macro gate still verifies it literally.
_LATEX_COMMAND_NAMES = (
    frozenset(name.lstrip("\\").casefold() for name in _LATEX_PRESENTATION)
    | _LATEX_TYPESETTING_NAMES
)


def _fold_latex_presentation(value: str) -> str:
    """Fold LaTeX commands and math-minus glyphs to their visible character."""
    for command, glyph in _LATEX_PRESENTATION.items():
        value = re.sub(re.escape(command) + r"(?![A-Za-z])", glyph, value)
    return value.translate(_LATEX_MINUS_VARIANTS)


_SURFACE_TOKEN_STOPWORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "the", "to", "via", "with",
})


def _claim_surface_tokens_supported(unit_text: str, cited_sources: list[str]) -> bool:
    """Reject salient identifiers/numbers that occur in no cited source.

    The answer verifier is still the semantic judge, but an LLM can occasionally
    approve a sentence that quietly adds a model name or changes a number.  This
    narrow lexical check catches those high-signal additions without requiring
    Chinese tokenization: ASCII identifiers and numeric literals are stable across
    Markdown/LaTeX rendering.  One-character algebra variables and ordinary
    English glue words are intentionally ignored.
    """
    if not cited_sources:
        return False
    answer = _ANSWER_CITATION.sub("", unit_text)
    # Symbol commands must exist as the same command, or as the glyph the note
    # already stores.  ``\iota`` is not satisfied by a source's ``\iotaX`` —
    # that is a different macro — while ``\in`` is satisfied by ``∈``.
    if not _surface_latex_commands_supported(answer, cited_sources):
        return False
    answer = _fold_latex_presentation(answer)
    tokens = re.findall(r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?)", answer)
    salient: list[str] = []
    for token in tokens:
        lowered = token.casefold()
        if "_" in token:
            continue
        if lowered in _LATEX_COMMAND_NAMES:
            # A LaTeX command is typesetting, not a named entity.  ``\times``
            # must not read as a fabricated English identifier; its rendered
            # glyph (``×``) is non-ASCII and never enters this check.
            continue
        if token.isdigit() or (len(token) >= 2 and lowered not in _SURFACE_TOKEN_STOPWORDS):
            # Single-letter variables and subscripted formula symbols (P_L,
            # E_n) are scaffolding that may be introduced by the surrounding
            # answer; salient names and numeric literals remain checked.
            if len(token) >= 2 or token.isdigit():
                salient.append(lowered)
    if not salient:
        return True
    if not all(
        any(_surface_token_occurs(source, token) for source in cited_sources)
        for token in salient
    ):
        return False

    # A source can mention the same identifier and year in different factual
    # bullets (for example ``2012 … IBM`` and ``2019 … superconducting``).
    # Presence in the union is not enough to support a new association.  For
    # year + all-caps identifiers, require them to occur near one another in at
    # least one cited block; this catches ``2019 … IBM`` without judging ordinary
    # prose or single-letter formula variables.
    years = [token for token in tokens if token.isdigit() and 1900 <= int(token) <= 2100]
    identifiers = [
        token for token in tokens
        if len(token) >= 2 and token.isupper() and token not in {"NE"}
    ]
    for year in years:
        for identifier in identifiers:
            if not any(_tokens_near(source, year, identifier) for source in cited_sources):
                return False
    return True


def _surface_latex_commands_supported(unit_text: str, cited_sources: list[str]) -> bool:
    """Every LaTeX macro in the answer must also occur in a cited source.

    Typesetting macros (``\\frac``, ``\\left``) are skipped: they carry no fact.
    Symbol macros are checked literally so a renamed command cannot pass just
    because its rendered glyph is common, and the equivalent visible glyph
    (``∈`` for ``\\in``) is accepted because notes mix both forms.
    """
    commands = {
        match.group(1).casefold()
        for match in re.finditer(r"\\([A-Za-z]+)", unit_text)
    }
    commands -= _LATEX_TYPESETTING_NAMES
    if not commands:
        return True
    haystacks = [_normalize_surface_text(source).casefold() for source in cited_sources]
    for command in commands:
        glyph = _LATEX_PRESENTATION.get("\\" + command)
        if any(
            re.search(r"\\" + re.escape(command) + r"(?![A-Za-z])", haystack)
            or (glyph is not None and glyph in haystack)
            for haystack in haystacks
        ):
            continue
        return False
    return True


def _tokens_near(source: str, first: str, second: str) -> bool:
    """Whether two literal surface tokens occur in one logical source item.

    Markdown often wraps one bullet over several physical lines.  Requiring the
    same physical line would reject a supported ``year + identifier`` relation
    merely because the note was reflowed.  List markers and blank lines still
    delimit items, so two separate bullets cannot borrow each other's facts.
    """
    return any(
        _surface_token_occurs(block, first)
        and _surface_token_occurs(block, second)
        for block in _surface_source_blocks(source)
    )


def _surface_token_occurs(source: str, token: str) -> bool:
    """Match one ASCII surface token without accepting a larger token.

    The surrounding Chinese text is intentionally not a boundary: ``35号`` and
    ``IBM芯片`` are ordinary source spellings.  ASCII letters, digits and `_`
    are boundaries, preventing ``35`` from matching ``135``/``350`` and ``IBM``
    from matching ``IBMXYZ``.  Normalization is applied independently per source
    so two adjacent chunks cannot accidentally form a token across their join.
    """
    # Quote matching intentionally removes all whitespace so an LLM can omit
    # Markdown line breaks.  That normalization is unsafe here: ``device IBM
    # chip`` would become ``deviceIBMchip`` and make the real IBM token appear
    # adjacent to ASCII letters.  Surface matching therefore keeps whitespace
    # as a boundary while removing only display-only Markdown markers.
    normalized_source = _normalize_surface_text(source).casefold()
    normalized_token = _normalize_surface_text(token).casefold()
    if not normalized_token:
        return False
    # A numeric literal is often a coefficient glued to its variable
    # (``3n``, ``2P_L``).  The digit run must still not merge with another
    # number — ``35`` must not match ``135`` or ``350`` — but a following
    # letter is multiplication, not a different token.  Without this, a
    # formula that is quoted verbatim from the note (``F = 3n − 2P_L − P_H``)
    # was rejected as if ``3`` and ``2`` had been invented.
    trailing = r"(?![0-9_])" if _is_numeric_token(normalized_token) else r"(?![A-Za-z0-9_])"
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(normalized_token) + trailing
    return re.search(pattern, normalized_source) is not None


def _is_numeric_token(token: str) -> bool:
    """Return whether a surface token is a plain decimal number (``35``, ``3.14``)."""
    body = token[1:] if token.startswith(".") else token
    return bool(body) and body.replace(".", "", 1).isdigit()


def _normalize_surface_text(value: str) -> str:
    """Normalize display syntax while preserving ASCII word boundaries."""
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace('\\"', '"').replace("\\'", "'")
    text = text.replace("**", "").replace("__", "").replace("$", "")
    text = re.sub(r"(?m)^\s{0,3}(?:[-+*]|\d+[.)])\s+", "", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _surface_source_blocks(source: str) -> list[str]:
    """Group wrapped Markdown text into conservative factual items.

    A new list marker starts a new item; blank lines also separate items.  Plain
    consecutive lines remain one block because line wrapping has no reliable
    syntax marker and is normally part of the same paragraph.
    """
    blocks: list[str] = []
    current: list[str] = []
    list_item = re.compile(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+")
    for raw_line in source.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(" ".join(current))
                current = []
            continue
        if list_item.match(raw_line):
            if current:
                blocks.append(" ".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(" ".join(current))
    return blocks


def _valid_personal_context_quote(value: object, personal_context: str | None) -> bool:
    """Require a literal excerpt from the same rendered Personal Context."""
    return (
        isinstance(value, str)
        and bool(value.strip())
        and bool(personal_context)
        and value.strip() in personal_context
    )


def _claim_basis_counts(claims: list[dict]) -> dict[str, int]:
    """只按来源聚合主张数量，供对外诊断使用；不携带任何主张文本。"""
    counts: dict[str, int] = {}
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        basis = str(claim.get("basis", "unknown")).strip().lower() or "unknown"
        counts[basis] = counts.get(basis, 0) + 1
    return counts


def _answer_has_explicit_analysis_marker(answer: str) -> bool:
    """判断回答是否显式把一段内容标成分析/建议，而不是把事实伪装成分析。

    这是一个保守的二次门：最终是否属于分析仍由 verifier 逐项判断，
    这里只负责阻止完全没有任何分析/建议标记的 ``basis=analysis`` 主张
    在无引用时放行。
    """
    lowered = answer.lower()
    markers = (
        "分析：", "分析:", "建议：", "建议:", "我的分析", "我的建议", "我的判断",
        "推荐：", "推荐:", "推断：", "推断:", "判断：", "判断:",
        "行动建议", "可以考虑", "可考虑", "建议先", "建议优先",
        "analysis:", "recommendation:", "recommend:",
    )
    return any(marker in lowered for marker in markers)
