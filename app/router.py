"""Query Router：判断一个问题应该到哪些知识域里检索。

第一版是纯规则，不调 LLM、不做 Agent：问题里命中了哪个域的术语，就路由到哪个域；
一个都没命中就 global（全库检索）。术语表在 config/knowledge_domains.yaml 的
router_terms 段，代码里不出现任何具体领域名。

设计取向是保守（宁漏勿误）：
    - 漏判的代价 = 退回全库检索，结果不会更差，只是没省事；
    - 误判的代价 = 把真答案过滤在候选池外，那是不可挽回的错误。
所以「出现"合作"不等于 game_theory」——只有足够特异的术语和实体才进术语表。

mode 只有三种：single_domain / multi_domain / global。confidence 是规则版的
粗略自估（按模式给固定档位），不是概率；reason 记录命中的术语，只用于 debug，
下游逻辑不允许依赖它的自然语言内容。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import BaseLLM

# Router 的三种输出模式。global 时 domains 恒为空列表，下游据此走全库检索。
MODE_SINGLE = "single_domain"
MODE_MULTI = "multi_domain"
MODE_GLOBAL = "global"

# 每种模式的 confidence 固定档位。规则版没有校准过的概率可言，
# 给出的是"按模式分档的自估"，用法是排序/打日志，不是当真分数比大小。
CONFIDENCE_SINGLE = 0.85
CONFIDENCE_MULTI = 0.75
CONFIDENCE_GLOBAL = 0.40

# RouteDecision.origin：这条决策是规则路由出来的，还是规则零命中后
# LLM fallback 给出的，或是 LLM fallback 失败后兜底的 global。
# 下游逻辑只允许用 mode/domains，origin 仅用于 debug 和 eval 统计。
ORIGIN_RULES = "rules"
ORIGIN_LLM_FALLBACK = "llm_fallback"
ORIGIN_LLM_FALLBACK_ERROR = "llm_fallback_error"


@dataclass
class RouteDecision:
    """一次路由的结果。reason 只用于 debug，不参与任何逻辑判断。"""

    mode: str
    domains: list[str] = field(default_factory=list)
    confidence: float = CONFIDENCE_GLOBAL
    reason: str = ""
    origin: str = ORIGIN_RULES


class QueryRouter:
    """术语表路由器。domains -> terms 的映射来自 YAML，构造时注入。"""

    def __init__(self, router_terms: dict[str, list[str]]):
        self._router_terms = router_terms

    def route(self, question: str) -> RouteDecision:
        """按术语命中决定路由。大小写不敏感的子串匹配。"""
        question_lower = question.lower()
        # domain -> 命中的术语。用 dict 保序，输出领域顺序稳定，测试才可复现。
        hits: dict[str, list[str]] = {}
        for domain, terms in self._router_terms.items():
            matched = [term for term in terms if term.lower() in question_lower]
            if matched:
                hits[domain] = matched

        if not hits:
            return RouteDecision(
                mode=MODE_GLOBAL,
                domains=[],
                confidence=CONFIDENCE_GLOBAL,
                reason="没有命中任何领域术语，回退全库检索",
            )
        if len(hits) == 1:
            domain, matched = next(iter(hits.items()))
            return RouteDecision(
                mode=MODE_SINGLE,
                domains=[domain],
                confidence=CONFIDENCE_SINGLE,
                reason=f"命中 {domain} 术语: {'、'.join(matched)}",
            )
        return RouteDecision(
            mode=MODE_MULTI,
            domains=list(hits),
            confidence=CONFIDENCE_MULTI,
            reason="；".join(f"{d}: {'、'.join(ts)}" for d, ts in hits.items()),
        )

    @property
    def domain_names(self) -> list[str]:
        """术语表覆盖的全部 domain 名，供 LLM fallback 做合法域校验。"""
        return list(self._router_terms)


# ============================================================
# LLM Router fallback（阶段：Router Fallback）
#
# 触发纪律：只有规则 Router 零命中（mode=global）时才允许调 LLM——
# 规则命中时 LLM 一律不介入，规则是第一优先级，这条由 route_with_fallback
# 统一保证，调用方没有机会绕过。
#
# LLM 的权限约束：只能从 allowed_domains（来自 YAML 的真实 domain 清单）
# 里选，选单域或多域；拿不准就返回空列表 = global。宁可 global 不可误限，
# 所以 prompt 里明确告诉它「不确定就选 global」。
#
# 可信度处理：LLM 返回的 domains 全部与 allowed_domains 求交，域外的名字
# （幻觉出来的新领域）直接丢弃；mode 不信 LLM 自己写的，按最终域数重新推导。
# ============================================================

LLM_ROUTER_SYSTEM_PROMPT = """你是知识库问答系统的查询路由器。给定用户问题和一个知识域清单，判断这个问题应该到哪些知识域里检索。

规则：
1. 只能从给出的知识域清单中选择，一个或多个；禁止创造清单之外的新领域。
2. 只选择和问题真正相关的领域，不要贪多。
3. 如果拿不准、问题太宽泛、或清单里没有合适的领域，返回空列表（系统会回退全库检索）——这比选错领域安全。

只输出一个 JSON 对象，不要输出任何其它文字。格式：
{"domains": ["清单中的领域名"], "confidence": 0.0到1.0的小数, "reason": "一句话理由"}"""

LLM_ROUTER_USER_TEMPLATE = """【用户问题】
{question}

【可选知识域清单】
{domains}

这个问题应该到哪些知识域里检索？只输出 JSON。"""


def route_with_fallback(
    router: QueryRouter,
    question: str,
    allowed_domains: list[str],
    llm: BaseLLM | None = None,
    fallback_enabled: bool = False,
) -> RouteDecision:
    """统一入口：先走规则，规则零命中且开关开启时才进 LLM fallback。

    allowed_domains 从 YAML 的 knowledge_domains 段取（真实存在的域全集），
    比术语表的域更全——术语表只配了「能靠词识别」的域。
    llm 为 None 或开关关闭时这个函数就等价于 router.route(question)。
    """
    decision = router.route(question)
    if decision.mode != MODE_GLOBAL or not fallback_enabled or llm is None:
        return decision
    return llm_route_fallback(llm, question, allowed_domains)


def llm_route_fallback(llm: BaseLLM, question: str, allowed_domains: list[str]) -> RouteDecision:
    """规则零命中时让 LLM 选域。任何失败都返回 global 兜底，不让路由层挂掉。"""
    allowed = list(dict.fromkeys(allowed_domains))  # 去重且保序
    if not allowed:
        # 没有可选域就没必要花一次 LLM 调用，直接 global
        return RouteDecision(
            mode=MODE_GLOBAL, reason="没有可选知识域，跳过 LLM fallback",
            origin=ORIGIN_LLM_FALLBACK_ERROR,
        )

    messages = [
        {"role": "system", "content": LLM_ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": LLM_ROUTER_USER_TEMPLATE.format(
            question=question, domains="\n".join(f"- {d}" for d in allowed),
        )},
    ]
    try:
        raw = llm.chat(messages)
    except Exception as exc:
        return RouteDecision(
            mode=MODE_GLOBAL, reason=f"LLM Router fallback 调用失败：{exc}",
            origin=ORIGIN_LLM_FALLBACK_ERROR,
        )

    domains, confidence, reason, rejected = _parse_llm_domains(raw, allowed)
    if not domains:
        return RouteDecision(
            mode=MODE_GLOBAL, confidence=confidence,
            reason=reason + ("；丢弃域外名称: " + "、".join(rejected) if rejected else "，LLM 未选择任何领域"),
            origin=ORIGIN_LLM_FALLBACK,
        )

    mode = MODE_SINGLE if len(domains) == 1 else MODE_MULTI
    return RouteDecision(
        mode=mode, domains=domains, confidence=confidence,
        reason=reason + ("；丢弃域外名称: " + "、".join(rejected) if rejected else ""),
        origin=ORIGIN_LLM_FALLBACK,
    )


def _parse_llm_domains(raw: str, allowed: list[str]) -> tuple[list[str], float, str, list[str]]:
    """解析 LLM 的域选择，返回 (合法域, confidence, reason, 被剔除的域外名称)。

    解析不出 JSON 时返回空域列表——调用方据此落 global，而不是把乱码当路由。
    """
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return [], 0.0, "LLM 输出里没有 JSON", []
    try:
        obj = json.loads(text[start : end + 1])
    except ValueError:
        return [], 0.0, "LLM 输出不是合法 JSON", []
    if not isinstance(obj, dict):
        return [], 0.0, "LLM 输出的 JSON 顶层不是对象", []

    raw_domains = obj.get("domains")
    if not isinstance(raw_domains, list):
        raw_domains = []
    # 与 allowed 求交：域外名字（LLM 幻觉）丢弃并记录，交给 reason 供 debug。
    picked = [str(d).strip() for d in raw_domains if str(d).strip()]
    valid = [d for d in picked if d in allowed]
    rejected = [d for d in picked if d not in allowed]
    # 再去一次重：LLM 可能在合法清单里重复报同一个域
    valid = list(dict.fromkeys(valid))

    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    reason = str(obj.get("reason", ""))
    return valid, confidence, reason, rejected
