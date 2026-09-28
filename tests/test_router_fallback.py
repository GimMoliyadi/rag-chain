"""LLM Router fallback 单元测试。

核心契约：
- 规则命中时 LLM 一律不介入（用空脚本假 LLM 断言：任何调用即测试失败）；
- 只有规则零命中且开关开启时才调 LLM；
- LLM 只能从合法域清单里选，域外名称（幻觉）被丢弃并记录；
- mode 由最终域数推导，不信 LLM 自己写的 mode 字段；
- 任何失败都落 global 兜底，路由层永不挂掉。
"""
from __future__ import annotations

from app.router import (
    MODE_GLOBAL,
    MODE_MULTI,
    MODE_SINGLE,
    ORIGIN_LLM_FALLBACK,
    ORIGIN_LLM_FALLBACK_ERROR,
    ORIGIN_RULES,
    QueryRouter,
    route_with_fallback,
)
from fakes import ScriptedLLM

# 术语表只配两个域；allowed 清单刻意比术语表多两个域，
# 模拟「mapping 全集 > 术语表覆盖」的真实关系。
ROUTER_TERMS = {"game_theory": ["博弈论"], "philosophy": ["临在"]}
ALLOWED = ["game_theory", "philosophy", "fiction", "unknown"]


def _llm_domain_response(domains: list, confidence: float = 0.8, **extra) -> str:
    import json
    payload = {"domains": domains, "confidence": confidence, "reason": "按问题语义判断", **extra}
    return json.dumps(payload, ensure_ascii=False)


def test_rule_hit_never_calls_llm():
    router = QueryRouter(ROUTER_TERMS)
    llm = ScriptedLLM([])  # 空脚本：任何 LLM 调用都会让测试失败

    decision = route_with_fallback(
        router, "博弈论里的囚徒困境是什么？", ALLOWED,
        llm=llm, fallback_enabled=True,
    )

    assert decision.mode == MODE_SINGLE
    assert decision.domains == ["game_theory"]
    assert decision.origin == ORIGIN_RULES
    assert llm.calls == []


def test_rule_zero_hit_with_fallback_disabled_stays_global():
    router = QueryRouter(ROUTER_TERMS)
    llm = ScriptedLLM([])

    decision = route_with_fallback(router, "随便一个问题", ALLOWED, llm=llm, fallback_enabled=False)

    assert decision.mode == MODE_GLOBAL
    assert decision.origin == ORIGIN_RULES
    assert llm.calls == []


def test_zero_hit_with_fallback_enabled_uses_llm():
    router = QueryRouter(ROUTER_TERMS)
    llm = ScriptedLLM([_llm_domain_response(["philosophy"])])

    decision = route_with_fallback(
        router, "冥想的时候注意力总是飘走怎么办？", ALLOWED,
        llm=llm, fallback_enabled=True,
    )

    assert decision.mode == MODE_SINGLE
    assert decision.domains == ["philosophy"]
    assert decision.origin == ORIGIN_LLM_FALLBACK
    # 第一次也是唯一一次 LLM 调用是路由，prompt 里带可选域清单
    assert "冥想的时候注意力总是飘走怎么办？" in llm.calls[0][1]["content"]


def test_zero_hit_multi_domain_selection():
    router = QueryRouter(ROUTER_TERMS)
    llm = ScriptedLLM([_llm_domain_response(["game_theory", "philosophy"])])

    decision = route_with_fallback(
        router, "一个既玄又抽象的问题", ALLOWED, llm=llm, fallback_enabled=True,
    )

    assert decision.mode == MODE_MULTI
    assert decision.domains == ["game_theory", "philosophy"]


def test_out_of_menu_domains_dropped_and_recorded():
    router = QueryRouter(ROUTER_TERMS)
    llm = ScriptedLLM([_llm_domain_response(["philosophy", " cooking_recipe", "哲学人生"])])

    decision = route_with_fallback(
        router, "无法按术语分类的问题", ALLOWED, llm=llm, fallback_enabled=True,
    )

    # 域外名称被丢弃，mode 按剩下的域数推导；丢弃记录进 reason 供 debug
    assert decision.domains == ["philosophy"]
    assert decision.mode == MODE_SINGLE
    assert "cooking_recipe" in decision.reason and "哲学人生" in decision.reason


def test_llm_mode_field_is_ignored():
    """LLM 自报 mode=global 但给了域：以域为准，不信它的 mode 字段。"""
    router = QueryRouter(ROUTER_TERMS)
    llm = ScriptedLLM([_llm_domain_response(["game_theory"], mode=MODE_GLOBAL)])

    decision = route_with_fallback(
        router, "无法分类的问题", ALLOWED, llm=llm, fallback_enabled=True,
    )
    assert decision.mode == MODE_SINGLE


def test_llm_selects_only_invalid_domains_falls_back_to_global():
    router = QueryRouter(ROUTER_TERMS)
    llm = ScriptedLLM([_llm_domain_response(["新的领域"])])

    decision = route_with_fallback(
        router, "无法分类的问题", ALLOWED, llm=llm, fallback_enabled=True,
    )
    assert decision.mode == MODE_GLOBAL
    assert decision.domains == []
    assert decision.origin == ORIGIN_LLM_FALLBACK


def test_llm_selects_nothing_is_global():
    router = QueryRouter(ROUTER_TERMS)
    llm = ScriptedLLM([_llm_domain_response([])])

    decision = route_with_fallback(
        router, "无法分类的问题", ALLOWED, llm=llm, fallback_enabled=True,
    )
    assert decision.mode == MODE_GLOBAL
    assert decision.origin == ORIGIN_LLM_FALLBACK


def test_llm_output_without_json_falls_back_to_global():
    router = QueryRouter(ROUTER_TERMS)
    llm = ScriptedLLM(["抱歉，我不明白。"])

    decision = route_with_fallback(
        router, "无法分类的问题", ALLOWED, llm=llm, fallback_enabled=True,
    )
    assert decision.mode == MODE_GLOBAL
    assert decision.origin == ORIGIN_LLM_FALLBACK


def test_llm_exception_falls_back_to_global_with_error_origin():
    class ExplodingLLM(ScriptedLLM):
        def chat(self, messages):
            self.calls.append(messages)
            raise TimeoutError("超时")

    router = QueryRouter(ROUTER_TERMS)
    decision = route_with_fallback(
        router, "无法分类的问题", ALLOWED, llm=ExplodingLLM([]), fallback_enabled=True,
    )
    assert decision.mode == MODE_GLOBAL
    assert decision.origin == ORIGIN_LLM_FALLBACK_ERROR


def test_empty_allowed_domains_skips_llm():
    router = QueryRouter(ROUTER_TERMS)
    llm = ScriptedLLM([])

    decision = route_with_fallback(router, "任何问题", [], llm=llm, fallback_enabled=True)

    assert decision.mode == MODE_GLOBAL
    assert llm.calls == []
