"""最小 LLM API smoke test：确认 ANTHROPIC_* 环境变量指向的端点真能返回文本。

    python scripts/smoke_llm_api.py

它不属于默认 pytest，也不 import 本项目任何模块。目的是把「端点协议对不对」
这个前提单独验证掉，而不是让它在 /api/chat 请求时才暴露。

读取的环境变量与 Claude Code 自己用的完全一致：
    ANTHROPIC_BASE_URL    端点，SDK 会往 {base_url}/v1/messages 发请求
    ANTHROPIC_AUTH_TOKEN  凭证，SDK 以 Authorization: Bearer 形式发送
    ANTHROPIC_MODEL       模型 ID

为什么用 SDK 的零参构造而不是自己拼请求：base_url 和凭证的解析规则
（API_KEY 优先于 AUTH_TOKEN、Bearer 与 x-api-key 的差别）是 SDK 的责任，
自己重写一份就埋了第二个可能的错源。

token 只会检查「是否设置」和长度，值本身不打印、不写文件。
"""
from __future__ import annotations

import os
import sys

import anthropic

# 单条短回答的预算。故意给得小：本脚本只验证「能返回文本」，不需要长输出，
# 小一点也让失败路径更快暴露。
MAX_TOKENS = 256
PROMPT = "请用一句话回答：水的化学式是什么？"


def _say(msg: str) -> None:
    """统一输出并 flush，避免后台跑时看不到进度。"""
    print(msg, flush=True)


def _main() -> int:
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    model = os.environ.get("ANTHROPIC_MODEL", "").strip()

    _say("=== LLM API smoke test ===")
    _say(f"ANTHROPIC_BASE_URL = {base_url or '<未设置>'}")
    _say(
        f"ANTHROPIC_AUTH_TOKEN = "
        + (f"已设置，长度 {len(auth_token)}（值不打印）" if auth_token else "<未设置>")
    )
    _say(f"ANTHROPIC_MODEL    = {model or '<未设置>'}")
    _say(f"anthropic SDK      = {anthropic.__version__}")

    missing = [name for name, val in [("ANTHROPIC_BASE_URL", base_url),
                                      ("ANTHROPIC_AUTH_TOKEN", auth_token),
                                      ("ANTHROPIC_MODEL", model)] if not val]
    if missing:
        _say(f"\n缺少环境变量：{', '.join(missing)}")
        _say("本脚本要求 Claude Code 的 ANTHROPIC_* 环境变量可见。")
        return 2

    # 零参构造：base_url 和凭证都从环境变量取。timeout 和 max_retries 压到
    # 低一点，好让「协议不对 / 认证不对」这类硬失败快速暴露，而不是重试三轮。
    client = anthropic.Anthropic(timeout=30.0, max_retries=1)

    _say(f"\n请求 POST {base_url}/v1/messages")
    _say(f"model={model}  max_tokens={MAX_TOKENS}")

    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": PROMPT}],
        )
    except anthropic.APIConnectionError as exc:
        _say(f"[FAIL] 连接失败：{exc.__class__.__name__}")
        _say("DNS / 代理 / 端口不通，或端点根本没开。")
        return 1
    except anthropic.AuthenticationError as exc:
        _say(f"[FAIL] 认证失败：{exc.__class__.__name__} - {exc.message}")
        return 1
    except anthropic.APIStatusError as exc:
        _say(f"[FAIL] HTTP {exc.status_code}：{exc.__class__.__name__} - {exc.message}")
        _say("404 通常意味着这个端点不认 /v1/messages 路径；400 常是 model id 不被接受。")
        return 1
    except Exception as exc:
        _say(f"[FAIL] 未预期异常 {exc.__class__.__name__}: {exc}")
        return 1

    # 响应内容是一个 block 列表，text / thinking / tool_use 等混在一起，
    # 必须按 type 挑，不能直接假设第一个就是文本。
    texts = [block.text for block in response.content if block.type == "text"]
    answer = "\n".join(texts).strip()

    _say(f"\nstop_reason = {response.stop_reason}")
    _say(f"model       = {response.model}")
    _say(
        "usage       = input "
        f"{response.usage.input_tokens} / output {response.usage.output_tokens}"
    )
    _say(f"\n--- 模型返回文本（{len(answer)} 字符）---")
    _say(answer)

    if not answer:
        _say("\n[FAIL] 端点通了但没有返回 text 块")
        return 1

    _say("\n[OK] 端点可用，模型能返回文本")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
