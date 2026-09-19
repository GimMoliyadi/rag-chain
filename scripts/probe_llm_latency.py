"""测量当前真实 LLM provider 的延迟分布。

上游 LLM 延迟受服务负载、网络和时段影响较大，之前不同时间的单次测量给出过
很不一样的结果，所以本脚本的用途是重新量一遍「现在」的分布，而不是去复现
某个历史数字。只测量，不优化，不修改 app/ 下任何代码。

要得出可靠的结论，请在多个时段各跑一次；不要拿单次结果去改 timeout。

三段各跑 RUNS 次：

    Part 0  纯网络：到端点的 TCP 建连 + TLS 握手
            用独立 socket 量，和 SDK 无关。量它是为了确认建连现在是不是
            瓶颈——结论以这次测量为准，不引用任何历史数字。

    Part A  最简请求 "只回答：测试成功"，走流式调用
            这是判断瓶颈位置的关键：对比它和 Part B 的首字节耗时。
            如果最简请求的首字节和 Part B 差不多，说明这段等待固定发生在
            服务端（排队 / 模型起头），和 prompt 体量无关；
            如果最简请求只要一两秒，说明首字节耗时和输入体量成正比，
            RAG 的总耗时属于"payload 太大"而不是"服务端慢"。

    Part B  真实 /api/chat，走完整的 检索 → 拼 Prompt → LLM 链路
            通过 create_app() 构造真实 app，TestClient 打真实端点。
            注意：这个端点不流式（本轮明确要求不加流式），所以它观测不到首字节，
            只有总耗时。首字节的问题由 Part A 回答。

TTFT 记两个口径，因为这两者可能差很多：
    首字节   = 收到服务端第一个 SSE 事件（message_start）
    首 token = 收到第一个 text_delta
    如果端点在出正文之前先吐 thinking，首 token 会明显晚于首字节。
    上一版探针在这里撞过 AttributeError：必须先吐 thinking_delta（字段 thinking），
    text_delta 才带 text，所以必须按 delta.type 分派。

token 数从 message_start.usage.input_tokens 和 message_delta.usage.output_tokens 取。
第三方兼容端点不一定返回 usage，取不到就标"未提供"，不猜。

只打印 host / 耗时 / token 数，不打印任何凭证，不打印 thinking 正文。

用法：python scripts/probe_llm_latency.py
"""
from __future__ import annotations

import os
import socket
import ssl
import sys
import time
from pathlib import Path
from statistics import median
from urllib.parse import urlparse

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic
from httpx2 import Timeout

from app.config import Settings
from app.main import create_app

RUNS = 5
SIMPLE_PROMPT = "只回答：测试成功"
SIMPLE_MAX_TOKENS = 64
RAG_QUESTION = "什么是囚徒困境？"


def percentile(ordered: list[float], p: float) -> float:
    """线性插值分位数（numpy 默认口径）。

    n=5 时最近秩法会直接返回最大值，P95 和最大变成一个数，看不出分布形状；
    线性插值能落在这两个数之间，更接近"如果有更多样本，P95 大概在哪儿"。
    """
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= len(ordered):
        return ordered[-1]
    return ordered[lo] + frac * (ordered[lo + 1] - ordered[lo])


def summarize(name: str, values: list[float]) -> None:
    if not values:
        print(f"  {name:<18} 无成功样本")
        return
    ordered = sorted(values)
    print(
        f"  {name:<18} n={len(ordered)}  最小={ordered[0]:>8.0f}  "
        f"中位={median(values):>8.0f}  P95={percentile(ordered, 0.95):>8.0f}  "
        f"最大={ordered[-1]:>8.0f}  ms"
    )


def banner(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def part_connect(host: str, port: int) -> None:
    """独立 socket 量 TCP 建连 + TLS 握手。每次都是全新连接，不复用。"""
    banner(f"Part 0  纯网络耗时  {host}:{port}  ×{RUNS}")

    tcp, tls = [], []
    for i in range(1, RUNS + 1):
        sock = None
        t1 = t2 = None
        try:
            started = time.perf_counter()
            sock = socket.create_connection((host, port), timeout=60)
            t1 = (time.perf_counter() - started) * 1000

            ctx = ssl.create_default_context()
            started = time.perf_counter()
            wrapped = ctx.wrap_socket(sock, server_hostname=host)
            t2 = (time.perf_counter() - started) * 1000
            wrapped.close()
        except Exception as exc:
            print(f"  第 {i} 次：失败  {type(exc).__name__}: {str(exc)[:100]}")
            continue
        finally:
            if sock is not None:
                sock.close()

        tcp.append(t1)
        tls.append(t2)
        print(f"  第 {i} 次：TCP 建连 {t1:>7.0f}ms   TLS 握手 {t2:>7.0f}ms")

    print()
    summarize("TCP 建连", tcp)
    summarize("TLS 握手", tls)
    summarize("两者合计", [a + b for a, b in zip(tcp, tls)])


def make_client(settings: Settings) -> anthropic.Anthropic:
    """按 app/llm.py 里同样的方式构造客户端：同样的超时配置，凭证从进程环境取。

    每次调用都新建一个客户端，不复用连接池。这样每次测量的首字节里都真实包含
    一次握手开销，五次之间可比较；如果端点保持长连接，这个做法会略微高估 TTFT。
    """
    return anthropic.Anthropic(
        timeout=Timeout(timeout=settings.llm_timeout_total, connect=settings.llm_timeout_connect)
    )


def measure_streamed(prompt: str, max_tokens: int, model: str, settings: Settings) -> dict:
    """一次流式调用，拆出首字节 / 首 token / 总耗时 / token 数。"""
    client = make_client(settings)
    started = time.perf_counter()
    first_byte = first_token = None
    thinking_chars = text_chars = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    answer = ""
    error: str | None = None

    try:
        with client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ) as stream:
            for event in stream:
                now = (time.perf_counter() - started) * 1000
                if first_byte is None:
                    first_byte = now

                if event.type == "message_start":
                    usage = getattr(getattr(event, "message", None), "usage", None)
                    if usage is not None:
                        input_tokens = getattr(usage, "input_tokens", None)
                        output_tokens = getattr(usage, "output_tokens", None)
                elif event.type == "content_block_delta":
                    delta_type = event.delta.type
                    if delta_type == "thinking_delta":
                        thinking_chars += len(event.delta.thinking)
                    elif delta_type == "text_delta":
                        if first_token is None:
                            first_token = now
                        chunk = event.delta.text or ""
                        text_chars += len(chunk)
                        answer += chunk
                elif event.type == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        got = getattr(usage, "output_tokens", None)
                        if got is not None:
                            output_tokens = got
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:120]}"
    finally:
        client.close()

    return {
        "total_ms": (time.perf_counter() - started) * 1000,
        "first_byte_ms": first_byte,
        "first_token_ms": first_token,
        "thinking_chars": thinking_chars,
        "text_chars": text_chars,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "answer": answer,
        "error": error,
    }


def part_simple(model: str, settings: Settings) -> None:
    banner(f"Part A  最简请求  {SIMPLE_PROMPT!r}  max_tokens={SIMPLE_MAX_TOKENS}  ×{RUNS}")

    first_byte: list[float] = []
    first_token: list[float] = []
    total: list[float] = []

    for i in range(1, RUNS + 1):
        r = measure_streamed(SIMPLE_PROMPT, SIMPLE_MAX_TOKENS, model, settings)
        if r["error"] is not None:
            print(f"  第 {i} 次：失败  耗时={r['total_ms']:>8.0f}ms  {r['error']}")
            continue

        total.append(r["total_ms"])
        if r["first_byte_ms"] is not None:
            first_byte.append(r["first_byte_ms"])
        if r["first_token_ms"] is not None:
            first_token.append(r["first_token_ms"])

        def fmt(value: float | None) -> str:
            return f"{value:>8.0f}" if value is not None else "     未提供"

        def tok(value: int | None) -> str:
            return f"{value}" if value is not None else "未提供"

        print(
            f"  第 {i} 次：首字节={fmt(r['first_byte_ms'])}  首token={fmt(r['first_token_ms'])}"
            f"  总耗时={r['total_ms']:>8.0f}  输入token={tok(r['input_tokens']):>6}  "
            f"输出token={tok(r['output_tokens']):>4}  thinking={r['thinking_chars']:>5}字"
        )
        print(f"           答案文本={r['answer'][:60]!r}（正文 {r['text_chars']} 字）")

    print()
    summarize("首字节 (TTFB)", first_byte)
    summarize("首 token (TTFT)", first_token)
    summarize("总耗时", total)


def part_rag(model: str, settings: Settings) -> None:
    """真实 /api/chat。create_app() 会加载 bge-m3，这段耗时不算在测量里。"""
    banner(f"Part B  真实 /api/chat  问题={RAG_QUESTION!r}  ×{RUNS}")

    loaded_at = time.perf_counter()
    app = create_app(settings)
    print(f"  app 构造完成（含加载 embedding 模型）：{time.perf_counter() - loaded_at:.1f}s")
    print("  注意：该端点不流式，因此观测不到首字节，只有总耗时。")
    print()

    from fastapi.testclient import TestClient

    client = TestClient(app)
    total: list[float] = []
    for i in range(1, RUNS + 1):
        started = time.perf_counter()
        try:
            resp = client.post("/api/chat", json={"question": RAG_QUESTION})
            elapsed = (time.perf_counter() - started) * 1000
            if resp.status_code != 200:
                print(f"  第 {i} 次：HTTP {resp.status_code}  耗时={elapsed:>8.0f}ms")
                continue
            body = resp.json()
            answer = body.get("answer", "")
            sources = body.get("sources", [])
            total.append(elapsed)
            print(
                f"  第 {i} 次：HTTP {resp.status_code}  总耗时={elapsed:>8.0f}ms  "
                f"答案={len(answer):>5}字  引用={len(sources)}条"
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            print(f"  第 {i} 次：失败  耗时={elapsed:>8.0f}ms  {type(exc).__name__}: {str(exc)[:120]}")

    print()
    summarize("总耗时", total)

    app.state.conn.close()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = Settings()
    model = (settings.llm_model or os.environ.get("ANTHROPIC_MODEL", "")).strip()
    if not model:
        print("错误：没有模型 ID。请在 .env 的 LLM_MODEL 或 ANTHROPIC_MODEL 里填一处。")
        return 1

    parsed = urlparse(os.environ["ANTHROPIC_BASE_URL"])
    host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)

    print(f"端点 host   : {host}:{port}")
    print(f"模型        : {model}")
    print(f"超时配置    : connect={settings.llm_timeout_connect}s  total={settings.llm_timeout_total}s")
    print(f"provider    : {settings.llm_provider}")
    print(f"每段次数    : {RUNS}")

    part_connect(host, port)
    part_simple(model, settings)
    part_rag(model, settings)

    banner("结论判读")
    print("  对比 Part A 和 Part B 的首字节：两者接近 => 等待固定发生在服务端，")
    print("  和 prompt 体量无关，本机无法优化，只能改端点。")
    print("  Part A 远小于 Part B => 首字节耗时与输入体量成正比，RAG 的总耗时")
    print("  属于 payload 过大，可以通过缩小 context 来改善。")
    print("  以上都是判读方法，不是预设结论；具体数字以本次输出为准，")
    print("  并且建议在多个时段重复测量后再决定要不要改 timeout。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
