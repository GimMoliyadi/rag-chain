"""对比「等第一字节」和「生成完毕」的耗时，判断慢在哪里。

前提：上游 LLM 延迟受服务负载、网络和时段影响较大，历史测量数字不稳定，
所以这里不引用任何历史耗时。probe_connect_time.py / probe_proxy_connect.py
已经确认两段建连本身很快，剩余时间都发生在请求发出之后。流式调用能把
这段时间拆成三段：

    首字节耗时  = 排队 + 模型起头（这段没法优化，只能等）
    thinking 段 = 模型在内部推理，不产出可用内容（可以关掉来省时间）
    text 段     = 逐字吐出答案（流式能让用户先看到内容，但总时长不变）

上一版探针在这里撞过 AttributeError：这个端点先吐 thinking_delta（字段是
thinking，不是 text），所以必须按 delta.type 分派，不能假设第一个 delta 就是文本。

这里故意构造和真实 RAG 请求同量级的 prompt，否则测出来的是 ping 级请求的延迟，
没有参考价值。对比走代理和直连两条路：如果两条路的首字节接近，瓶颈就在模型侧。

只打印 host / 耗时 / 各段字符数，不打印任何凭证，也不打印 thinking 的正文。
"""
from __future__ import annotations

import os
import sys
import time

import anthropic

# 一段足够长、但完全无意义的内容，用来把 prompt 撑到 RAG 的真实体量
# （5 个 chunk × 约 800 字 ≈ 4000 字）。取重复中文是为了让 token 数接近真实 chunk。
FILLER_UNIT = "本文讨论了食品安全标签合规相关的通用要点，包括配料表顺序、添加剂使用范围和营养声称的判定边界。"
FILLER = FILLER_UNIT * 80
PROMPT = f"以下是知识库检索到的资料，请基于资料回答问题。\n\n{FILLER}\n\n问题：请用三句话总结上述资料的主要结论。"

MAX_TOKENS = 1024
MODEL = os.environ.get("ANTHROPIC_MODEL", "sensenova-6.8-flash-lite")


def measure(label: str, trust_env: bool | None) -> None:
    http_client = anthropic.DefaultHttpxClient(trust_env=False) if trust_env is False else None
    client = anthropic.Anthropic(http_client=http_client, timeout=300.0)

    started = time.perf_counter()
    first_byte: float | None = None
    thinking_chars = 0
    text_chars = 0
    try:
        with client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": PROMPT}],
            stream=True,
        ) as stream:
            for event in stream:
                # message_start 是连接建立后的第一个事件，用它标记"服务端开始回话"
                if event.type == "message_start":
                    first_byte = time.perf_counter() - started
                if event.type != "content_block_delta":
                    continue
                # thinking_delta 的字段是 thinking，text_delta 的字段才是 text
                if event.delta.type == "thinking_delta":
                    thinking_chars += len(event.delta.thinking)
                elif event.delta.type == "text_delta":
                    text_chars += len(event.delta.text)
        total = time.perf_counter() - started
        head = first_byte * 1000 if first_byte is not None else -1
        print(
            f"  [{label}] 成功  首字节={head:>7.0f}ms  总耗时={total*1000:>7.0f}ms"
            f"  thinking={thinking_chars:>5} 字  答案={text_chars:>4} 字"
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"  [{label}] 失败  耗时={elapsed*1000:>7.0f}ms  {type(exc).__name__}: {str(exc)[:120]}")
    finally:
        client.close()


print(f"prompt 长度={len(PROMPT)} 字符   max_tokens={MAX_TOKENS}")
print("\n走系统代理")
measure("proxy", True)
print("\n直连（trust_env=False）")
measure("direct", False)

sys.exit(0)
