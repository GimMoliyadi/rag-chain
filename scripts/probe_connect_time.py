"""测量到 LLM 端点的 TCP 建连 + TLS 握手耗时分布。

目的只有一个：把 timeout 从"猜一个数"变成"实测分位数"。
只打印耗时和 host，不打印任何凭证。
"""
from __future__ import annotations

import socket
import ssl
import sys
import time
from urllib.parse import urlparse

base_url = __import__("os").environ["ANTHROPIC_BASE_URL"]
parsed = urlparse(base_url)
host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)

ATTEMPTS = 6

connects: list[float] = []
handshakes: list[float] = []

for i in range(1, ATTEMPTS + 1):
    tcp = tls = None
    sock = None
    try:
        started = time.perf_counter()
        sock = socket.create_connection((host, port), timeout=60)
        tcp = (time.perf_counter() - started) * 1000

        ctx = ssl.create_default_context()
        started = time.perf_counter()
        wrapped = ctx.wrap_socket(sock, server_hostname=host)
        tls = (time.perf_counter() - started) * 1000
        wrapped.close()
    except Exception as exc:
        print(f"  第 {i} 次：失败  {type(exc).__name__}")
        continue
    finally:
        if sock is not None:
            sock.close()

    connects.append(tcp)
    handshakes.append(tls)
    print(f"  第 {i} 次：TCP 建连 {tcp:>7.0f}ms   TLS 握手 {tls:>7.0f}ms")

def stats(name: str, values: list[float]) -> None:
    if not values:
        print(f"  {name}: 无成功样本")
        return
    ordered = sorted(values)
    print(f"  {name}: n={len(ordered)}  最小={ordered[0]:.0f}ms  中位={ordered[len(ordered)//2]:.0f}ms  最大={ordered[-1]:.0f}ms")


print(f"\n目标 {host}:{port}")
stats("TCP 建连耗时", connects)
stats("TLS 握手耗时", handshakes)
stats("两者合计", [c + h for c, h in zip(connects, handshakes)])

sys.exit(0)
