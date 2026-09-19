"""测量「经本机代理建立 CONNECT 隧道」这一步单独花多久。

为什么单独测这一步：
    走系统代理时整个请求的耗时和直连差别很大，需要判断慢在「代理转发」
    还是「模型推理」。probe_connect_time.py 量直连端点的 TCP+TLS，
    本脚本量经代理的 CONNECT 隧道。两段建连如果都很快，说明剩余的时间
    全部发生在请求发出之后，与代理无关。

    注意：上游 LLM 延迟受服务负载、网络和时段影响较大，不同时间的单次
    测量给出过很不一样的结果，所以这里不引用任何历史数字；以本次输出为准，
    并在多个时段重复测量。

结论（定性，不是数字）：隧道能正常建立、系统代理是活的、健康的，
慢不等于死。直连反而更容易被掐断，所以本项目保持默认走代理，
不加 trust_env=False。

这里不发模型请求，只发一个 HTTP CONNECT 然后读首行响应，所以单次耗时
就是纯粹的代理侧开销。不发任何凭证，只打印 host、耗时、HTTP 首行。
"""
from __future__ import annotations

import socket
import sys
import time
from urllib.parse import urlparse

base_url = __import__("os").environ["ANTHROPIC_BASE_URL"]
host = urlparse(base_url).hostname
target_port = 443

PROXY_HOST, PROXY_PORT = "127.0.0.1", 9674  # 注册表里的 ProxyServer，见 proxy-network-setup
ATTEMPTS = 5
TUNNEL_TIMEOUT = 90.0


def probe_once(n: int) -> None:
    sock = None
    started = time.perf_counter()
    try:
        sock = socket.create_connection((PROXY_HOST, PROXY_PORT), timeout=TUNNEL_TIMEOUT)
        sock.settimeout(TUNNEL_TIMEOUT)
        request = f"CONNECT {host}:{target_port} HTTP/1.1\r\nHost: {host}:{target_port}\r\n\r\n"
        sock.sendall(request.encode("ascii"))
        # CONNECT 的响应只有首行有信息量，读到空行就说明隧道已经建立
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buffer += chunk
        elapsed = (time.perf_counter() - started) * 1000
        first_line = buffer.split(b"\r\n", 1)[0].decode("ascii", "replace")
        print(f"  第 {n} 次：隧道建立 {elapsed:>8.0f}ms  {first_line}")
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        print(f"  第 {n} 次：失败 {elapsed:>8.0f}ms  {type(exc).__name__}: {str(exc)[:100]}")
    finally:
        if sock is not None:
            sock.close()


print(f"目标 {host}:{target_port}   经代理 {PROXY_HOST}:{PROXY_PORT}")
for n in range(1, ATTEMPTS + 1):
    probe_once(n)

sys.exit(0)
