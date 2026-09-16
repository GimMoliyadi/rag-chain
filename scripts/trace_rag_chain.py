"""把一次 /api/chat 的完整链路逐层打印出来。

    python scripts/trace_rag_chain.py

运行前提：uvicorn 已经在 127.0.0.1:8765 跑着（app.main:create_app --factory）。

分两段，各自解决一个问题：

    阶段一  用 httpx 打一次真实 HTTP 请求到 /api/chat。
            这一步验证的是「跑在服务进程里的链路」真的通了——
            不是我们在这个脚本里手搓的。

    阶段二  本地组装一模一样的 RAGEngine（同一个 retriever、同一个 rag.py），
            但把 LLM 换成 RecordingLLM 包一层。目的是拿到
            rag.py 实际拼出来、真的发给模型的 messages 原文。
            /api/chat 的返回里没有这个字段（只有拆开的 sources），
            想看「发给 LLM 的 context」只能这样拿到。

阶段二会重新加载一次 embedding 模型（本机实测约 36 秒），代价只为这一下。
凭证仍然只从 ANTHROPIC_* 环境变量读，脚本不打印 token、不写 .env。
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

# 输出统一 UTF-8。chunk 内容来自 Obsidian 笔记，可能含 cp936 覆盖不到的字符，
# 重定向到日志文件时按控制台编码写会直接 UnicodeEncodeError。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.embedding import get_embedding_client
from app.llm import AnthropicCompatibleLLM, BaseLLM
from app.rag import RAGEngine
from app.retriever import Retriever
from app.vectorstore import ChromaStore

API_URL = "http://127.0.0.1:8765/api/chat"
QUESTION = "为什么纳什均衡一定存在？存在性证明的关键工具是什么？"
TOP_K = 5

LINE = "=" * 72


class RecordingLLM(BaseLLM):
    """包一层真实 LLM，只多记一件事：收到的 messages 原文。

    不改 chat() 的返回值，所以 RAGEngine 走的是和生产完全相同的路径。
    """

    def __init__(self, inner: BaseLLM) -> None:
        self._inner = inner
        self.last_messages: list[dict] = []

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def chat(self, messages: list[dict]) -> str:
        self.last_messages = messages
        return self._inner.chat(messages)


def _hr(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}", flush=True)


def _phase1_http() -> dict:
    """打一次真实 HTTP 请求，证明服务进程里的链路通了。"""
    _hr("阶段一：真实 HTTP 请求 POST /api/chat")
    print(f"Query  = {QUESTION}")
    print(f"top_k  = {TOP_K}")

    # trust_env=False：打本机服务不要走系统代理。本机开着 sing-box(127.0.0.1:7897)，
    # httpx 默认会从环境变量解析代理，把 8765 的请求转给代理后拿回一个 502 空响应
    # （实测见 scripts 外的对照：PowerShell 直连同一地址返回 200）。
    resp = httpx.post(
        API_URL,
        json={"question": QUESTION, "top_k": TOP_K},
        timeout=180,
        trust_env=False,
    )
    print(f"HTTP   = {resp.status_code}")
    if resp.status_code != 200:
        print(f"响应体：{resp.text[:2000]}")
        raise SystemExit("阶段一失败，停止。")

    data = resp.json()
    _hr(f"线上 LLM answer（{len(data['answer'])} 字符）")
    print(data["answer"])
    print(f"\nused_chunks = {data['used_chunks']}  sources = {len(data['sources'])} 条")
    return data


def _phase2_trace() -> None:
    """本地重放同一个问题，取出实际发给 LLM 的 messages。"""
    _hr("阶段二：重放链路，取 rag.py 实际拼出的 messages")

    settings = Settings()
    settings.validate_providers()

    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    retriever = Retriever(embedding=embedding, store=store, default_top_k=settings.top_k)

    real_llm = AnthropicCompatibleLLM(
        model=settings.llm_model or None,
        max_tokens=settings.llm_max_tokens,
    )
    llm = RecordingLLM(real_llm)
    engine = RAGEngine(
        retriever=retriever,
        llm=llm,
        top_k=settings.top_k,
        vault_name=settings.vault_name,
    )

    _hr("1. Query")
    print(QUESTION)

    result = engine.answer(QUESTION, top_k=TOP_K)

    _hr(f"2. 检索到的 Top-K（{len(result.sources)} 条，来自 Chroma cosine）")
    for s in result.sources:
        print(f"\n[{s.index}] score={s.score}")
        print(f"    title   : {s.title}")
        print(f"    file    : {s.file_path}")
        print(f"    heading : {s.heading}")

    _hr(f"3. 发给 LLM 的 messages（共 {len(llm.last_messages)} 条）")
    for i, msg in enumerate(llm.last_messages, 1):
        print(f"\n--- message[{i}] role={msg['role']}  content={len(msg['content'])} 字符 ---")
        print(msg["content"])

    _hr(f"4. LLM answer（{len(result.answer)} 字符，model={real_llm.model_name}）")
    print(result.answer)

    _hr(f"5. sources（{len(result.sources)} 条，answer 里的 [N] 对应 index=N）")
    for s in result.sources:
        print(f"\n[{s.index}] score={s.score}  chars={len(s.content)}")
        print(f"    {s.file_path} / {s.heading}")
        print(f"    内容前 240 字符：{s.content[:240]}")

    print(f"\n{LINE}")
    print(f"链路走通：Query -> Chroma Top-K -> rag.py context -> {real_llm.model_name} -> answer + sources")


def main() -> None:
    _phase1_http()
    _phase2_trace()


if __name__ == "__main__":
    main()
