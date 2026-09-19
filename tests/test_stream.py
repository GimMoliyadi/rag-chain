"""流式接口 /api/chat/stream 的事件协议测试。

调用方靠事件顺序区分「正在检索」和「正在生成」，所以这里逐条断言：
事件名合法、顺序正确、token 拼起来等于 done 里的 answer、
sources 字段和非流式接口一致、失败时吐 error 而不是让响应直接中断。

非流式 /api/chat 的行为由 test_api.py 覆盖，这里只顺带确认两个端点的
sources 序列化结果完全一致。
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import create_app
from app.vectorstore import ChromaStore

# 后端允许吐出的全部事件名；出现这个集合之外的名字说明前后端协议漂移了
ALLOWED = {
    "retrieval_started",
    "retrieval_completed",
    "sources",
    "generation_started",
    "token",
    "done",
    "error",
}

SOURCES_KEYS = ("index", "title", "file_path", "heading", "score", "chunk")

QUESTION = "合同违约了该怎么处理？"


def _events(body: str) -> list[dict]:
    """把 SSE 文本拆回事件列表。空行是帧分隔符，漏了它调用方会把多帧粘成一条。"""
    out = []
    for frame in body.split("\n\n"):
        for line in frame.splitlines():
            line = line.strip()
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: "):]))
    return out


def _stream(client: TestClient, question: str = QUESTION) -> list[dict]:
    return _events(client.post("/api/chat/stream", json={"question": question}).text)


def _at(events: list[dict], name: str) -> dict:
    return next(e for e in events if e["event"] == name)


def test_stream_response_headers(client):
    r = client.post("/api/chat/stream", json={"question": QUESTION})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    # 关掉代理缓冲，否则阶段事件会被攒成一包，调用方看不到进度切换
    assert r.headers.get("x-accel-buffering") == "no"
    assert r.headers.get("cache-control") == "no-cache"
    # 每一帧都以空行结尾，这是 SSE 的帧边界（不能先 strip，那会把尾行空行吃掉）
    assert r.text.endswith("\n\n")


def test_stream_event_names_and_order(client):
    events = _stream(client)
    names = [e["event"] for e in events]

    assert names[0] == "retrieval_started"
    assert names[-1] == "done"
    assert set(names) <= ALLOWED
    assert "error" not in names

    # 阶段必须按 检索开始 → 检索完成 → 来源 → 开始生成 → 完成 推进
    pos = {n: names.index(n) for n in names}
    assert (
        pos["retrieval_started"]
        < pos["retrieval_completed"]
        < pos["sources"]
        < pos["generation_started"]
        < pos["token"]
        < pos["done"]
    )


def test_stream_tokens_reassemble_done_answer(client):
    events = _stream(client)
    tokens = [e["data"]["text"] for e in events if e["event"] == "token"]
    done = _at(events, "done")["data"]

    assert tokens, "至少应该收到一个 token"
    assert "".join(tokens) == done["answer"], "token 拼起来必须等于 done 里的 answer"
    assert done["answer"].strip()
    assert isinstance(done["elapsed_ms"], int)
    assert done["used_chunks"] > 0


def test_stream_retrieval_started_carries_question(client):
    first = _at(_stream(client), "retrieval_started")["data"]
    assert first["question"] == QUESTION
    assert isinstance(first["top_k"], int) and first["top_k"] > 0

    completed = _at(_stream(client), "retrieval_completed")["data"]
    assert isinstance(completed["count"], int)
    assert isinstance(completed["elapsed_ms"], int)


def test_stream_sources_match_blocking_chat(client):
    """两个端点的 sources 结构必须逐字段一致，调用方只有一套解析代码。"""
    blocking = client.post("/api/chat", json={"question": QUESTION}).json()
    sources = _at(_stream(client), "sources")["data"]

    assert sources["sources"], "测试库里应该能检索到结果"
    assert [s["index"] for s in sources["sources"]] == list(
        range(1, len(sources["sources"]) + 1)
    )
    for s in sources["sources"]:
        for key in SOURCES_KEYS:
            assert key in s, f"sources 缺少字段 {key}"
    assert sources["sources"] == blocking["sources"]
    assert sources["used_chunks"] == blocking["used_chunks"]


def test_stream_empty_knowledge_base(settings):
    """空库不调用 LLM：检索完成 → 空来源 → done，不出现 generation_started。"""
    empty = create_app(
        settings=settings,
        store=ChromaStore(settings.chroma_path_resolved / "empty"),
    )
    with TestClient(empty) as c:
        events = _stream(c)

    names = [e["event"] for e in events]
    assert names == ["retrieval_started", "retrieval_completed", "sources", "done"]
    assert "token" not in names

    done = _at(events, "done")["data"]
    assert _at(events, "sources")["data"]["sources"] == []
    assert done["used_chunks"] == 0
    assert "知识库中目前没有足够的信息" in done["answer"]


class _ExplodingLLM:
    """吐几个 token 后抛错：验证部分回答保留、错误走事件而不是掐断响应。"""

    model_name = "explode"

    def chat(self, messages):
        raise AssertionError("流式链路不该走非流式 chat()")

    def chat_stream(self, messages):
        yield "前半句"
        raise RuntimeError("上游中断了")


class _ExplodingRetriever:
    """检索阶段就失败：验证 error 事件能出现在 generation_started 之前。"""

    def search(self, question, top_k=None):
        raise ValueError("检索库不可用")


def test_stream_error_after_partial_answer(settings, ingested):
    # 复用 fixture 里的 embedding 和 store，不为造一个坏 LLM 再加载一遍 bge-m3
    broken = create_app(
        settings=settings,
        embedding=ingested[1],
        store=ingested[0],
        llm=_ExplodingLLM(),
    )
    with TestClient(broken) as c:
        events = _stream(c)

    names = [e["event"] for e in events]
    assert "error" in names
    assert "done" not in names, "中途失败不应伪装成正常完成"
    assert _at(events, "error")["data"]["message"] == "上游中断了"
    assert "".join(e["data"]["text"] for e in events if e["event"] == "token") == "前半句"


def test_stream_error_during_retrieval(client):
    client.app.state.rag._retriever = _ExplodingRetriever()
    events = _stream(client)

    names = [e["event"] for e in events]
    assert names == ["retrieval_started", "error"]
    assert "done" not in names
    assert "检索库不可用" in _at(events, "error")["data"]["message"]


def test_stream_rejects_empty_question(client):
    assert client.post("/api/chat/stream", json={"question": ""}).status_code == 422
    assert client.post("/api/chat/stream", json={"top_k": 99}).status_code == 422
