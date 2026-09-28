"""测试六、七：/health 与 /api/search 正常，顺带验证 /api/chat 的引用编号。"""
from __future__ import annotations


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["documents"] == 2
    assert body["chunks"] > 0
    assert body["embedding_model"]
    assert body["llm_model"] == "mock-echo"


def test_api_search_ok(client):
    r = client.get("/api/search", params={"q": "最小可行产品", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "最小可行产品"
    assert body["count"] > 0
    assert body["count"] == len(body["results"])

    first = body["results"][0]
    # 调用方需要的字段一个都不能少
    for key in ("rank", "score", "title", "file_path", "heading", "content"):
        assert key in first, f"缺少字段 {key}"
    assert first["rank"] == 1
    assert first["file_path"].endswith(".md")
    assert first["content"].strip()


def test_api_search_requires_query(client):
    assert client.get("/api/search").status_code == 422


def test_api_documents_list_and_detail(client):
    r = client.get("/api/documents", params={"limit": 10})
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert len(docs) == 2
    assert "content" not in docs[0]  # 列表接口不该返回全文

    detail = client.get(f"/api/documents/{docs[0]['id']}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["title"]
    assert body["chunk_count"] > 0
    assert "content_hash" not in body  # 内部字段不泄漏给调用方

    assert client.get("/api/documents/999999").status_code == 404


def test_api_chat_returns_answer_and_citations(client):
    r = client.post("/api/chat", json={"question": "合同违约了该怎么处理？"})
    assert r.status_code == 200
    body = r.json()

    assert body["answer"].strip()
    assert body["used_chunks"] > 0

    sources = body["sources"]
    assert len(sources) == body["used_chunks"]
    # 引用编号从 1 开始且连续——编号和 sources 数组必须一一对应，不允许伪造
    assert [s["index"] for s in sources] == list(range(1, len(sources) + 1))
    for s in sources:
        assert s["title"] and s["file_path"].endswith(".md")
        assert s["chunk"].strip()
        assert 0.0 <= s["score"] <= 1.0


def test_api_chat_rejects_empty_question(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_api_chat_rejects_unknown_mode(client):
    assert client.post("/api/chat", json={"question": "问题", "mode": "advisior"}).status_code == 422


def test_api_chat_advisor_reports_mode_and_missing_personal_context(client):
    """PERSONAL_CONTEXT_ENABLED 默认 false：advisor 照常回答，
    但 personal_context 明确报告未启用，而不是静默假装没有这一栏。"""
    r = client.post("/api/chat", json={"question": "合同违约了该怎么处理？", "mode": "advisor"})

    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "advisor"
    assert body["personal_context"] == {"loaded": False, "reason": "Personal Context 未启用"}
    assert body["answer"].strip()


def test_api_chat_normal_keeps_personal_context_unset(client):
    r = client.post("/api/chat", json={"question": "合同违约了该怎么处理？", "mode": "normal"})

    body = r.json()
    assert body["mode"] == "normal"
    assert body["personal_context"] is None


def test_api_chat_advisor_loads_personal_context(settings, ingested, tmp_path):
    """从文件读到 Personal Context 后真正进 Prompt：端到端验证 app 的装配链路。"""
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    context_path = tmp_path / "context.yaml"
    context_path.write_text(
        "profile:\n  current_direction: AI 应用开发\n"
        "projects:\n  - name: 观察项目\n    status: 进行中\n",
        encoding="utf-8",
    )
    live = Settings(
        vault_path=settings.vault_path,
        db_path=settings.db_path,
        chroma_path=settings.chroma_path,
        embedding_provider="local",
        llm_provider="mock",
        top_k=5,
        vault_name="测试知识库",
        personal_context_enabled=True,
        personal_context_path=str(context_path),
    )
    app = create_app(settings=live, embedding=ingested[1], store=ingested[0])

    with TestClient(app) as c:
        r = c.post("/api/chat", json={"question": "我目前主要方向是什么？", "mode": "advisor"})

    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "advisor"
    info = body["personal_context"]
    assert info["loaded"] is True
    assert info["available_sections"] == ["profile", "projects"]
    assert info["project_count"] == 1
    assert info["warnings"] == []


def test_api_ingest_reports_stats(client, settings):
    r = client.post("/api/ingest", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["unchanged"] == 2  # fixture 已经导过一次，再次触发应全部跳过
    assert body["failed"] == 0


def test_api_ingest_fails_cleanly_when_vault_missing(tmp_path):
    """Vault 路径配错时，ingest 接口应返回 400 而不是抛异常让服务挂掉。"""
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    # embedding_provider 固定为 local：测试不能因为 .env 里的 local_semantic 而去下载模型
    # db_path / chroma_path 也指到临时目录，测试不该打开真实的 data/
    bad = create_app(
        settings=Settings(
            vault_path=str(tmp_path / "不存在"),
            top_k=5,
            embedding_provider="local",
            db_path=str(tmp_path / "test.db"),
            chroma_path=str(tmp_path / "chroma"),
        )
    )
    with TestClient(bad) as c:
        r = c.post("/api/ingest", json={})
    assert r.status_code == 400
