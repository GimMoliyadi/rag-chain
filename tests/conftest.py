"""测试公共设施。

所有测试都在 tmp_path 里造一个极小的假 Vault，用完即弃——
测试绝对不能碰用户真实的 Obsidian Vault 和 data/ 目录。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embedding import get_embedding_client
from app.ingest import run_ingest
from app.main import create_app
from app.vectorstore import ChromaStore

# 一篇带 YAML frontmatter、多级标题的合同笔记
CONTRACT_MD = """---
title: 合同违约的认定与救济
tags:
  - 法律
  - 合同法
source: 民法典笔记
created: 2026-01-10
---

# 合同违约的认定与救济

合同违约是指合同当事人不履行或不适当履行合同约定的义务。

## 违约的类型

违约可以分为预期违约与届期违约两种。预期违约在履行期限届满前发生，
届期违约则在履行期限届满后发生。

## 违约责任的承担方式

违约责任主要包括继续履行、采取补救措施和赔偿损失。
当事人可以选择要求对方继续履行合同，也可以要求赔偿因此造成的损失。

## 免责事由

因不可抗力导致不能履行合同的，可以根据不可抗力的影响部分或全部免除责任。
"""

# 一篇带内联 #标签、flow 列表 tags 的创业笔记
MVP_MD = """---
title: 最小可行产品
tags: [创业, MVP]
source: 精益创业2.0
---

# 最小可行产品

最小可行产品（MVP）是用最少的成本验证一个核心假设的产品版本。

MVP 的目的不是做出完整产品，而是尽快获得真实的用户反馈。
创业者应该用最简单的方式验证需求，而不是一开始就投入大量资源。
"""


@pytest.fixture
def fake_vault(tmp_path: Path) -> Path:
    """造一个假 Vault：两篇真文档 + 一个应被忽略的目录 + 一个附件。"""
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    (vault / ".obsidian" / "app.json").write_text('{"a":1}', encoding="utf-8")
    (vault / "法律").mkdir()
    (vault / "创业").mkdir()
    (vault / "法律" / "合同违约.md").write_text(CONTRACT_MD, encoding="utf-8")
    (vault / "创业" / "MVP.md").write_text(MVP_MD, encoding="utf-8")
    (vault / "图片.png").write_bytes(b"\x89PNG fake")
    return vault


@pytest.fixture
def settings(fake_vault: Path, tmp_path: Path) -> Settings:
    """把数据库和向量库都重定向到临时目录，provider 固定用离线的 local / mock。"""
    return Settings(
        vault_path=str(fake_vault),
        db_path=str(tmp_path / "test.db"),
        chroma_path=str(tmp_path / "chroma"),
        embedding_provider="local",
        llm_provider="mock",
        chunk_size=800,
        chunk_overlap=120,
        top_k=5,
        vault_name="测试知识库",
    )


@pytest.fixture
def ingested(settings: Settings):
    """跑一遍完整入库，返回 (store, embedding, stats)，供检索和 API 测试复用。"""
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    stats = run_ingest(settings, embedding, store)
    return store, embedding, stats


@pytest.fixture
def client(settings: Settings, ingested):
    """已入库数据的 FastAPI 测试客户端，不需要真的起 uvicorn。"""
    with TestClient(create_app(settings=settings, embedding=ingested[1], store=ingested[0])) as c:
        yield c
