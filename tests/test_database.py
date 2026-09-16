"""测试二、四：SQLite 能正确存文档；相同文档不会无限重复写入。"""
from __future__ import annotations

import pytest

from app.database import connect, content_hash, count_documents, get_document, init_db, upsert_document
from app.parser import parse_markdown


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.db")
    init_db(c)
    yield c
    c.close()


def _doc(fake_vault):
    return parse_markdown(fake_vault / "法律" / "合同违约.md", fake_vault)


def test_document_round_trips_through_sqlite(conn, fake_vault):
    doc = _doc(fake_vault)
    doc_id, changed, created = upsert_document(conn, doc)

    assert created is True and changed is True
    got = get_document(conn, doc_id)

    assert got["title"] == "合同违约的认定与救济"
    assert got["file_path"] == "法律/合同违约.md"
    assert got["tags"] == ["法律", "合同法"]  # 读回来是列表，不是逗号字符串
    assert got["source"] == "民法典笔记"
    assert got["content_hash"] == content_hash(doc.content)
    assert got["chunk_count"] == 0
    assert count_documents(conn) == 1


def test_identical_document_is_not_inserted_again(conn, fake_vault):
    """增量 ingest 的核心保证：内容没变就跳过，不重复插库。"""
    doc = _doc(fake_vault)
    first_id, _, _ = upsert_document(conn, doc)
    second_id, changed, created = upsert_document(conn, doc)

    assert changed is False
    assert created is False
    assert second_id == first_id          # 复用同一条记录
    assert count_documents(conn) == 1     # 而不是 2


def test_changed_document_is_updated_in_place(conn, fake_vault):
    """内容真变了才算更新，而且必须换掉 id，不是新增一行。"""
    doc = _doc(fake_vault)
    doc_id, changed, _ = upsert_document(conn, doc)
    assert changed is True

    doc.content += "\n\n新增的一段内容。"
    new_id, changed, created = upsert_document(conn, doc)

    assert changed is True and created is False
    assert new_id == doc_id
    assert count_documents(conn) == 1
    assert get_document(conn, doc_id)["content_hash"] != content_hash(_doc(fake_vault).content)


def test_unique_file_path_rejects_duplicates(conn, fake_vault):
    doc = _doc(fake_vault)
    upsert_document(conn, doc)
    # file_path 上建了 UNIQUE 约束，同一路径不可能出现两行
    assert conn.execute(
        "SELECT COUNT(*) FROM documents WHERE file_path = ?", (doc.file_path,)
    ).fetchone()[0] == 1
