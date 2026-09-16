"""SQLite 存储层。

两张表：
    documents  一篇 Markdown 文档一行，content 存全文，content_hash 用于增量判断
    chunks     切块结果，外键指向 documents，文档删除时级联删除

为什么 SQLite 和 ChromaDB 要并存，不能只留一个？
    - SQLite 存的是"文档本身"：全文、标题、来源、时间戳，给人看、给接口查。
    - ChromaDB 存的是"向量索引"：同一份文本的另一种表示，只为算相似度。
    向量索引是可以删掉重建的（重新 ingest 一遍就有），文档原文不能丢。
    所以文档原文放 SQLite，ChromaDB 只是它的派生物。
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

from .chunker import Chunk
from .parser import ParsedDocument

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT    NOT NULL,
    file_path    TEXT    NOT NULL UNIQUE,          -- 相对 Vault 的路径，天然是主键候选
    folder       TEXT    NOT NULL DEFAULT '',
    tags         TEXT    NOT NULL DEFAULT '',      -- SQLite 不支持数组，用逗号分隔
    source       TEXT    NOT NULL DEFAULT '',
    content      TEXT    NOT NULL,
    content_hash TEXT    NOT NULL,                 -- 正文 sha256，判断内容有没有变
    created_at   TEXT    NOT NULL DEFAULT '',
    updated_at   TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS chunks (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id    INTEGER NOT NULL,
    chunk_index    INTEGER NOT NULL,
    content        TEXT    NOT NULL,
    heading        TEXT    NOT NULL DEFAULT '',
    start_position INTEGER NOT NULL DEFAULT 0,
    end_position   INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT    NOT NULL DEFAULT '',
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""

# 连接时的默认 PRAGMA，让 SQLite 行为可预测
_CONNECT_PRAGMAS = [
    "PRAGMA foreign_keys = ON",   # 必须显式打开，否则级联删除不生效
    "PRAGMA journal_mode = WAL",  # 读写可并发，ingest 时 API 还能查
]


def connect(db_path: Path) -> sqlite3.Connection:
    """打开数据库连接。父目录不存在会自动创建。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row  # 让查询结果能按列名取
    for pragma in _CONNECT_PRAGMAS:
        conn.execute(pragma)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """建表。重复调用安全（IF NOT EXISTS）。"""
    conn.executescript(SCHEMA)
    conn.commit()


def content_hash(content: str) -> str:
    """正文的 sha256，用来判断文件内容是否变化。"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def upsert_document(conn: sqlite3.Connection, doc: ParsedDocument) -> tuple[int, bool, bool]:
    """写入或更新文档，返回 (document_id, changed, created)。

    changed=False 表示内容哈希没变、库里已经有这份内容，调用方可以跳过切块和向量化——
    这就是增量 ingest 的核心：重复跑不会重复插库。
    created=True 表示这是第一次入库（新文档），否则是内容变化后的更新。
    """
    digest = content_hash(doc.content)
    row = conn.execute("SELECT id, content_hash FROM documents WHERE file_path = ?", (doc.file_path,)).fetchone()

    if row and row["content_hash"] == digest:
        return row["id"], False, False

    tags_csv = ",".join(doc.tags)
    if row:
        conn.execute(
            """
            UPDATE documents
               SET title = ?, folder = ?, tags = ?, source = ?, content = ?,
                   content_hash = ?, created_at = ?, updated_at = ?
             WHERE id = ?
            """,
            (doc.title, doc.folder, tags_csv, doc.source, doc.content,
             digest, doc.created_at, doc.updated_at, row["id"]),
        )
        conn.commit()
        return row["id"], True, False

    cur = conn.execute(
        """
        INSERT INTO documents
            (title, file_path, folder, tags, source, content, content_hash, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (doc.title, doc.file_path, doc.folder, tags_csv, doc.source, doc.content,
         digest, doc.created_at, doc.updated_at),
    )
    conn.commit()
    return cur.lastrowid, True, True


def replace_chunks(conn: sqlite3.Connection, document_id: int, chunks: list[Chunk]) -> int:
    """整体替换某个文档的切块结果，返回写入的块数。

    先删旧块再插新块，放在同一个事务里：
    切块策略变了或者文档改短了，旧的多余块不会残留。
    """
    now = _now()
    with conn:  # with 连接 = 自动 commit/rollback
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.executemany(
            """
            INSERT INTO chunks
                (document_id, chunk_index, content, heading, start_position, end_position, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (document_id, i, c.content, c.heading, c.start_position, c.end_position, now)
                for i, c in enumerate(chunks)
            ],
        )
    return len(chunks)


def get_document(conn: sqlite3.Connection, document_id: int) -> dict | None:
    """按 id 查单个文档，顺带查它有多少个 chunk。"""
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["tags"] = [t for t in result.pop("tags").split(",") if t]
    result["chunk_count"] = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,)
    ).fetchone()[0]
    return result


def list_documents(conn: sqlite3.Connection, limit: int = 50, offset: int = 0) -> list[dict]:
    """分页列出文档元信息（不含 content 全文，避免响应过大）。"""
    rows = conn.execute(
        """
        SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
          FROM documents d
         ORDER BY d.id
         LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item.pop("content", None)
        item.pop("content_hash", None)
        item["tags"] = [t for t in item.pop("tags").split(",") if t]
        results.append(item)
    return results


def get_chunks(conn: sqlite3.Connection, document_id: int) -> list[dict]:
    """取某个文档的全部切块，按 chunk_index 排序。"""
    rows = conn.execute(
        "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (document_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_chunks_for_reindex(conn: sqlite3.Connection) -> list[dict]:
    """一次性取出全库所有 chunk 及其文档元信息，供重建向量索引用。

    这是重建 Chroma 的唯一数据来源。tags 已经是 CSV 字符串，和 Chroma
    的标量 metadata 天然兼容，这里不再拆成列表。
    """
    rows = conn.execute(
        """
        SELECT c.id AS chunk_id, c.document_id, c.chunk_index, c.content, c.heading,
               d.title, d.file_path, d.source, d.folder, d.tags
          FROM chunks c
          JOIN documents d ON d.id = c.document_id
         ORDER BY d.id, c.chunk_index
        """
    ).fetchall()
    return [dict(r) for r in rows]


def count_documents(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


def count_chunks(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
