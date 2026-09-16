"""ChromaDB 向量存储层。

职责单一：只管向量的写入和删除，不做相似度计算（那是 retriever 的事）。

一个必须知道的限制：ChromaDB 的 metadata 只支持标量
（str / int / float / bool），不支持 list。
所以 tags 这类列表字段必须压成逗号分隔的字符串存进去。

索引指纹：
    建库时用的是哪个 embedding 模型，单独记在同级的 embedding_index.json 里。
    不写进 Chroma 的 collection metadata，是因为 get_or_create_collection
    对已存在的 collection 传不同 metadata 会直接报错——迁移过来的旧索引会被
    这个错误挡在门外。sidecar JSON 读写独立，不会和 Chroma 的持久化逻辑冲突。
"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb

COLLECTION_NAME = "obsidian_chunks"

# 指纹文件名：记录"建这个向量库时用的是哪个 embedding 模型"
INDEX_INFO_FILENAME = "embedding_index.json"


class ChromaStore:
    """本地持久化的 ChromaDB 封装。"""

    def __init__(self, persist_dir: Path, collection_name: str = COLLECTION_NAME):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        # hnsw:space = cosine 表示用余弦距离算相似度，
        # 这样返回的 distance 直接就是 1 - cos_sim 的形态，转成相似度很方便
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # 指纹放 chroma 目录的同级（data/embedding_index.json）
        self._info_path = persist_dir.parent / INDEX_INFO_FILENAME

    def read_index_info(self) -> dict | None:
        """读索引指纹。没有这个文件说明是旧版代码建的库（或刚清空过）。"""
        if not self._info_path.exists():
            return None
        return json.loads(self._info_path.read_text(encoding="utf-8"))

    def write_index_info(self, info: dict) -> None:
        """在向量全部写完之后再调，这样半成品的索引不会谎称自己是有效的。"""
        self._info_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def clear(self) -> None:
        """删掉整个 collection 并重建一个空的。

        换 embedding 模型时必须走这条路：旧向量和新向量不在同一个向量空间里，
        不能增量更新，只能整体重建。SQLite 里的 chunk 原文完全不受影响。
        """
        self._client.delete_collection(self._col.name)
        self._col = self._client.get_or_create_collection(
            name=self._col.name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        """释放 Chroma 客户端持有的 SQLite 连接。

        Windows 上文件被打开时无法删除或重命名。两阶段重建要在替换目录之前
        把目录里所有 handle 关掉，否则 rename 直接失败。PersistentClient
        的 close() 递减底层 System 的引用计数，最后一个引用关掉时释放数据库
        连接——chromadb 文档里明确说这对 PersistentClient 是必须的。
        """
        self._client.close()

    def count(self) -> int:
        return self._col.count()

    def upsert_chunks(self, rows: list[dict]) -> None:
        """写入一批 chunk 的向量。

        每行需要包含：
            chunk_id   int   SQLite chunks 表的 id（这里当 Chroma 的主键用）
            embedding  list[float]
            以及任意可标量化的元数据字段（document_id / title / file_path ...）
        """
        if not rows:
            return
        ids = [str(r["chunk_id"]) for r in rows]
        self._col.upsert(
            ids=ids,
            embeddings=[r["embedding"] for r in rows],
            documents=[r["content"] for r in rows],
            metadatas=[_metadata(r) for r in rows],
        )

    def query(self, embedding: list[float], n_results: int = 5) -> list[dict]:
        """按向量找最近的 n_results 条，返回 [{content, distance, **metadata}]。

        Chroma 原生的返回是 result["ids"][0] / result["metadatas"][0] 这种嵌套结构，
        在这里摊平成列表，上层就不用知道 Chroma 的返回格式了。
        """
        result = self._col.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            {"content": docs[i], "distance": distances[i], **(metas[i] or {})}
            for i in range(len(ids))
        ]

    def delete_by_document(self, document_id: int) -> None:
        """删掉某个文档的全部向量。

        在 ingest 里用于"内容变了就先把旧向量清掉再写新的"，
        避免文档改短之后残留已经失效的旧 chunk。
        """
        if self._col.count() == 0:
            return
        self._col.delete(where={"document_id": document_id})


def _metadata(row: dict) -> dict:
    """把一行 chunk 记录转成 ChromaDB 能接受的标量 metadata。"""
    return {
        "document_id": int(row["document_id"]),
        "chunk_id": int(row["chunk_id"]),
        "chunk_index": int(row["chunk_index"]),
        "title": str(row.get("title", "")),
        "file_path": str(row.get("file_path", "")),
        "heading": str(row.get("heading", "")),
        "source": str(row.get("source", "")),
        "folder": str(row.get("folder", "")),
        "tags": ",".join(row["tags"]) if isinstance(row.get("tags"), list) else str(row.get("tags", "")),
    }
