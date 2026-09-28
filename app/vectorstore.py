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

    def query(self, embedding: list[float], n_results: int = 5, where: dict | None = None) -> list[dict]:
        """按向量找最近的 n_results 条，返回 [{content, distance, **metadata}]。

        where 是 Chroma 的 metadata 过滤条件（scoped retrieval 用它限定 domain），
        None / 空值表示不过滤——全库检索的路径必须保持和以前逐字节一致。
        Chroma 原生的返回是 result["ids"][0] / result["metadatas"][0] 这种嵌套结构，
        在这里摊平成列表，上层就不用知道 Chroma 的返回格式了。
        """
        query_kwargs = {"where": where} if where else {}
        result = self._col.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
            **query_kwargs,
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

    def refresh_domain_metadata(self, rows: list[dict]) -> int:
        """把 SQLite 行算出的期望 metadata 回填到 Chroma，只改 metadata。

        为什么不用重建：现有向量的数值本身没有错，错的只是 metadata 缺 domain 字段。
        col.update 只写 metadata，不重算 embedding、不删任何向量——这是对
        "改映射表就能让全库生效"这个需求的最低代价路径。重建（rebuild_vectors）
        要把 14303 个 chunk 全部重新过一遍 BGE-M3，那是给换模型用的，杀鸡用牛刀。

        rows 是 get_chunks_for_reindex 的输出（SQLite 是唯一事实来源）；
        期望值按 _metadata 计算，和 ingest 写入的格式逐字段一致。
        只更新和当前值不同的条目，返回更新的条数。
        SQLite 有、Chroma 没有的 chunk 在这里跳过——补齐它属于 ingest/重建的职责，
        元数据回填不顺便造向量。
        """
        got = self._col.get(include=["metadatas"])
        current = dict(zip(got["ids"], got["metadatas"]))
        changed_ids: list[str] = []
        changed_metas: list[dict] = []
        for row in rows:
            chunk_id = str(row["chunk_id"])
            existing = current.get(chunk_id)
            if existing is None:
                continue
            desired = _metadata(row)
            if {key: existing.get(key) for key in desired} != desired:
                changed_ids.append(chunk_id)
                changed_metas.append(desired)

        for start in range(0, len(changed_ids), METADATA_UPDATE_BATCH):
            self._col.update(
                ids=changed_ids[start : start + METADATA_UPDATE_BATCH],
                metadatas=changed_metas[start : start + METADATA_UPDATE_BATCH],
            )
        return len(changed_ids)


# metadata 回填时一次 col.update 带多少条。Chroma 单次调用负载和批大小成正比，
# 500 条 metadata（不含向量）约几百 KB，分批只影响内存峰值不影响结果。
METADATA_UPDATE_BATCH = 500


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
        "domain": str(row.get("domain", "")),
        "source_type": str(row.get("source_type", "")),
    }
