"""Obsidian Markdown 解析器。

职责：把一个 .md 文件读成一个结构化的 ParsedDocument。
只读文件，绝不写回 Vault。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

# YAML frontmatter 必须出现在文件最开头：首行就是 ---，然后到下一个 --- 结束
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
# 正文里第一个 # 开头的标题，用来在 frontmatter 没写 title 时兜底
FIRST_HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
# Obsidian 行内标签：#标签，要求前面是行首或空白，避免误伤网址里的 #
INLINE_TAG_RE = re.compile(r"(?:^|[\s(])#([\w一-鿿][\w一-鿿/-]*)")

# 明确不入库的目录。Obsidian 的配置、插件、回收站、缓存都不属于知识内容
IGNORED_DIRS = {".obsidian", ".trash", ".git", "__pycache__", ".claudian", ".spec-workflow"}

# 带二进制内容的扩展名，避免把附件当文档读
ATTACHMENT_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".pdf", ".docx", ".xlsx",
    ".pptx", ".zip", ".rar", ".7z", ".mp3", ".mp4", ".wav", ".mov",
    ".ttf", ".otf", ".woff", ".woff2", ".woff", ".css", ".js", ".exe", ".dll",
}


@dataclass
class ParsedDocument:
    """解析后的单个 Markdown 文档。file_path 是相对 Vault 的路径，跨机器可复现。"""

    title: str
    file_path: str
    folder: str
    tags: list[str] = field(default_factory=list)
    source: str = ""
    content: str = ""            # 去掉 frontmatter 后的正文
    created_at: str = ""
    updated_at: str = ""


def iter_markdown_files(vault_path: Path, path_filter: str | None = None, limit: int | None = None) -> list[Path]:
    """递归找出 Vault 里所有需要处理的 .md 文件。

    path_filter 是相对路径的片段匹配，用来只 ingest 某个子目录，
    比如 ingest.py --path "01-精益创业2.0"。这对调试很重要：
    全库 900 多个文件，先跑一个子目录就能验证链路。
    limit 用于限制数量，做小规模冒烟测试。
    """
    results: list[Path] = []
    for root, dirs, files in os.walk(vault_path):
        # 原地过滤子目录：隐藏目录（.obsidian 等）直接不往下走，省时间
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]

        for name in files:
            if not name.endswith(".md") or name.startswith("."):
                continue
            full = Path(root) / name
            if full.suffix.lower() in ATTACHMENT_EXTS:
                continue
            rel = full.relative_to(vault_path).as_posix()
            if path_filter and path_filter not in rel:
                continue
            results.append(full)
            if limit and len(results) >= limit:
                return results
    return results


def parse_markdown(path: Path, vault_path: Path) -> ParsedDocument:
    """解析单个 .md 文件为 ParsedDocument。"""
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta, content = _split_frontmatter(raw)

    tags = _collect_tags(meta, content)
    return ParsedDocument(
        title=_pick_title(meta, content, path),
        file_path=path.relative_to(vault_path).as_posix(),
        folder=str(Path(path.relative_to(vault_path).parent)) if path.parent != vault_path else "",
        tags=tags,
        source=_pick_source(meta),
        content=content,
        created_at=_pick_field(meta, ("created_at", "created", "date")),
        updated_at=_pick_field(meta, ("updated_at", "updated", "modified")) or _mtime_str(path),
    )


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    """拆出 YAML frontmatter 和正文。没有 frontmatter 就返回空 dict。"""
    m = FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        # frontmatter 写坏了不应该让整次 ingest 崩溃，降级当作无 frontmatter
        return {}, raw
    if not isinstance(meta, dict):
        return {}, raw
    return meta, raw[m.end():].lstrip("\n")


def _pick_title(meta: dict, content: str, path: Path) -> str:
    """标题优先级：frontmatter.title > 正文第一个一级标题 > 文件名。"""
    if meta.get("title"):
        return str(meta["title"]).strip()
    m = FIRST_HEADING_RE.search(content)
    if m:
        return m.group(1).strip()
    return path.stem


def _pick_source(meta: dict) -> str:
    """来源优先级：source 字段 > sources 列表 > type 字段。

    sources 可能是字符串列表，也可能是 [{source_id: xxx}] 这种嵌套结构，
    两种都要能吃，所以递归提取所有字符串叶子节点。
    """
    if meta.get("source"):
        return _join_values(meta["source"])
    sources = meta.get("sources")
    if sources:
        return _join_values(sources)
    if meta.get("type"):
        return _join_values(meta["type"])
    return ""


def _join_values(value) -> str:
    """把标量、列表、或嵌套 dict 统一压成逗号分隔的字符串。"""
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.append(_join_values(item))
        return ",".join(p for p in parts if p)
    if isinstance(value, dict):
        # 嵌套 dict 取 id/name/url 这类最有信息量的键，取不到就整体拼接
        for key in ("source_id", "id", "title", "name", "url"):
            if value.get(key):
                return str(value[key])
        return ",".join(str(v) for v in value.values() if v)
    if value is None:
        return ""
    return str(value)


def _pick_field(meta: dict, keys: tuple[str, ...]) -> str:
    """按候选键顺序取第一个非空值，兼容同一含义的多种写法。

    同一个字段在不同人的 frontmatter 里写法不同（created / created_at / date），
    全都要能吃，否则时间戳会大面积丢失。YAML 会把裸日期解析成 date 对象，str() 即可。
    """
    for key in keys:
        value = meta.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _collect_tags(meta: dict, content: str) -> list[str]:
    """frontmatter.tags + 正文行内 #标签，去重且保序。"""
    tags: list[str] = []
    raw_tags = meta.get("tags")
    if isinstance(raw_tags, str):
        # 有人写 tags: 法律,合同（字符串），有人写列表，两种都支持
        raw_tags = [t.strip() for t in raw_tags.split(",")]
    if isinstance(raw_tags, list):
        for t in raw_tags:
            tag = str(t).strip().lstrip("#")
            if tag:
                tags.append(tag)
    for m in INLINE_TAG_RE.finditer(content):
        tags.append(m.group(1))
    return _dedup(tags)


def _dedup(values: list[str]) -> list[str]:
    """去重同时保持原有顺序。"""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _mtime_str(path: Path) -> str:
    """把文件修改时间格式化成 ISO 字符串，作为 updated_at 的兜底值。"""
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
