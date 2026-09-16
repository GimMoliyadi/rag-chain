"""测试一：Markdown 可以成功读取，且只读不改原文。"""
from __future__ import annotations

from pathlib import Path

from app.parser import iter_markdown_files, parse_markdown


def test_frontmatter_and_body_are_parsed(fake_vault: Path):
    doc = parse_markdown(fake_vault / "法律" / "合同违约.md", fake_vault)

    # frontmatter 字段
    assert doc.title == "合同违约的认定与救济"
    assert doc.source == "民法典笔记"
    assert "法律" in doc.tags and "合同法" in doc.tags
    assert doc.created_at  # created: 2026-01-10 被解析出来了
    # 文件正文里的内联 #标签也会被收集
    assert "免责事由" in doc.content

    # file_path 是相对 Vault 的 posix 路径，换机器也成立
    assert doc.file_path == "法律/合同违约.md"
    assert doc.folder == "法律"


def test_flow_list_tags_and_inline_tags(fake_vault: Path):
    doc = parse_markdown(fake_vault / "创业" / "MVP.md", fake_vault)
    assert doc.title == "最小可行产品"
    assert set(doc.tags) >= {"创业", "MVP"}


def test_content_is_not_stripped_away(fake_vault: Path):
    """frontmatter 本身不该混进 content。"""
    doc = parse_markdown(fake_vault / "创业" / "MVP.md", fake_vault)
    assert "tags:" not in doc.content
    assert doc.content.startswith("# 最小可行产品")


def test_hidden_dirs_and_attachments_are_skipped(fake_vault: Path):
    """只要 .md；.obsidian/ 和附件必须排除，否则会污染索引。"""
    files = [p for p in iter_markdown_files(fake_vault)]
    names = {p.name for p in files}
    assert names == {"合同违约.md", "MVP.md"}
    assert all(".obsidian" not in p.parts for p in files)


def test_vault_is_left_untouched(fake_vault: Path):
    """核心红线：解析过程绝不修改原始 Vault。"""
    before = (fake_vault / "法律" / "合同违约.md").read_bytes()
    parse_markdown(fake_vault / "法律" / "合同违约.md", fake_vault)
    assert (fake_vault / "法律" / "合同违约.md").read_bytes() == before
