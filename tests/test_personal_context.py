"""Personal Context 加载 / 校验 / 渲染测试。

核心契约：
- 文件不存在、空文件、字段留空都不报错——缺内容不是错误；
- 结构写错才报 PersonalContextError（projects 写成字符串、条目缺 name）；
- 多出来的字段被忽略但记告警，不让用户自己的备注搞崩系统；
- 不触碰 Chroma / SQLite / embedding：这是一个纯文件读取模块。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.personal_context import (
    PersonalContext,
    PersonalContextError,
    load_personal_context,
    render_for_prompt,
)

FULL_YAML = """
profile:
  background: 华工安全工程
  current_direction: AI 应用开发
  skills:
    - Python
    - RAG

goals:
  long_term: 做出一个能用的个人顾问
  current: 完善 RAG 检索层
  near_term: 跑一遍 personal advisor 评测

projects:
  - name: Obsidian RAG Personal Advisor
    status: 进行中
    priority: 高
    next_step: 跑评测
    blockers:
      - 缺少人工标注的评测集
  - name: 另一个项目
    status: 未开始

constraints:
  time: 每天两小时
  resources: 一台 6GB 显存的笔记本
  hard_constraints:
    - 不能改动已有的 Obsidian Vault

decision_preferences:
  - 优先把已有的东西做完
  - 能验证的结论比感觉可靠
"""

EMPTY_YAML = """
profile:
  background: ""
  current_direction: ""
  skills: []

goals:
  long_term: ""
  current: ""
  near_term: ""

projects: []

constraints:
  time: ""
  resources: ""
  hard_constraints: []

decision_preferences: []
"""


def test_missing_file_is_loaded_false(tmp_path):
    ctx = load_personal_context(tmp_path / "不存在.yaml")
    assert ctx.loaded is False
    assert ctx.has_content is False
    assert ctx.available_sections == []
    assert render_for_prompt(ctx) == ""


def test_empty_file_is_loaded_false(tmp_path):
    path = tmp_path / "context.yaml"
    path.write_text("# 只有注释\n\n   \n", encoding="utf-8")

    ctx = load_personal_context(path)

    assert ctx.loaded is False
    assert ctx.has_content is False


def test_all_empty_fields_are_loaded_but_have_no_content(tmp_path):
    """字段留空 = 用户还没填，必须当成「有文件但没有内容」而不是结构错误。"""
    path = tmp_path / "context.yaml"
    path.write_text(EMPTY_YAML, encoding="utf-8")

    ctx = load_personal_context(path)

    assert ctx.loaded is True
    assert ctx.has_content is False
    assert render_for_prompt(ctx) == ""


def test_full_yaml_parses_every_section(tmp_path):
    path = tmp_path / "context.yaml"
    path.write_text(FULL_YAML, encoding="utf-8")

    ctx = load_personal_context(path)

    assert ctx.loaded is True and ctx.has_content is True
    assert ctx.available_sections == ["profile", "goals", "projects", "constraints", "decision_preferences"]
    assert ctx.profile["current_direction"] == "AI 应用开发"
    assert ctx.profile["skills"] == ["Python", "RAG"]
    assert ctx.goals["near_term"] == "跑一遍 personal advisor 评测"
    assert len(ctx.projects) == 2
    assert ctx.projects[0]["blockers"] == ["缺少人工标注的评测集"]
    assert ctx.constraints["hard_constraints"] == ["不能改动已有的 Obsidian Vault"]
    assert len(ctx.decision_preferences) == 2
    assert ctx.warnings == []


def test_single_empty_field_does_not_error(tmp_path):
    """只填一个字段，其余留空：不能因为某个字段空就抛错。"""
    path = tmp_path / "context.yaml"
    path.write_text("profile:\n  current_direction: 只做后端\n", encoding="utf-8")

    ctx = load_personal_context(path)

    assert ctx.loaded is True
    assert ctx.available_sections == ["profile"]
    assert ctx.profile["current_direction"] == "只做后端"
    assert "background" not in ctx.profile


def test_project_without_name_is_schema_error(tmp_path):
    path = tmp_path / "context.yaml"
    path.write_text("projects:\n  - status: 进行中\n", encoding="utf-8")

    with pytest.raises(PersonalContextError, match="name"):
        load_personal_context(path)


def test_projects_not_a_list_is_schema_error(tmp_path):
    path = tmp_path / "context.yaml"
    path.write_text("projects: 我在同时做三个项目\n", encoding="utf-8")

    with pytest.raises(PersonalContextError, match="projects 必须是列表"):
        load_personal_context(path)


def test_profile_not_a_mapping_is_schema_error(tmp_path):
    path = tmp_path / "context.yaml"
    path.write_text("profile:\n  - background\n", encoding="utf-8")

    with pytest.raises(PersonalContextError, match="profile 必须是字段映射"):
        load_personal_context(path)


def test_non_string_value_is_schema_error(tmp_path):
    """YAML 里 `time: 120` 会解析成 int：必须拦住，否则会静默混进 Prompt。"""
    path = tmp_path / "context.yaml"
    path.write_text("constraints:\n  time: 120\n", encoding="utf-8")

    with pytest.raises(PersonalContextError, match="time"):
        load_personal_context(path)


def test_top_level_not_a_mapping_is_schema_error(tmp_path):
    path = tmp_path / "context.yaml"
    path.write_text("- 只是个列表\n", encoding="utf-8")

    with pytest.raises(PersonalContextError, match="顶层必须是 YAML 映射"):
        load_personal_context(path)


def test_unknown_fields_are_ignored_with_warnings(tmp_path):
    """用户加自己的备注不该崩，但拼错的字段名必须能在告警里看到。"""
    path = tmp_path / "context.yaml"
    path.write_text(
        "profile:\n"
        "  current_direction: 后端\n"
        "  currrrent_direction: 拼错了\n"
        "  notes: 我的自定义备注\n"
        "custom_section: 我自己的段\n"
        "projects:\n"
        "  - name: 项目A\n"
        "    deadline: 下周三\n",
        encoding="utf-8",
    )

    ctx = load_personal_context(path)

    assert ctx.profile["current_direction"] == "后端"
    assert "currrrent_direction" not in ctx.profile
    assert "notes" not in ctx.profile
    assert "deadline" not in ctx.projects[0]
    assert len(ctx.warnings) == 4
    joined = " ".join(ctx.warnings)
    for field in ("currrrent_direction", "notes", "custom_section", "deadline"):
        assert field in joined, f"拼错的字段名 {field} 必须出现在告警里"


def test_render_omits_empty_sections_and_labels_correctly(tmp_path):
    path = tmp_path / "context.yaml"
    path.write_text(
        "profile:\n  current_direction: AI 应用开发\n"
        "projects:\n  - name: 项目A\n    status: 进行中\n",
        encoding="utf-8",
    )

    rendered = render_for_prompt(load_personal_context(path))

    assert "【个人档案】" in rendered and "- 当前方向：AI 应用开发" in rendered
    assert "【项目】" in rendered and "1. 项目A" in rendered and "- 状态：进行中" in rendered
    assert "【目标】" not in rendered      # 整段为空 → 不出空标题
    assert "【约束】" not in rendered
    assert "【决策偏好】" not in rendered


def test_render_lists_are_joined_inline(tmp_path):
    path = tmp_path / "context.yaml"
    path.write_text("profile:\n  skills:\n    - Python\n    - RAG\n", encoding="utf-8")

    assert "- 技能：Python、RAG" in render_for_prompt(load_personal_context(path))


def test_debug_dict_shape(tmp_path):
    path = tmp_path / "context.yaml"
    path.write_text("profile:\n  background: x\n", encoding="utf-8")

    info = load_personal_context(path).to_debug_dict()

    assert info == {
        "loaded": True,
        "source_path": str(path),
        "available_sections": ["profile"],
        "project_count": 0,
        "warnings": [],
    }


def test_missing_file_debug_dict(tmp_path):
    info = load_personal_context(tmp_path / "nope.yaml").to_debug_dict()
    assert info["loaded"] is False
    assert info["available_sections"] == []


def test_module_only_imports_standard_library_and_yaml():
    """Personal Context 绝不能进 Chroma / SQLite / embedding。

    只扫真正的 import 语句（文档字符串里提到关键词不算），
    依赖面被锁死在 pathlib / dataclasses / yaml 三样。
    """
    source = (Path(__file__).resolve().parent.parent / "app" / "personal_context.py").read_text(
        encoding="utf-8",
    )
    modules: set[str] = set()
    for line in source.splitlines():
        match = re.match(r"^(?:from|import)\s+([\w.]+)", line)
        if match:
            modules.add(match.group(1).split(".")[0])

    assert modules == {"__future__", "dataclasses", "pathlib", "yaml"}


def test_render_is_pure(tmp_path):
    """渲染同一个上下文两次结果一致：没有隐藏状态。"""
    ctx = PersonalContext(loaded=True, profile={"background": "x"})
    assert render_for_prompt(ctx) == render_for_prompt(ctx)
