"""Personal Context：用户自己维护的「当前状态」。

它不是知识库，也不属于 RAG 链路里的任何一环：
    - 不进 SQLite、不进 Chroma、不进那 14303 个 chunk；
    - 不算 embedding，不做向量检索；
    - V1 只读：程序只读不写，不自动推断、不自动写入、不自动更新。

数据来源是用户手写的单个 YAML 文件（模板见 personal_context/example.yaml）。
之所以用 YAML 而不是五个 Markdown：schema 可以直接校验、Git diff 更干净、
一个文件就能覆盖「当前状态」的全部五个段；人类可读性靠 YAML 注释 +
personal_context/README.md 的字段说明来补。

文件不存在、文件为空、单个字段留空，都是正常状态——系统照常工作，
只是回答时明确说「当前没有个人状态信息」，绝不凭空编造用户情况。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---- 显式 schema ----
# 顶层五个段。顺序决定渲染顺序，不参与校验。
SECTIONS = ("profile", "goals", "projects", "constraints", "decision_preferences")

PROFILE_FIELDS = ("background", "current_direction", "skills")
GOALS_FIELDS = ("long_term", "current", "near_term")
PROJECT_FIELDS = ("name", "status", "priority", "next_step", "blockers")
CONSTRAINT_FIELDS = ("time", "resources", "hard_constraints")

# 渲染 Prompt 用的中文标签。key 顺序即展示顺序。
_PROFILE_LABELS = {"background": "背景", "current_direction": "当前方向", "skills": "技能"}
_GOALS_LABELS = {"long_term": "长期目标", "current": "当前目标", "near_term": "近期目标"}
_CONSTRAINT_LABELS = {"time": "时间", "resources": "资源", "hard_constraints": "硬约束"}
_PROJECT_LABELS = {"status": "状态", "priority": "优先级", "next_step": "下一步", "blockers": "阻塞"}

# 字段值可以是字符串、字符串列表，或者空（YAML 里 `key:` 单独成行 = None）
Scalar = str | list[str]


class PersonalContextError(ValueError):
    """YAML 结构写错了（例如 projects 写成了字符串）。

    这里区分两种情况：内容缺失不该报错，结构错了必须报错。
    前者代表「用户还没填」，后者代表「程序读不懂」，混在一起会让用户无从下手。
    """


@dataclass(frozen=True)
class PersonalContext:
    """一次读取的结果。loaded=False 表示文件不存在或没有任何内容。"""

    loaded: bool
    profile: dict[str, Scalar] = field(default_factory=dict)
    goals: dict[str, Scalar] = field(default_factory=dict)
    projects: list[dict[str, Scalar]] = field(default_factory=list)
    constraints: dict[str, Scalar] = field(default_factory=dict)
    decision_preferences: list[str] = field(default_factory=list)
    source_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def available_sections(self) -> list[str]:
        """哪些段真的有内容。渲染 Prompt 时只出这些段，
        一堆「- 当前方向：」这样的空行会让模型误以为系统掌握了什么。"""
        sections: list[str] = []
        if _mapping_has_content(self.profile):
            sections.append("profile")
        if _mapping_has_content(self.goals):
            sections.append("goals")
        if any(_mapping_has_content(p) for p in self.projects):
            sections.append("projects")
        if _mapping_has_content(self.constraints):
            sections.append("constraints")
        if self.decision_preferences:
            sections.append("decision_preferences")
        return sections

    @property
    def has_content(self) -> bool:
        """有没有任何一段内容。False 时调用方必须明确告诉模型「没有个人状态」。"""
        return bool(self.available_sections)

    def to_debug_dict(self) -> dict:
        """给 /api/chat 响应用的过程信息：调用了什么文件、有哪些段、有哪些告警。"""
        return {
            "loaded": self.loaded,
            "source_path": self.source_path,
            "available_sections": self.available_sections,
            "project_count": len(self.projects),
            "warnings": list(self.warnings),
        }


def load_personal_context(path: Path) -> PersonalContext:
    """读一个 Personal Context YAML。文件不存在 / 空文件 → loaded=False，不报错。

    schema 校验只针对结构：段类型不对、projects 条目缺 name 才抛 PersonalContextError。
    多出来的字段（包括拼错的字段名）不报错，只记进 warnings 后忽略——
    用户手写文件会加自己的备注，把这类情况当崩溃会让维护体验很差。
    """
    if not path.is_file():
        return PersonalContext(loaded=False, source_path=str(path))

    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) if text.strip() else None
    if raw is None:
        return PersonalContext(loaded=False, source_path=str(path))
    if not isinstance(raw, dict):
        raise PersonalContextError(
            f"Personal Context 顶层必须是 YAML 映射，实际是 {type(raw).__name__}"
        )

    warnings: list[str] = []
    for unknown in sorted(set(raw) - set(SECTIONS)):
        warnings.append(f"未知顶层字段 {unknown!r} 已忽略")

    return PersonalContext(
        loaded=True,
        source_path=str(path),
        profile=_load_mapping(raw.get("profile"), PROFILE_FIELDS, "profile", warnings),
        goals=_load_mapping(raw.get("goals"), GOALS_FIELDS, "goals", warnings),
        projects=_load_projects(raw.get("projects"), warnings),
        constraints=_load_mapping(raw.get("constraints"), CONSTRAINT_FIELDS, "constraints", warnings),
        decision_preferences=_load_string_list(
            raw.get("decision_preferences"), "decision_preferences",
        ),
        warnings=warnings,
    )


def render_for_prompt(ctx: PersonalContext) -> str:
    """把 Personal Context 渲染成 Prompt 里的【用户当前情况】正文。

    没有任何内容时返回空串，由调用方决定怎么占位——
    「没有个人状态」必须在 Prompt 里显式说，不能让模型看着空段自己脑补。
    """
    blocks: list[str] = []

    mapping_blocks = (
        _render_mapping("【个人档案】", ctx.profile, _PROFILE_LABELS),
        _render_mapping("【目标】", ctx.goals, _GOALS_LABELS),
        _render_mapping("【约束】", ctx.constraints, _CONSTRAINT_LABELS),
    )
    for block in mapping_blocks:
        if block is not None:
            blocks.append(block)

    projects_block = _render_projects(ctx.projects)
    if projects_block is not None:
        blocks.append(projects_block)

    preferences_block = _render_preferences(ctx.decision_preferences)
    if preferences_block is not None:
        blocks.append(preferences_block)

    return "\n\n".join(blocks)


# ---- 校验与归一化 ----


def _load_mapping(
    value: object, allowed: tuple[str, ...], name: str, warnings: list[str],
) -> dict[str, Scalar]:
    """读一个「字段映射」段（profile / goals / constraints）。整段缺失 → 空 dict。"""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PersonalContextError(f"{name} 必须是字段映射，实际是 {type(value).__name__}")

    out: dict[str, Scalar] = {}
    for key, item in value.items():
        if key not in allowed:
            warnings.append(f"{name}.{key} 不在 schema 中，已忽略")
            continue
        out[key] = _as_scalar(item, f"{name}.{key}")
    return out


def _load_projects(value: object, warnings: list[str]) -> list[dict[str, Scalar]]:
    """读 projects 段。整段缺失 → 空列表；条目缺非空 name → schema 错误。

    name 是唯一强校验的字段：没有名字的项目条目在 Prompt 里没有意义，
    留着只会让模型把它当成一条无来源的断言。
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise PersonalContextError(f"projects 必须是列表，实际是 {type(value).__name__}")

    out: list[dict[str, Scalar]] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise PersonalContextError(
                f"projects[{i}] 必须是字段映射，实际是 {type(item).__name__}"
            )
        entry: dict[str, Scalar] = {}
        for key, v in item.items():
            if key not in PROJECT_FIELDS:
                warnings.append(f"projects[{i}].{key} 不在 schema 中，已忽略")
                continue
            entry[key] = _as_scalar(v, f"projects[{i}].{key}")
        if not str(entry.get("name", "")).strip():
            raise PersonalContextError(f"projects[{i}] 缺少非空的 name 字段")
        out.append(entry)
    return out


def _load_string_list(value: object, name: str) -> list[str]:
    """读 decision_preferences 段：字符串列表。空白条目丢弃，不让它们占位。"""
    if value is None:
        return []
    if not isinstance(value, list):
        raise PersonalContextError(f"{name} 必须是字符串列表，实际是 {type(value).__name__}")
    return [str(v).strip() for v in value if v is not None and str(v).strip()]


def _as_scalar(value: object, label: str) -> Scalar:
    """归一化单个字段：空 → 空串，字符串 → 去首尾空白，列表 → 递归归一化。

    其它类型一律报错：YAML 里 `time: 120` 会解析成 int，
    不拦住的话它会静默进 Prompt，用户看不出任何异常。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [_as_scalar(item, f"{label}[{i}]") for i, item in enumerate(value)]
    raise PersonalContextError(
        f"{label} 只能是字符串或字符串列表，实际是 {type(value).__name__}"
    )


def _has_content(value: Scalar | None) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return any(str(item).strip() for item in value)


def _mapping_has_content(mapping: dict[str, Scalar]) -> bool:
    return any(_has_content(v) for v in mapping.values())


# ---- 渲染 ----


def _format_value(value: Scalar | None) -> str:
    """把字段值压成一行：列表用顿号连起来，避免渲染出 YAML 的方括号。"""
    if isinstance(value, list):
        return "、".join(str(item) for item in value if str(item).strip())
    return str(value).strip()


def _render_mapping(
    title: str, mapping: dict[str, Scalar], labels: dict[str, str],
) -> str | None:
    """渲染一个字段映射段。全段为空 → 返回 None（调用方跳过，不留空标题）。"""
    lines = [title]
    for key, label in labels.items():
        value = mapping.get(key)
        if _has_content(value):
            lines.append(f"- {label}：{_format_value(value)}")
    return "\n".join(lines) if len(lines) > 1 else None


def _render_projects(projects: list[dict[str, Scalar]]) -> str | None:
    lines: list[str] = []
    for i, project in enumerate(projects, start=1):
        if not _mapping_has_content(project):
            continue
        name = str(project.get("name", "")).strip()
        lines.append(f"{i}. {name}")
        for key, label in _PROJECT_LABELS.items():
            value = project.get(key)
            if _has_content(value):
                lines.append(f"   - {label}：{_format_value(value)}")
    if not lines:
        return None
    return "【项目】\n" + "\n".join(lines)


def _render_preferences(preferences: list[str]) -> str | None:
    if not preferences:
        return None
    lines = ["【决策偏好】"] + [f"{i}. {item}" for i, item in enumerate(preferences, start=1)]
    return "\n".join(lines)
