# Personal Context

给 Personal Advisor 用的「用户当前状态」。它回答的问题不是「知识库里有什么」，
而是「你现在在做什么」——两者是不同性质的数据，所以在系统里走完全不同的路径。

## 它是什么 / 不是什么

| | Personal Context（本目录） | 知识库（Obsidian Vault） |
|---|---|---|
| 回答 | 用户当前的状态、目标、约束 | 用户存下来的知识 |
| 内容举例 | 当前方向、在推的项目、近期目标 | 博弈论、精益创业、机械设计、韩立模式、哲学笔记 |
| 存储 | 一个 YAML 文件 | SQLite + Chroma（911 篇文档 / 14303 个 chunk） |
| 读取方式 | 每次 advisor 提问**全量读取** | 向量检索 Top-K |
| 程序是否写入 | **只读**，改文件即生效 | 由 ingest 维护 |

明确的红线：Personal Context **不写入** SQLite、Chroma 或那 14303 个 chunk，
**不需要** embedding，**不做**向量检索。它体量小、语义是「此刻的状态」，
所以每次 advisor 提问直接全文读进来塞进 Prompt 的第一段。

## 文件

- `context.yaml` —— 实际文件。程序读的是这个（路径由 `.env` 的 `PERSONAL_CONTEXT_PATH` 控制）。
- `example.yaml` —— 带填写说明的模板。
- `README.md` —— 本文件。

## 字段说明

schema 是显式的，五个顶层段：

| 段 | 字段 | 说明 |
|---|---|---|
| `profile` | `background` / `current_direction` / `skills` | 你是谁、现在往哪走、手上有什么能力 |
| `goals` | `long_term` / `current` / `near_term` | 三档粒度，别混写 |
| `projects` | `name` / `status` / `priority` / `next_step` / `blockers` | 每个项目一条；**`name` 必填且不能为空**，其余可留空 |
| `constraints` | `time` / `resources` / `hard_constraints` | 什么会限制你的选择 |
| `decision_preferences` | 字符串列表 | 你做选择时一贯的原则，直接写句子 |

除了 `projects[].name`，**所有字段都允许留空**——空字段不会报错，也不会被渲染进 Prompt。

不要往里写这些：人格画像、能力评分、从聊天记录推断出来的习惯、
任何你自己都没确认过的信息。这个文件是你的亲述事实，系统会把它当事实用。

## 校验行为

- 文件不存在 / 空文件 / 全部字段留空 → 系统照常工作，但回答时**明确说缺少你的当前情况**，不猜。
- 结构写错（如 `projects` 写成了字符串、某个项目条目没有 `name`）→ 报 `PersonalContextError`，
  该次问答照常回答（个人情况那一栏会显示缺失），错误原因出现在响应的 `personal_context.error` 里。
- 多出来的字段（含拼错的字段名）→ 忽略，并在 `personal_context.warnings` 里留一条告警。
  这样你加自己的备注不会崩，但拼错也能被看到。

## 怎么验证它有没有生效

```bat
python ask.py --mode advisor "我目前主要方向是什么？"
```

看输出的 `[personal_context]` 那行：来源文件 + 哪些段有内容。
如果是「无内容」或「未启用」，回答开头一定会说明缺少你的当前情况。

用普通模式（默认 `normal`）时 Personal Context 完全不会被读取。

## 提交到 Git

`context.yaml` 默认随仓库一起提交，方便你追踪自己状态的变更。
如果你觉得里面的内容不想进版本库，把 `personal_context/context.yaml` 加进 `.gitignore`，
模板 `example.yaml` 仍然保留。
