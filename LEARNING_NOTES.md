# LEARNING_NOTES —— 从概念到调用链

这份文档的目标：让零基础的人读完就能看懂这个项目在干什么、每个文件为什么存在、一个问题从输入到输出的完整路径。

概念部分先建立词汇表，最后一条完整的调用链把它们串起来。

---

# 第一部分：十个概念

## 1. SQLite 是什么

SQLite 是一个**文件形式的数据库**。整个数据库就是一个 `.db` 文件（这里是 `data/knowledge.db`），没有独立的服务进程，不需要密码，复制文件就等于备份数据库。

和 Excel 的本质区别不是"能存更多行"，而是**你可以写 SQL 去精确查询**：

```sql
-- 找出所有标签包含"法律"的文档，只取标题和路径
SELECT title, file_path FROM documents WHERE tags LIKE '%法律%' ORDER BY title;

-- 找出某篇文档被切成了几块
SELECT count(*) FROM chunks WHERE document_id = 42;
```

Excel 里你想干这两件事只能手动筛选排序。数据量一大（本库 911 篇文档 / 14303 个 chunk），手动就干不了了。

本项目里 SQLite 存两张表：

```
documents（一篇文档一行）
  id, title, file_path, folder, tags, source, content, content_hash, created_at, updated_at

chunks（一个切块一行）
  id, document_id → documents(id), chunk_index, content, heading, start_position, end_position, created_at
```

两张表用 `document_id` 关联（外键）。这就是"关系型"的含义：两张表通过一个字段互相指着。

---

## 2. 为什么 Obsidian 不等于数据库

Obsidian 本质是一堆**纯文本文件**放在文件夹里。它能搜索，是因为它在你打开笔记时把内容扫一遍做全文匹配。

这带来几个根本限制：

| | 文件夹 + 文本文件 | 数据库 |
|---|---|---|
| 按标签筛选 | 要装插件扫描 frontmatter | 一行 SQL |
| 按字数排序 | 要外部脚本统计 | `ORDER BY length(content)` |
| 统计文档数量 | 数文件个数 | `SELECT count(*)` |
| 记录"这段文字属于哪篇笔记的第几块" | 文本文件里没有这个概念 | 一张 chunks 表 |
| 判断"这篇笔记有没有变化" | 靠文件修改时间（不可靠） | 存内容哈希（精确） |

关键的一条：**Obsidian 不知道"chunk"这个概念**。文本文件只有"整个文件"，没有"这个文件被切成了 12 段，每段来自哪个位置"。而 RAG 需要的是段落级的东西，所以必须有一个数据库来存这种结构。

本项目把 Obsidian 当作**唯一的内容来源（只读）**，把结构化的东西（chunk 边界、内容哈希、关联关系）存进 SQLite。这样既不动你的笔记，又能做数据库能做的事。

---

## 3. 什么是 Chunk（切块）

Chunk 就是**把长文档切成的一小段一小段**。

为什么要切？两个原因：

1. **LLM 有上下文长度上限**，不可能把 911 篇笔记（几十万字）一次塞进去。
2. **即使塞得下也没用**。检索时你需要的是"最相关的那几段"，不是整本《本草纲目》。

为什么不能随便切？**因为切块的方式直接决定检索质量和引用可信度。**

本项目的切块逻辑（[app/chunker.py](app/chunker.py)）是分层递进的：

```
原文
  └─ ① 按 Markdown 标题（# ## ###）切成 section     → 保留章节边界
       └─ ② 每个 section 按空行切成段落
            └─ ③ 段落超过 chunk_size 时按中文句号切成句子
                 └─ ④ 贪心合并到接近 800 字
                      └─ ⑤ 相邻 chunk 保留 120 字重叠
```

几个刻意的设计取舍：

- **不跨章节合并。** 第 3 节末尾和第 4 节开头的文字拼成一个 chunk 看起来很省空间，但语义混杂，检索命中后会把两个主题一起塞给模型。宁可短章节产生偏小的 chunk。
- **保留重叠。** 一句话如果正好落在两个 chunk 的交界处，完全不重叠就会两头都缺半句。重叠 120 字让边界信息两边都有。
- **记录字符下标。** 每个 chunk 存 `start_position` / `end_position`，是真的能切回原文的（`原文[1234:2034] == chunk.content`）。调试时可以直接定位"这个 chunk 到底来自文档哪一段"。
- **跳过只有标题的 section。** 否则每篇文档开头都会多出一个十几个字的垃圾 chunk（比如内容只是 `# 向量数据库入门`）。

有一条测试专门验证边界质量：chunk 的结尾必须落在 `。！？；、` 等句读上。这就是"不按固定字符数硬切"的可执行证据。

但要留意一个隐含的单位换算：切块按**字符数**切，而 embedding 模型有 **token 数**上限。中文实测约 0.71 token/字符，`CHUNK_SIZE=800` 通常落在 560 token 左右，远低于本项目设置的 1024 token 上限；但少数 chunk 会超——全库实测 **12 / 14303** 个 chunk 超过 1024 token，最长 3654 token（《本草纲目》的长表格）。这 12 条会被截断到 1024 token，`--rebuild-vectors` 会逐条打印 `chunk_id / document / token_length` 并在结束时汇总数量，**不是静默截断**。彻底解决要改成按 token 边界切块（token-aware chunk splitting），目前列在 README 的「后续优化」里，本轮刻意没做。

---

## 4. 什么是 Embedding

Embedding 就是**把一段文字变成一串数字（向量）**，让"意思相近"变成"数字相近"。

```
"什么是 POVM？"        → [0.12, -0.34, 0.87, 0.05, ...]   ← 1024 个数字
"正算符值测度怎么定义"  → [0.11, -0.31, 0.85, 0.08, ...]   ← 非常接近
"今天中午吃什么"        → [0.67,  0.22, -0.10, 0.44, ...]  ← 差得很远
```

这是整个 RAG 的核心魔法。SQL 的 `WHERE` 只能找**字面相同**的东西；embedding 能找到**意思相同但用词不同**的东西。"POVM" 和 "正算符值测度" 字面毫无关系，但向量距离很近。

实现方式（[app/embedding.py](app/embedding.py)）抽象成一个接口：

```python
class BaseEmbedding:
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

任何模型只要实现这一个方法就能接入。本项目的三个实现，由 `.env` 的 `EMBEDDING_PROVIDER` 决定用哪个：

- `OpenAICompatibleEmbedding` —— 调真实的模型 API（DeepSeek、智谱、vLLM、Ollama 都行），**有语义理解**
- `LocalSemanticEmbedding` —— 本地加载 sentence-transformers 模型（本项目默认 `BAAI/bge-m3`），**有语义理解且完全离线**
- `HashingEmbedding` —— 本地离线的哈希伪向量，**没有语义理解**，只用来跑通代码链路和写测试

重要提醒：本项目实测过，哈希伪向量在 14303 个 chunk 的语料上，把 chunk 原文整段抄下来当查询，约 3/4 的情况自己都排不进前 5。所以它只能证明"代码能跑"，不能证明"检索准"。见 [app/embedding.py](app/embedding.py) 里 `HashingEmbedding` 的类注释。

---

## 5. 什么是向量数据库

向量数据库就是**专门用来快速找"最相近的向量"的存储**。

你可能想：为什么不用 SQLite 存向量，然后 SQL 里算距离？

理论上可以，但实际上不行。算一次相似度要乘加 1024 个数字，14303 个 chunk 全扫一遍是 1400 万次乘加，而且 SQLite 根本没有优化这种"多维空间最近邻搜索"的索引结构。你查 911 篇时还能忍，查 90 万篇时每句话要等几分钟。

向量数据库（ChromaDB）在里面做了专门的**索引优化**（本项目用的是 HNSW 算法 + 余弦距离），能把搜索压缩到毫秒级。它存的是**向量 + 原文 + 元信息**，查的时候返回的是原文和元信息，不是那一串数字。

---

## 6. 为什么需要 Chroma

三个具体原因：

1. **检索速度。** 如上，14303 个 256 维向量全量暴力扫一遍和用索引找，差的数量级不是 2 倍而是 100 倍。
2. **语义检索。** SQLite 的 `LIKE '%合同%'` 只匹配字面。Chroma 用余弦相似度匹配"意思"。
3. **零配置 + 本地持久化。** `PersistentClient(path="data/chroma")` 一行，数据写在本机，不需要装服务、不需要 Docker。

本项目的封装（[app/vectorstore.py](app/vectorstore.py)）很薄，只做三件事：写入 `upsert_chunks`、查询 `query`、按文档删除 `delete_by_document`。

有一个容易踩的坑：**Chroma 的 metadata 只接受标量**（字符串/数字/布尔），不接受列表。所以 `tags` 存进去之前要拼成逗号分隔字符串。

---

## 7. 什么是检索（Retrieval）

检索就是**根据你的问题，从知识库里找出最相关的几段材料**。

完整过程：

```
问题 "什么是 POVM？"
   │  用和文档完全相同的 embedding 模型
   ▼
问题向量 [0.12, -0.34, 0.87, ...]
   │
   ▼
在 ChromaDB 里算这个向量和所有 chunk 向量的余弦距离
   │
   ▼
按距离从小到大排序，取前 K 个
   │
   ▼
返回 [chunk 1, chunk 2, chunk 3]，每个带 title / file_path / heading / content
```

检索为什么单独做一步而不直接问 LLM？因为**LLM 的知识是训练时截止的、且是通用的**。它不知道你的笔记里写了什么，也不知道你笔记里的具体内容到底在哪一页。检索负责"告诉 LLM 该看哪几段"，LLM 负责"基于看到的材料组织答案"。

本项目的检索实现是 [app/retriever.py](app/retriever.py)，只做一件事：把向量距离换算成 0~1 的相似度（`similarity = 1 - cosine_distance`，截断到 [0,1]），然后返回结构化的 `SearchResult`。

一个独立的命令可以单独验证检索，不调 LLM：

```bash
python search.py "什么是 POVM？" --k 5
```

**这是排查 RAG 效果的第一站。** 答案不好时，先跑这个看检索找到了什么。检索没找到对的 chunk，LLM 再好也救不回来。

---

## 8. 什么是 Top-K

Top-K 就是**"返回最相关的前 K 个结果"**。K 是你要几个。

- `K=1`：只要最像的那一个。快，但相关答案分散在多篇笔记时会漏。
- `K=5`（本项目默认）：给模型 5 段材料。信息量够，噪声可控。
- `K=20`：几乎不会漏，但会塞进一堆勉强相关的噪声，模型可能被带偏，token 费用也高。

这就是 RAG 里最主要的"旋钮"。调优思路：

- 答案明显漏了关键信息 → 调大 K
- 答案里出现了不相关的内容 → 调小 K，或者检查切块是否太碎

本项目里 K 有三层来源：`.env` 的 `TOP_K=5`（默认）→ `/api/search` 和 `/api/chat` 的 `k` / `top_k` 参数（单次覆盖，1~20）→ `Retriever` 的 `default_top_k`。

---

## 9. 什么是 RAG

RAG = **Retrieval-Augmented Generation**，检索增强生成。

一句话：**先查资料，再让模型基于资料回答。**

对比三种做法：

| 做法 | 怎么做 | 问题 |
|---|---|---|
| 纯 LLM | 直接把问题丢给模型 | 不知道你的笔记，会一本正经地编 |
| 长上下文 | 把所有笔记塞进 prompt | 贵、慢、有长度上限、注意力被稀释 |
| **RAG** | 先检索最相关的几段，只把这几段给模型 | 可控、便宜、可溯源 |

本项目的 RAG 引擎在 [app/rag.py](app/rag.py)，它做的不只是"把检索结果拼进 prompt"，它严格处理了三件事：

1. **去重** —— `overlap` 会让同一个 chunk 的尾巴出现在相邻块里，按 `chunk_id` 去重，保证引用编号不重复。
2. **限制条数** —— 最多 `MAX_CITATIONS = 8` 条进 prompt，再多模型记不住编号。
3. **编号对齐** —— prompt 里 `[1] 文档：xxx` 的编号和 API 返回的 `sources[0].index` 严格一一对应。

Prompt 里明确写了几条硬约束：只能依据参考资料回答；信息不足就说"知识库中目前没有足够的信息"并说明缺什么；**绝对不要编造编号**——没有检索到来源的编号不会出现在 prompt 里，所以模型也没机会编。

检索不到任何结果时，系统返回固定的提示语，而不是让模型在空上下文里自由发挥（那几乎必然产生幻觉）。

---

## 10. FastAPI 在这个系统里负责什么

FastAPI 不是 RAG 的一部分，它是**把上面所有东西包成 HTTP 接口的那层外壳**。

没有它，你只能写 Python 脚本调用 `RAGEngine.answer()`。有了它，任何程序（网页、手机 App、Slack 机器人、另一个后端服务）都能通过 HTTP 请求你的知识库。

它负责四件事：

1. **定义路由** —— 哪个 URL 对应哪个函数（[app/main.py](app/main.py)）
2. **校验输入** —— `{"question": ""}` 会被自动拒绝（返回 422），不用你手写 if
3. **格式化输出** —— 把 Python 对象变成 JSON
4. **生成文档** —— `/docs` 页面是自动生成的，改代码就自动更新

### 可选的接口调用方式

**这个项目的日常入口是命令行**：`ask.py` 问答、`ingest.py` 导入。HTTP 接口是可选的，
不启动它项目照样能用，所以不要把它理解成「必须配一个前端才能用」。

如果确实想从别的程序调用（脚本、另一个后端、你自己的工具），可以直接发 HTTP 请求：

```javascript
// 先启动： uvicorn app.main:app
const res = await fetch("http://127.0.0.1:8000/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: "什么是 POVM？", top_k: 5 })
});
const { answer, sources } = await res.json();
// answer 里带 [1][2][3] 上标，sources 是和它对应的引用列表
```

`sources` 里每条有 `index`（和答案里的编号对应）、`title`、`file_path`、`heading`、`score`、`chunk`（原文）。
`file_path` 就是 Obsidian 里能定位到的路径，可以拿它做「点击引用跳到对应笔记」。

其余接口的用途：

- `GET /api/documents` —— 列出已导入的文档（分页）
- `GET /api/documents/{id}` —— 单篇详情
- `GET /api/search` —— 只检索、不调 LLM，排查检索效果用
- `POST /api/ingest` —— 等价于跑一次 `ingest.py`
- `GET /health` —— 探活

---

# 第二部分：完整调用链

## 场景：用户在 `/docs` 页面点了 Try it out，向 `POST /api/chat` 发送 `{"question": "什么是 POVM？"}`

### 第 1 步：请求到达哪个文件？

`http://127.0.0.1:8000/api/chat` → uvicorn 收到 HTTP 请求 → 按路由表找到 [app/main.py](app/main.py) 里的 `chat()` 函数（第 123 行）。

```python
@app.post("/api/chat")
def chat(request: ChatRequest):
    result = app.state.rag.answer(request.question, top_k=request.top_k)
    return {...}
```

FastAPI 在进入函数**之前**就用 pydantic 校验过 `request`：`question` 不能是空字符串（否则直接返回 422，函数体根本不会执行）。

`app.state.rag` 是启动时组装好的 `RAGEngine` 实例——依赖不在端点里现 new，而是在 `create_app()` 里一次性装好（[app/main.py](app/main.py) 第 26~68 行）。这也是测试能注入假数据的原因。

### 第 2 步：RAG 引擎做了什么

[app/rag.py](app/rag.py) 的 `RAGEngine.answer()`（第 67 行）：

```python
def answer(self, question, top_k=None):
    hits = self._retriever.search(question, top_k=top_k)      # ① 检索
    hits = _dedup_by_chunk_id(hits)[: MAX_CITATIONS]          # ② 去重 + 截断
    if not hits:
        return RAGAnswer(answer=固定提示语, sources=[])       # ③ 检索为空 → 拒绝生成
    context = _build_context(hits)                             # ④ 拼参考资料
    messages = [系统提示词, 用户提示词(含 context + question)]  # ⑤ 组装 prompt
    answer_text = self._llm.chat(messages)                     # ⑥ 调 LLM
    return RAGAnswer(answer=answer_text, sources=..., used_chunks=len(hits))
```

### 第 3 步：哪个函数生成 embedding？

[app/retriever.py](app/retriever.py) 的 `Retriever.search()`（第 37 行）第一行：

```python
question_vector = self._embedding.embed([question])[0]
```

`self._embedding` 是一个 `BaseEmbedding` 实例，具体是哪个类取决于 `.env` 的 `EMBEDDING_PROVIDER`，由工厂函数 [app/embedding.py](app/embedding.py) 的 `get_embedding_client()`（第 120 行）决定：

- `openai` → `OpenAICompatibleEmbedding.embed()`，它内部调 `client.embeddings.create(model=..., input=texts)`，把问题发给远程模型
- `local` → `HashingEmbedding._one()`，纯本地算，不发任何网络请求

两种情况下 `search()` 的代码**完全不用改**——这就是把 embedding 抽象成接口的收益。

### 第 4 步：Chroma 在哪里被查询？

还是 `Retriever.search()` 的下一行：

```python
rows = self._store.query(question_vector, n_results=min(k, self._store.count()))
```

落到 [app/vectorstore.py](app/vectorstore.py) 的 `ChromaStore.query()`（第 52 行），它调 `self._col.query(...)`。

这里有个刻意的设计：Chroma 原生的返回是 `result["ids"][0]` / `result["metadatas"][0]` 这种嵌套结构，`query()` 把它**摊平成 `[{content, distance, **metadata}]`**。上层只需要看一个普通列表，完全不需要知道 Chroma 的返回格式。

回到 `search()`，每条结果再被包成 `SearchResult`，其中距离换算成相似度：

```python
score = _to_similarity(row["distance"])   # 1 - cosine_distance，截断到 [0, 1]
```

### 第 5 步：SQLite 在哪里被读取？

**分两种情况，这是最容易搞混的地方。**

**问答链路里，SQLite 不参与。** 检索走 ChromaDB，因为相似度计算是 Chroma 的强项，SQLite 干不了。

**入库链路里，SQLite 是主存储。** `python ingest.py` 的流程（[app/ingest.py](app/ingest.py) 的 `_ingest_one()`，第 88 行）：

```python
doc = parse_markdown(path, ...)                 # 读 .md（只读 Vault）
document_id, changed, created = upsert_document(conn, doc)   # 写 documents 表
if not changed: return                          # 内容哈希没变 → 整篇跳过
chunks = chunk_markdown(doc.content, ...)       # 切块
replace_chunks(conn, document_id, chunks)       # 写 chunks 表
rows = _build_vector_rows(conn, document_id)    # ← SQLite 在这里被读回来
vectors = embed_documents(embedding, rows)
for row, vector in zip(rows, vectors):
    row["embedding"] = vector
store.delete_by_document(document_id)           # 先清掉旧向量
store.upsert_chunks(rows)                       # 再写新向量
```

注意倒数第 4 步：chunk 先写进 SQLite，然后**从 SQLite 读回来**再算向量写进 Chroma。`_build_vector_rows()` 从 `documents` 和 `chunks` 两张表拼出 Chroma 需要的那一行——标题、路径、章节、标签都是从 SQLite 取，不从内存里的 `ParsedDocument` 对象取。

为什么这么绕？因为还有另一条路径叫 `rebuild_vectors()`（`python ingest.py --rebuild-vectors`），它手头**没有**解析结果，只有 SQLite。两条路径共用 `_build_vector_rows()`，产出的行才完全一致。

好处是 SQLite 是唯一的原文存档，Chroma 是派生的索引——删掉 Chroma 完全无损，SQLite 永远是对的。

### 第 6 步：LLM 在哪里被调用？

回到 [app/rag.py](app/rag.py)，`answer()` 的最后：

```python
answer_text = self._llm.chat(messages)
```

`self._llm` 同样是工厂函数 [app/llm.py](app/llm.py) 的 `get_llm_client()`（第 66 行）按 `LLM_PROVIDER` 决定：

- `openai` → `OpenAICompatibleLLM.chat()`，调 `client.chat.completions.create(...)`，取 `choices[0].message.content`
- `mock` → `MockLLM.chat()`，不发请求，直接回显你拼好的 prompt

`MockLLM` 的用途很明确：没有 API Key 时也能验证"我拼的 prompt 对不对、引用编号有没有对齐"。你在 `/api/chat` 返回的 `answer` 字段里看到的就是实际会发给模型的内容。

### 第 7 步：API 怎么返回

[app/main.py](app/main.py) 的 `chat()` 把 `RAGAnswer` 转成 JSON：

```python
return {
    "answer": result.answer,
    "used_chunks": result.used_chunks,
    "sources": [{"index": s.index, "title": s.title, "file_path": s.file_path,
                 "heading": s.heading, "score": s.score, "chunk": s.content}
                for s in result.sources],
}
```

**关键不变式**：`answer` 文本里出现的每个上标编号，都能在 `sources` 里找到相同 `index` 的条目；反之 `sources` 里的每一条也都真实存在于检索结果中。这条不变式有测试守着（`tests/test_api.py::test_api_chat_returns_answer_and_citations` 断言 `used_chunks == len(sources)` 且 `index` 是 1..N 连续）。

### 完整链路一张图

入口有三个，都由 `app/engine.py` 装配出同一个 `RAGEngine`。下面这条画的是从 HTTP 走的路径：

```
浏览器打开 /docs 点 Try it out      ← Swagger 自动生成的接口调试页，不是聊天界面
  │  HTTP POST /api/chat  {"question": "什么是 POVM？"}
  ▼
app/main.py      chat()                          ← 路由 + pydantic 校验
  │
  ▼
app/rag.py       RAGEngine.answer()              ← 编排整个流程
  │
  ├─► app/retriever.py   Retriever.search()      ← 检索
  │      │
  │      ├─► app/embedding.py  embed([question])  ← 问题变成向量
  │      │       └─ OpenAICompatibleEmbedding（真模型）或 HashingEmbedding（离线）
  │      │
  │      ├─► app/vectorstore.py  ChromaStore.query()  ← 查 ChromaDB 拿 Top-K
  │      │
  │      └─ 换算 distance → similarity，返回 list[SearchResult]
  │
  ├─ _dedup_by_chunk_id() + 截断到 8 条          ← 保证引用编号不重复
  ├─ _build_context()                             ← 拼 [1] [2] [3] 参考资料
  │
  └─► app/llm.py       llm.chat(messages)        ← 生成答案
          └─ OpenAICompatibleLLM（真模型）或 MockLLM（回显 prompt）
  │
  ▼
{"answer", "used_chunks", "sources"}              ← RAGEngine 的返回值
  │
  ├─► ask.py         打印到终端：答案 + 引用列表
  ├─► mcp_server.py  交给 Claude Code，在对话里带引用回答
  └─► app/main.py    包成 JSON，HTTP 客户端自己解析
```

### SQLite 在哪里？（单独说明）

SQLite 在**入库时写**，在**问答时不读**。它是原文的权威存档：

```
app/ingest.py  run_ingest()          ← CLI 和 /api/ingest 的入口
  │
  ├─► app/parser.py      iter_markdown_files() + parse_markdown()   ← 只读 Vault
  │
  ├─► app/database.py    upsert_document()     ← 写 documents 表（含内容哈希）
  │
  ├─► app/chunker.py     chunk_markdown()      ← 切块
  │
  ├─► app/database.py    replace_chunks()      ← 写 chunks 表
  │
  ├─► app/database.py    get_chunks()          ← 读回来，拼向量行（_build_vector_rows）
  │
  ├─► app/ingest.py      embed_documents()     ← 行 → 文本
  │       └─► app/embedding.py   LocalSemanticEmbedding.embed()   ← 内部按 EMBEDDING_BATCH_SIZE 分批
  │
  └─► app/vectorstore.py delete_by_document() + upsert_chunks()   ← 重建 Chroma
```

`embed_documents()` 住在 `app/ingest.py` 而不是 `app/embedding.py`，因为它做的是**数据形状的转换**：把 SQLite 行（带 `content` 键的 dict）变成模型需要的纯字符串列表。这是入库流程独有的形状，不是模型封装的一部分。

分批**不在**这里，在 `LocalSemanticEmbedding.embed()` 内部。理由：`batch_size` 管的是 GPU 显存上限，只有 provider 知道自己装得下多少条；在这一层再切一遍是重复逻辑，而且省不了内存——返回的向量条数不管怎么切都一样多（14303 条向量约 20 MB，切与不切无关）。

`REBUILD_PAGE_SIZE = 512` 是另一件事：全量重建时**一次在内存里保留多少行的向量**，跟模型显存无关，是为了不一次性装下 500 MB+ 的中间结果。

---

# 第三部分：调试入口速查

| 现象 | 先看哪里 |
|---|---|
| 答案引用了错的来源 | `python search.py "你的问题"` —— 90% 是检索没找到对的内容 |
| 所有相似度都很低 | 是不是还在用 `EMBEDDING_PROVIDER=local`（离线伪向量，数字无意义） |
| 想改 chunk 大小 | `.env` 的 `CHUNK_SIZE` / `CHUNK_OVERLAP`，改完清空 `data/` 重新 ingest |
| 想换 embedding 模型 | 改 `.env`，然后 `python ingest.py --rebuild-vectors`（向量空间变了，必须重建） |
| 启动时报 `Embedding model mismatch` | 上条的另一种表现：`.env` 的模型和 `data/embedding_index.json` 里的指纹不一致，跑 `--rebuild-vectors` |
| 向量库是空的但 SQLite 有数据 | 指纹文件丢了或没写完，同样跑 `python ingest.py --rebuild-vectors` |
| 某篇文档导入失败 | `python ingest.py` 输出的 `失败 : N` 和 `errors` 列表 |
| 想确认引用编号没编 | 看 `tests/test_api.py::test_api_chat_returns_answer_and_citations` |
| 想确认没动 Vault | 看 `tests/test_parser.py::test_vault_is_left_untouched`（字节比对） |

---

# 第四部分：如果参加 AI 应用比赛，重点理解哪些代码

按重要度排序，前四个是核心竞争力：

**1. [app/chunker.py](app/chunker.py) —— 切块策略**
评委最容易问"你的切块怎么做的"。要能讲清分层逻辑（标题 → 段落 → 句子 → 贪心合并 → overlap）、为什么不按字符数硬切、为什么不跨章节合并、以及字符下标如何保证引用可追溯。这里的设计取舍直接决定检索质量和引用可信度，是最能体现工程判断的地方。

**2. [app/rag.py](app/rag.py) —— Prompt 工程 + 引用治理**
"不伪造引用"是 RAG 项目最容易翻车的点。要能讲清去重（overlap 导致重复）、截断（太多引用模型记不住编号）、编号对齐（prompt 里的 `[1]` 和 API 的 `sources[0].index` 一一对应）、以及检索为空时拒绝生成（避免空上下文幻觉）这四道防线。

**3. [app/embedding.py](app/embedding.py) + [app/llm.py](app/llm.py) —— 可替换性设计**
用一个抽象接口收口所有模型实现，换模型只改 `.env` 不改代码。要能讲清为什么不用 LangChain（想看清每一层的输入输出），以及离线 fallback 的能力边界（实测过、写在注释里、不吹）。

**4. [app/database.py](app/database.py) —— 增量入库**
"改了一篇文章会不会无限重复写入"是必问题。答案：内容 sha256 哈希做判据，相同则整篇跳过（不重新切块、不重新算向量）。要能讲清为什么用内容哈希而不是文件修改时间。

**5. [app/vectorstore.py](app/vectorstore.py) —— 对 Chroma 的封装**
重点是"为什么需要向量数据库"和那个 metadata 只接受标量的坑。

**6. 整体架构：SQLite 和 ChromaDB 为什么共存**
原文存档（SQLite，不可重建）+ 派生索引（Chroma，随时可重建）。这是被问到最多也最容易答错的问题——很多人会让两个库都存原文，导致不一致。

**最能加分的一句话总结**：
> 我们的检索质量由切块策略决定，我们的可信度由引用治理决定，我们的可维护性由 embedding/LLM 可替换设计决定，我们的数据一致性由 SQLite 作为唯一原文存档保证。向量库是可随时重建的派生物。

---

# 第五部分：本地 Embedding 到底发生了什么

这一章只用一个例子讲到底：**"曲柄滑块机构什么时候会出现死点"** 这句话，是怎么变成一个能被检索到的东西的。每个数字都是本机实测的，不是书上的。

## 先说结论

一个 embedding 模型做的事只有一件：**把任意一段文本变成一个点**。点在多维空间里，意思相近的文本点得近，意思相远的点得远。"POVM" 和 "正算符值测度" 一个汉字都不重合，但它们的点很近——这就是"语义"两个字真正的含义。

检索 = 把用户的问题也变成一个点，然后看它在 14303 个点里离谁最近。

## 第 1 步：文本先变成 token

模型不认识汉字，它只认识整数。所以在喂给模型之前，先经过 tokenizer。

本项目用的 `BAAI/bge-m3` 底层是 `XLMRobertaTokenizer`（BPE 分词），词表大小 **250002**。那句话实测被切成 **12 个 token**：

```
曲柄滑块机构什么时候会出现死点
→  <s>  ▁  曲  柄  滑  块  机构  什么时候  会出现  死  点  </s>
      0    6  9557  100315  25522  46775  11818  135375  113146  5680  2391  2
```

注意两点：

- `<s>` 和 `</s>` 是"句子开始/结束"标记，模型每次都带，不占语义。
- `机构` 和 `什么时候` 被当成**整体**一个 token，而 `曲柄滑块` 被拆成四个单字。BPE 是按词频学出来的：越常一起出现的字越可能被合并成一个 token。所以 token 数**不等于**字数——11 个汉字变 9 个语义 token。

这个 12 个数字之后就是模型的输入。

## 第 2 步：模型算出 1024 个数字

12 个 token 进模型，出来 **1024 个浮点数**。

`bge-m3` 是一个 XLM-RoBERTa 架构的模型：24 层 Transformer、16 个注意力头、**5.68 亿**参数，权重文件 `pytorch_model.bin` 实测 **2.17 GB**（fp32）。这些参数不是本项目算出来的，是 BAAI 拿海量语料训练出来的——里面存的是"什么样的句子组合通常是什么意思"。

本项目只负责把模型调起来，不负责训练。加载动作在 [app/embedding.py](app/embedding.py) 的 `_get_local_model()`：

```python
model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.float16})
model.eval()                          # 关掉 dropout：同一条文本永远编出同一个向量
model.max_seq_length = max_length     # 显存上限，见下面「上下文长度」
_MODEL_CACHE[key] = model
```

`_MODEL_CACHE` 是个字典，按 **四个值** 做键：`(模型名, 设备, 精度, max_length)`。**同一个进程里同一个组合只加载一次**——这是刻意的。加载要几十秒和 1 GB 显存，如果按 chunk 反复加载，14303 个 chunk 会慢到没法用。

`max_length` 为什么也得进键：`max_seq_length` 是写进 tokenizer 的**实例状态**。假设有两个客户端一个要 1024、一个要 2048，共用同一个实例时后创建的那个会静默改掉前一个的截断上限，入库的向量和指纹里记录的上限就对不上了。按上限拆开实例，就永远不会串。

另外注意 `model.eval()` 放在**加载处**而不是调用处：`encode()` 内部每次也会调 `self.eval()`（sentence-transformers 6.0.1 的 `model.py` 里 `encode()` 带 `@torch.inference_mode()` 装饰器，内部还有 `self.eval()`），所以这行不是必需的——它是把「这份模型只做推理、不许训练」这个不变量写在模型刚生出来、还没被任何调用碰过的位置，以后谁要改成训练用途第一眼就能看到。

设备怎么定的？`_resolve_device()`：

```python
return "cuda" if torch.cuda.is_available() else "cpu"
```

`.env` 里 `LOCAL_EMBEDDING_DEVICE=auto` 就是走这个分支。本机实测结果是 `cuda`（RTX 4050 Laptop，6 GB 显存）。启动时会打一行日志确认：

```
Embedding provider: local_semantic / Model: BAAI/bge-m3 / Device: cuda
```

想强制 CPU，把 `.env` 改成 `LOCAL_EMBEDDING_DEVICE=cpu`，它会真的用 CPU——不会因为"有 GPU"就偷偷换回去。

## 第 3 步：归一化（这一步最容易被忽略）

模型出来的 1024 个数字是**任意长度**的。归一化就是把向量除以自己的长度，让它变成单位长度（L2 范数 = 1）。

本项目在 [app/embedding.py](app/embedding.py) 的 `_encode()` 里做，文档侧和查询侧共用这一个方法：

```python
return self._model.encode(
    texts,
    normalize_embeddings=True,     # ← 关键
    show_progress_bar=False,
    convert_to_numpy=True,
)
```

**为什么要归一化？** 一句话：不归一化的余弦相似度计算量更大、数值更容易飘，而归一化之后"余弦相似度"退化成"两个向量直接相乘再求和"。

推导只有两行，值得手写一遍：

```
cosine(a, b) = (a · b) / (|a| × |b|)

当 |a| = |b| = 1 时：
cosine(a, b) = a · b
```

分母恒等于 1，直接消掉了。对本项目这意味着两件具体的事：

1. **计算更省**。Chroma 在 `hnsw:space: "cosine"` 下算距离，归一化之后底层就是点积，不用每次除长度。
2. **比较更稳**。不归一化时，一段 800 字的 chunk 和一段 20 字的 chunk 的向量长度天差地别，相似度会被"谁更长"污染。归一化之后长短不影响——只比方向。

本机实测归一化确实生效，但**偏差大小取决于 dtype**：

| dtype | L2 范数与 1 的最大偏差 |
|---|---|
| fp32 | **7.83e-08** |
| fp16（本项目 CUDA 上的实际配置） | **2.615e-04** |

这个偏差**不是归一化没生效**，是 fp16 自己的精度上限：fp16 只有 10 位尾数，归一化是逐元素除法，每个商都要被舍入一次。fp32 有 23 位尾数，舍入误差小三个数量级。对检索排序没有影响——fp16 归一化后向量的真实范数是 0.99997 而不是 1.00000000，比出来的余弦相似度只在小数点后第 4 位有差异。但如果你写「范数必须严格等于 1」的断言，得给 fp16 留容差（smoke test 里就是 `NORM_TOLERANCE = {"fp16": 2e-2, "fp32": 1e-5}`）。

`normalize_embeddings=True` 是 BGE 官方 README 推荐的用法，不是我们发明的。同族里的 `bge-large-zh-v1.5` 也一样推荐。

## 第 4 步：query 和 document 要不要区别对待

这是个常见的坑，值得单独说。

**`bge-m3` 的官方用法：文档和查询都不加前缀。** 所以本项目 `.env` 里 `LOCAL_EMBEDDING_QUERY_PREFIX=` 是空的。

但同族的 `bge-large-zh-v1.5` **要求查询加指令**（`为这个句子生成表示以用于检索相关文章：`），文档不加。因为它是通用编码器，需要一句话告诉它"这次是拿来检索的"；`bge-m3` 是专用检索模型，训练时就知道自己干嘛，不需要提醒。

代码怎么保证"前缀只加在查询侧，不污染文档"？靠 `BaseEmbedding` 把两个方法分开：

```python
class LocalSemanticEmbedding(BaseEmbedding):
    def embed(self, texts): ...              # 文档侧，永远不加前缀
    def embed_query(self, text):             # 查询侧，加不加前缀在这里决定
        query = f"{self._query_prefix}{text}" if self._query_prefix else text
        return self._encode(query).tolist()
```

`HashingEmbedding` 和 `OpenAICompatibleEmbedding` 用不上这个区别，直接继承基类默认的 `embed_query()`（就是 `embed([text])[0]`），一行代码都不用写。这个设计的好处是：**换模型时"查询/文档要不要区别对待"这个差异，只出现在需要它的那一个类里**。

测试守这条：`tests/test_embedding_semantic.py::test_query_prefix_only_applies_to_query_side` 断言 `fake_st[0].encoded == ["检索指令：死点", "文档正文"]` —— 文档侧那一条没被污染。

## 第 5 步：向量写进 Chroma

1024 个数字配着 chunk 原文和元信息，一起写进 ChromaDB。[app/vectorstore.py](app/vectorstore.py) 的 `upsert_chunks()`：

```python
self._col.upsert(
    ids=[str(r["chunk_id"]) for r in rows],
    embeddings=[r["embedding"] for r in rows],
    documents=[r["content"] for r in rows],
    metadatas=[_metadata(r) for r in rows],
)
```

collection 创建时固定了度量方式：

```python
self._col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
```

Chroma 只存**标量** metadata，所以 `tags` 是 list 的话必须 `",".join()` 成逗号串。这是这个库唯一的坑，`_metadata()` 里专门处理了。

写完之后还要写一份**指纹**到 `data/embedding_index.json`：

```json
{
  "model": "BAAI/bge-m3",
  "dimensions": 1024,
  "dtype": "fp16",
  "max_length": 1024
}
```

四个键都是刻意的——**改任何一个都等于换了向量空间**：换模型不用说；fp16 和 fp32 算出来的数字不同；长文本截到 1024 还是截到 2048 编码结果也不同。后两项如果不在指纹里，就会出现「改了参数但没重建」的静默不一致：服务看起来完全正常，只是相似度数字失去意义、Top-5 基本随机。

由 `write_index_info(embedding.fingerprint)` 完成，而且**只在向量全部落地、count 校验和自查询校验都通过之后才写**。半成品索引不会谎称自己是有效的。

## 第 6 步：用户提问时，问题走同样的路

用户输入"曲柄滑块机构什么时候会出现死点"，`/api/search` 调到 [app/retriever.py](app/retriever.py) 的 `Retriever.search()`：

```python
question_vector = self._embedding.embed_query(question)   # 问题也变成 1024 维
rows = self._store.query(question_vector, n_results=min(k, count))
```

**问题和文档必须用同一个模型、同一种处理方式编码**——这是 RAG 里最硬的约束。问题用 `bge-m3` 编码，文档用 `bge-small` 编码，两个向量根本不在同一个空间里，比出来的相似度是随机数。

所以检索之前有一道校验，`_check_index_matches()`：

```python
if info != embedding.fingerprint:
    raise EmbeddingMismatchError("Embedding model mismatch. ...")
```

读了 `data/embedding_index.json`，和当前 `.env` 算出的指纹比。**不一致就报错，绝不静默继续**。这是刻意的：昨天用 `bge-small`、今天改成 `bge-m3` 但忘了重建 Chroma，静默继续跑等于拿随机数当答案，你半天都查不出来为什么"检索突然变烂了"。宁可启动时炸掉。

## 第 7 步：算相似度，取前 5

Chroma 返回的是 `distance`（距离），本项目在 `_to_similarity()` 里换成直观分数：

```python
def _to_similarity(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(distance)))
```

`1 - cosine_distance`，截断到 `[0, 1]`。1.0 = 完全一样，0.0 = 完全无关。

为什么不用原始的余弦相似度（范围 -1 到 1）？因为对**已归一化**的向量，余弦相似度数学上落在 `[0, 1]` 区间（`1 - 0.5×欧氏距离²`），换算之后不用额外解释负数是什么意思。

拿到 Top-K 之后包成 `SearchResult`（`score, chunk_id, document_id, chunk_index, title, file_path, heading, source, content`），上层完全不需要知道 Chroma 返回的是嵌套 dict。

## 把 7 步串起来

```
"曲柄滑块机构什么时候会出现死点"
   │
   ▼  XLMRobertaTokenizer（词表 250002）
[0, 6, 9557, 100315, 25522, 46775, 11818, 135375, 113146, 5680, 2391, 2]   12 个 token
   │
   ▼  bge-m3（XLM-RoBERTa，24 层，5.68 亿参数，2.17 GB，cuda）
[0.023, -0.104, 0.057, ..., 0.088]                                        1024 个 float
   │
   ▼  normalize_embeddings=True（L2 范数 = 1；fp32 偏差 7.83e-08，fp16 2.6e-04）
   │
   ▼  同样这 1024 个数字，被拿去和 Chroma 里的 14303 个向量算余弦
   │
   ▼  hnsw:space = "cosine"，取最近 5 个
   │
   ▼  1 - distance，截断 [0,1]
SearchResult(score=0.71, title=..., heading=..., content=...)  × 5
   │
   ▼  Retriever 返回 list[SearchResult]
   │
   ▼  RAGEngine 去重 + 截断 + 拼 [1][2][3] + 调 LLM
```

## 那哈希伪向量为什么不行

`EMBEDDING_PROVIDER=local` 那个 `HashingEmbedding` 也在同一个抽象接口下，测试全靠它。但它的"语义能力约等于数相同单词的个数"：

- "POVM" 和 "正算符值测度" —— 没有共同 token，向量距离几乎随机
- 它的相似度数字**不能解释**，只能用来跑通链路

所以上面那张图里第 2 步换掉之后，整条链路的检索质量会断崖式下跌。这不是 bug，是它的设计边界——写在类注释里，测试也明确它是 fallback。

## 为什么换模型必须重建向量库

每个 embedding 模型都定义了一个**自己的**向量空间。`bge-m3` 的第 37 维和 `bge-small-zh-v1.5` 的第 37 维**没有任何关系**——就像摄氏度的 37 和华氏度的 37 不是一回事。

所以：`.env` 改了模型，Chroma 里的旧向量就必须全部作废重算。这是 `python ingest.py --rebuild-vectors` 存在的唯一理由。

它的设计纪律（[app/ingest.py](app/ingest.py) 的 `rebuild_vectors()`）：

- **只读 SQLite**，chunk 原文一个字都没变，不需要重新解析 911 个 Markdown
- **只写 `data/chroma_building/`、`data/chroma/` 和 `data/embedding_index.json`**，绝不碰 `data/knowledge.db`
- **两阶段**：向量先写进临时目录 `data/chroma_building/`，正式索引全程不动
- 校验 `count == 14303`，再校验一条真实查询能命中自己——count 对了但 HNSW 建坏时 count 照样对，所以必须真的发一条查询
- 校验通过才替换正式目录：关闭临时句柄 → 删掉旧 `data/chroma/` → 把 `chroma_building/` 改名成 `chroma/`
- 全部写完并校验通过之后才写指纹，写指纹是「这个索引可用」的唯一信号
- 分页 `REBUILD_PAGE_SIZE = 512`：每条向量 1024 个 float 约 40 KB，全量一次装完要 500 MB+
- 超过 `LOCAL_EMBEDDING_MAX_LENGTH` 的 chunk 逐条打 warning（`chunk_id / document / token_length`），结束时汇总数量——**不静默截断**
- 进度只打一行 `Embedding: 3200 / 14303`，不做复杂 UI

早期版本是原地 `store.clear()` 再重写。中途一旦失败就留下一个只有 2048 / 14303 条的半成品——`search.py` 看起来还能跑，只是结果明显少了。改成两阶段后，**半成品永远不会冒充正式索引**。

**SQLite 是原始存档，Chroma 是派生索引。**删掉 Chroma 完全无损，SQLite 永远是对的。反过来删掉 SQLite 就没救了。

## 本项目的实测数据

| 项目 | 实测值 |
|---|---|
| 项目 | 实测值 |
|---|---|
| 模型 | `BAAI/bge-m3`（XLM-RoBERTa，5.68 亿参数） |
| 输出维度 | 1024 |
| 模型自带上限 | 8192 token |
| **本项目强制上限** | **1024 token**（`LOCAL_EMBEDDING_MAX_LENGTH`，为显存留余量） |
| 权重体积 | 2.17 GB（fp32）→ 1.1 GB（fp16，CUDA 上的实际精度） |
| 缓存位置 | `%USERPROFILE%\.cache\huggingface\hub\models--BAAI--bge-m3\` |
| 设备 | `cuda`（RTX 4050 Laptop，6 GB） |
| 加载耗时 | 40~68 秒 / 每个新进程（实测 61.0 / 39.6 / 42.2 / 68.2 秒） |
| 编码速度 | 38.6 ms/条（近满长度压力批，batch=8）；全库重建 0.03 秒/条（含加载） |
| 归一化偏差 | L2 范数与 1 相差 7.83e-08（fp32）/ 2.615e-04（fp16） |
| 同文本重复编码 | 最大绝对元素差 **0.000e+00** |
| 向量库规模 | 14303 个 chunk |
| 全量重建耗时 | 364.2 秒，峰值显存 1275 MiB |

模型自带 8192，为什么本项目只给 1024？`encode()` 会把一整批 padding 到批内最长那条，注意力开销按 `batch × heads × seq²` 增长。允许 8192 的话，全库里最长的那条 chunk（3654 token）就能单独吃掉 6 GB 显存的全部。封顶到 1024 之后，只有 12 / 14303 条会被截尾，且逐条打 warning。

**为什么两次编码同一句话的差是 0.000e+00，而不是文档早先写的 2e-07？**

上面这句「两次编码漂移最大 2.01e-07，是 CUDA 浮点归约顺序不同」**是猜的，没测过**，本轮专门跑了诊断把它推翻。真实情况是：`encode()` 内部先 `self.to(device)` 再 `self.eval()`（sentence-transformers 6.0.1 的 `model.py`，`encode()` 还带 `@torch.inference_mode()` 装饰器），dropout 从加载完就关着。同一条文本重复编码时 `tokenize → padding → forward → pooling → normalize` 每一步的输入完全一样，输出自然逐位相同。`scripts/diagnose_embedding_determinism.py` 三组测试都得 0.000e+00：

- A：一次 `encode(["x", "x"])`，批内两条完全相同文本
- B：同一文本连续两次独立 `encode`
- C：`batch_size=1` 连续 5 次

所以最初那个「同一文本 fp16 两次 cosine = 0.9878」的告警**不是非确定性**，是 fp16 归一化的舍入——见上面「归一化」那节。真正会让同一条文本编出不同向量的只有换模型、换 dtype、换 `max_length` 三件事，前两项和后一项都在指纹里。

## 为什么 mock 测试全绿、真实模型仍然会失败

本项目 85 个测试全绿。然后第一次接真实模型时，第一版实现**跑不起来**：`ValueError: encode() got an unexpected keyword argument 'max_length'`。两件事同时为真，不矛盾——因为 mock 和真实库对参数的契约根本不同。

```python
# tests/test_embedding_semantic.py：用 monkeypatch 塞进 sys.modules 的假模块
class FakeModel:
    def __init__(self, *args, **kwargs): ...     # 接受任何构造参数
    def encode(self, texts, **kwargs):           # 也接受任何编码参数
        return [[0.0] * 1024 for _ in texts]
```

```python
# sentence-transformers 6.0.1 的 encode()：对参数做白名单校验
def encode(self, sentences, prompt_name=None, prompt=None, batch_size=32,
           show_progress_bar=None, output_value="sentence_embedding", ...):
    # 不认识的 extra kwargs 直接抛 ValueError
```

第一版实现图省事，把 `max_length=1024` 直接传进 `encode()`。mock 的 `**kwargs` 会**默默吞掉**这个参数——于是「传错了参数」在测试里看起来和「传对了参数」一模一样，85 个测试一路绿。换成真实的 6.0.1 立刻炸：`encode()` 对参数做白名单校验，`max_length` 不在名单里。

这类错误**永远测不出来**。mock 的职责是验我们自己的逻辑（分批、缓存、维度探测、前缀、指纹校验），它不承担「下游库的 API 长什么样」这件事——而后者只有真调一次真库才知道。

所以真实模型验证放在三个独立脚本里，**不进默认 pytest**：

```bash
python scripts/diagnose_embedding_determinism.py   # 同一文本重复编码是否可复现
python scripts/smoke_real_embedding.py             # 维度、归一化、NaN、语义区分度
python scripts/retrieval_check.py                  # 300 个真实 chunk，真实 Top-K
```

不进 pytest 的理由很具体：真实 bge-m3 加载要 40~68 秒，而且首次需要联网下载 2.17 GB 权重。塞进 pytest 的话，每次开发循环都要多等一分钟以上。真实模型的验证是低频事件——换模型、改 dtype、改 max_length 时才需要重跑——它的成本不该占用每次 `pytest` 的时间。

**代价要明确说**：真实模型的正确性现在靠手动跑这三个脚本保证，CI 里没人替你跑。如果以后要进 CI，得把这三个脚本挂在「换 embedding 配置时才触发」的独立 job 上，不能和 pytest 一起跑。

## 想亲手验证的时候

- **只看 tokenizer**：`AutoTokenizer.from_pretrained("BAAI/bge-m3")`，约 16 MB，不占显存
- **看指纹**：`type data\embedding_index.json`
- **只看检索不调 LLM**：`python search.py "你的问题"` —— 90% 的"AI 答错了"其实是检索没找对
- **换模型**：改 `.env` → `python ingest.py --rebuild-vectors`，指纹不一致时启动会直接报错
