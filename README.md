# Obsidian RAG Backend

把你自己的 Obsidian 笔记变成可以提问的知识库：`Obsidian → 文档解析 → SQLite → 切块 Chunk → Embedding → ChromaDB → 检索 → LLM 回答 → FastAPI 接口`。

**这是只读项目**：它只读取你的 Vault，绝不写入、移动或删除里面的任何文件。有一条测试专门用字节比对保证这一点（`tests/test_parser.py::test_vault_is_left_untouched`）。

> 想先搞清楚每个概念是什么意思，先读 [LEARNING_NOTES.md](LEARNING_NOTES.md)。

---

## 1. 这个项目是什么

三条入口，底层共用同一个 `RAGEngine`（装配代码集中在 `app/engine.py`，三份入口不再有各自的拷贝）：

- **`ask.py` 命令行**——日常用法，在终端里直接问自己的笔记，返回带 `[1][2][3]` 引用来源的回答；
- **`mcp_server.py` MCP 服务器**——让 Claude Code 在对话里直接检索你的知识库，见第 12 节；
- **FastAPI HTTP 接口**——供需要程序化调用的脚本或工具使用。

只想要 RAG 动作的话，命令行那条路就够了，不用起服务。

```
你的 Obsidian Vault（只读）
        │  解析 .md
        ▼
   SQLite  ←———————— 原文 + 元信息，人可以直接查
        │  切块
        ▼
   ChromaDB ←—— 每个 chunk 的向量，负责"按意思找相近内容"
        │  检索
        ▼
   RAG 引擎 ←—— 拼 Prompt
        │  生成
        ▼
   FastAPI 接口 ←—— /api/chat 返回答案 + 引用来源
```

两个存储库各司其职，互不替代：

| 库 | 存什么 | 用来干嘛 |
|---|---|---|
| SQLite | 原文全文、标题、路径、标签、切块原文 | 精确查询、翻页、查文档列表、备份 |
| ChromaDB | 每个 chunk 的向量 | 相似性检索（"意思相近"而不是"字面相同"） |

ChromaDB 是可以随时重建的派生物，SQLite 是唯一的原文来源。删掉 `data/chroma` 重来没有任何损失。

---

## 2. 数据流（一次问答实际发生了什么）

```
POST /api/chat  {"question": "什么是 POVM？"}
    │
    ├─ 1. retriever.search()      把问题变成向量
    ├─ 2. ChromaDB 查询           找最相近的 Top-K 个 chunk（含 title/file_path/heading）
    ├─ 3. rag 拼 Prompt           参考资料 + 引用编号 [1][2][3] + 用户问题
    ├─ 4. llm.chat()              调用模型生成答案
    └─ 5. 返回                    {answer, used_chunks, sources[]}
                                   sources 的 index 和答案里的 [1][2][3] 一一对应
```

SQLite 在这条链路上**不直接参与**（检索走 ChromaDB）。它的角色是：原文的唯一存档、增量更新的判据、以及查文档列表用。

---

## 3. 目录结构

```
obsidian-rag-backend/
├── app/                  核心代码，一个文件一个职责
│   ├── config.py         读 .env，所有配置的唯一入口
│   ├── parser.py         读 Vault 里的 .md（只读）
│   ├── chunker.py        中文友好的切块逻辑 ★最值得读
│   ├── database.py       SQLite 建表 + 增量写入
│   ├── embedding.py      Embedding 客户端（可替换）
│   ├── vectorstore.py    ChromaDB 封装
│   ├── retriever.py      检索：问题 → Top-K chunk
│   ├── router.py         知识域路由：先判断属于哪个域，再限定检索范围
│   ├── knowledge_domains.py 读域映射表和路由术语表（都在 config/ 下的 YAML）
│   ├── context_filter.py 过滤导航类 chunk（默认关闭）
│   ├── reranker.py       cross-encoder 精排（默认关闭）
│   ├── llm.py            LLM 客户端（可替换）
│   ├── evidence.py       证据门控：回答前判定资料够不够
│   ├── personal_context.py 读个人上下文，供 advisor 模式注入
│   ├── rag.py            拼 Prompt + 管引用编号 ★最值得读
│   ├── engine.py         装配 RAGEngine：CLI / HTTP / MCP 三个入口共用
│   ├── ingest.py         把上面几步串起来，含删除同步
│   └── main.py           FastAPI 路由
├── config/
│   └── knowledge_domains.yaml 知识域映射 + router_terms 路由术语（改归类只动这里）
├── personal_context/     context.yaml 是你的私有上下文（已 gitignore，不进库）
│                         example.yaml 是同构模板，进库
├── ingest.py             命令行：导入文档 / 增量更新 ← 构建完知识库后跑这条
├── search.py             命令行：只检索、不调 LLM（排查检索用）
├── ask.py                命令行：完整问答 ← 日常入口
├── mcp_server.py         MCP 服务器：让 Claude Code 在对话里直接检索
├── eval_retrieval.py     检索评测：Recall@K / MRR
├── eval_rag.py           端到端问答评测
├── eval/                 题目集与评测结果（JSON）
├── RETRIEVAL_BASELINE.md 优化前的成绩基线 + reranker A/B 实测
├── scripts/              验证与测量脚本，都不属于默认 pytest
│   ├── diagnose_embedding_determinism.py   Embedding 可复现性诊断（A/B/C/D/E 五组）
│   ├── smoke_real_embedding.py             真实模型 smoke test
│   ├── retrieval_check.py                  小规模真实检索验证（300 chunk 临时索引）
│   ├── probe_llm_latency.py                重新测量上游 LLM 延迟（决定 timeout 用）
│   └── probe_* / trace_* / download_*      其他测量脚本，见目录
├── tests/                298 个测试（全 mock，不加载模型）
├── .env                  你的配置（不要提交到 git）
├── .env.example          配置模板
├── data/                 运行时生成：knowledge.db + chroma/（已 gitignore）
├── README.md
└── LEARNING_NOTES.md     概念讲解 + 完整调用链
```

桌面启动器 `ask.bat` 放在 `C:\Users\<你的用户名>\Desktop\`，双击即用（它写死了项目路径，
可以放在任何位置运行）。

---

## 4. 安装依赖

Python 3.10+。

```bash
cd obsidian-rag-backend
python -m venv .venv
.venv/Scripts/activate        # Windows；macOS/Linux 用 .venv/bin/activate
pip install -r requirements.txt
```

`sentence-transformers` 会自动拉进 PyTorch。**这里有个容易踩的坑**：PyPI 默认装的是 **CPU 版** torch，装了之后 `torch.cuda.is_available()` 是 `False`，你以为在用 GPU 其实一直在跑 CPU。

有 NVIDIA 显卡想跑 CUDA，装完再额外装一次官方 CUDA 版：

```bash
# Windows / Linux，按你的驱动支持的 CUDA 大版本选（驱动 550+ 用 cu121，560+ 可用 cu124 / cu130）
pip install --force-reinstall --no-deps "torch==2.14.0" \
    --index-url https://download.pytorch.org/whl/cu130
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

必须看到 `True` 和显卡型号。`--force-reinstall` 不能省：已装的 `2.14.0+cpu` 在版本比较上**等于** `2.14.0`（PEP 440 忽略 local 版本号），普通 `pip install torch==2.14.0` 会显示 already satisfied 然后什么都不做。

**没有显卡也没关系**，CPU 能跑，只是重建全库慢（见第 8 节）。

---

## 5. 配置 `.env`

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
```

然后编辑 `.env`，重点两个：

```ini
# 你的 Vault 绝对路径（反斜杠改成正斜杠）
VAULT_PATH=C:/Users/你的用户名/Documents/Obsidian Vault

# 想用真实的语义检索和真实回答，就改成 openai
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.deepseek.com/v1
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_MODEL=deepseek-embedding-2

LLM_PROVIDER=openai
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-v4-flash
```

只要 `BASE_URL` / `API_KEY` / `MODEL` 这三个变量，任何 OpenAI 兼容的服务都行（DeepSeek、智谱、通义、vLLM、Ollama）。

`EMBEDDING_PROVIDER` 有三种取值：

| 值 | 是什么 | 什么时候用 |
|---|---|---|
| `local` | 本地特征哈希伪向量，纯 Python 算，零依赖 | 只想跑通链路和写测试。**没有语义能力**，不能用来评估检索效果 |
| `local_semantic` | 本地 sentence-transformers 模型，真语义，完全离线 | **推荐**。不需要 Key、不花钱、数据不出机器 |
| `openai` | 任意 OpenAI 兼容的 embeddings 接口 | 想用云端更强的模型，或本机算力不够 |

**默认是离线模式**（`EMBEDDING_PROVIDER=local`、`LLM_PROVIDER=mock`），不需要 Key 也能跑通全部代码。但**它不能验证检索效果**，只能验证"代码能不能跑"。详见下面第 8 节。

本项目 `.env` 用的是本地语义模型：

```ini
EMBEDDING_PROVIDER=local_semantic
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3
LOCAL_EMBEDDING_DEVICE=auto
LOCAL_EMBEDDING_QUERY_PREFIX=
LOCAL_EMBEDDING_MAX_LENGTH=1024     # 超过就截断，并在 rebuild 日志里逐条 warning
EMBEDDING_BATCH_SIZE=8              # 实测取值，见第 8b 节
```

模型怎么换、显存不够怎么办、换模型后旧向量怎么处理，见第 8b 节。

---

## 6. 配置 Vault 路径

改 `.env` 里的 `VAULT_PATH` 即可。路径不存在时服务不会崩，`/health` 依然正常，只是 `POST /api/ingest` 会返回 400 告诉你配错了。

自动忽略：`.obsidian/`、`.trash/`、`.git/`、`__pycache__/`，以及所有以 `.` 开头的目录、图片和附件、非 `.md` 文件。

---

## 7. 导入文档

```bash
python ingest.py                  # 全库导入
python ingest.py --limit 20       # 先试导 20 篇，确认没问题再全量
python ingest.py --path "法律"    # 只导入路径里包含"法律"的文件
```

也支持通过接口触发：

```bash
curl -X POST http://127.0.0.1:8000/api/ingest -H "Content-Type: application/json" -d "{\"limit\": 20}"
```

### 增量更新

典型用法是：**知识体系构建完成 → 跑一次 `python ingest.py`**。它自动扫描差异，
不需要常驻后台 watcher，也不需要手动判断哪些文件变了。四种变化分别这样处理：

| 情况 | SQLite | Chroma 向量 | 统计字段 |
|---|---|---|---|
| 新文档 | 插入 `documents` + `chunks` | 插入新 chunk 向量 | `added` |
| 修改文档（含改短） | 先删该文档旧 `chunks`，再插新的 | 先删该文档旧向量，再插新的 | `updated` |
| 删除文档 | 删 `documents` 行，`chunks` 靠 `ON DELETE CASCADE` 连带删 | 删该文档全部向量 | `removed` |
| 未变化 | 整篇跳过 | 整篇跳过 | `unchanged` |

**"删除同步"是必须做的一步**，否则会在库里留下「幽灵知识」：笔记在 Obsidian 里已经删了，
但向量还在，问相关问题时照样被检索出来，回答看起来言之凿凿其实来源已不存在。
删除顺序是**先删 Chroma 向量、再删 SQLite 行**——这样万一 Chroma 那步失败，SQLite 还没动，
下次再跑就能干净重试；单篇失败只计入 `failed` / `errors`，不会中断整批。

**两个刻意设计的保护：**

- `--path` / `--limit` 只扫了一部分文件，这时候**无法**判断哪些是「被删了」——
  没扫到的文件不是被删，只是没看到。所以这两种模式下删除同步自动跳过，
  并会打印提示。要全库同步，跑一次不带参数的 `python ingest.py`。
- 文件读不到（权限、临时锁定）不等于文件被删了。读失败的文件不会进入「已见」集合，
  因此不会被判成删除，它的知识不会凭空消失。

**正文被清空**（只剩 frontmatter）也算一种修改：切块后一个 chunk 都没有，
这种情况下旧的 chunk 和向量一样要清空，而不是保留。

```
第一次：扫描 911  新增 911  未变化跳过 0    写入 chunk 14303
第二次：扫描 911  新增 0    未变化跳过 911  写入 chunk 0
```

判据是内容 sha256，不是文件修改时间——只改了一行也有哈希变化，会被正确更新；
反复跑同一批文件不会产生重复插入。

### 重新测量上游 LLM 延迟

LLM 回答慢的时候，先量再决定，别改配置猜：

```bash
python scripts/probe_llm_latency.py
```

它会对比「空 prompt」和「RAG 同量级 prompt」两组，帮你区分瓶颈是服务端的固定等待，
还是输入体量大导致的。

上游 LLM 延迟受服务负载、网络和时段影响较大，历史单次测量已经证明不稳定，
所以 `.env` 里的 `LLM_TIMEOUT_CONNECT` / `LLM_TIMEOUT_TOTAL` 保持可配置、本轮未改值。
**建议在多个时段各跑一次**，拿到一组有代表性的数字之后，再决定要不要调整这两个值。

---

## 8. 测试检索效果（不调用 LLM）

```bash
python search.py "什么是 POVM？"
python search.py "演化稳定策略" --k 10
```

输出 Top-K，每条带相似度、文档名、章节名和内容预览。这一步**不用花钱、不用调模型**，是排查"为什么答案不好"的第一站：检索没找到对的 chunk，后面 LLM 再好也救不回来。

### ⚠️ 关于离线模式的重要说明

`EMBEDDING_PROVIDER=local` 用的是本地哈希伪向量，**它不具备语义理解**。实测在 14303 个 chunk 的语料上，把 chunk 里的原文整段抄下来当查询，约有 3/4 的情况自己都排不进前 5。

根因是余弦相似度的长度稀释：查询十几个 token、chunk 四五百个 token，`cos = 共享词数 / √(查询词数 × chunk 词数)`，量级只有 0.05，随便几个共同高频词就能压过真实匹配。这是哈希指纹的必然结果，换切词方式修不好。

所以离线模式只能证明"ingest → ChromaDB → 检索 → API 全链路代码是对的"，**不能证明检索效果**。想看到真实检索质量，必须配一个真正的 embedding 模型。

> **换 embedding 模型后必须重建向量库**：不同模型输出的向量不在同一个空间里，混在一起相似度数字就完全失去意义。不要手动 `rm -rf data/chroma`，用第 8b 节的重建命令——它只动 Chroma，不碰 SQLite。

---

## 8b. 本地 Embedding 模型

### 这台项目用的是哪个模型

`BAAI/bge-m3`，参数如下：

| 项 | 值 |
|---|---|
| 架构 | XLM-RoBERTa（多语言 RoBERTa） |
| 参数 | 约 5.68 亿 |
| 输出维度 | **1024** |
| 上下文上限 | **8192 token** |
| 权重体积 | 约 2.17 GB（fp32，`pytorch_model.bin`） |

为什么选它而不是 bge-large-zh-v1.5（3.26 亿参数、只有 512 维）：bge-large-zh 的上下文上限是 **512 token**，而本项目 `CHUNK_SIZE=800` 中文字符 ≈ 800~1000 token——**每个 chunk 都会被削掉一半**。你以为在编码整段，其实模型只看了开头一小截。bge-m3 的 8192 token 上限能完整装下一个 chunk。

### 上下文长度与超长 chunk

模型上限是 8192，但本项目把 `LOCAL_EMBEDDING_MAX_LENGTH` 设成 **1024**，远低于上限。原因不是精度，是显存：`encode()` 会把一整批 padding 到批内最长那条，注意力开销按 `batch × heads × seq²` 增长，允许 8192 就会让偶发的长 chunk 吃掉全部显存。

实测全库 14303 个 chunk 的 token 分布（中文约 0.71 token/字符）：

| 分位 | token 数 |
|---|---|
| p50 | 304 |
| p99 | 780 |
| max | 3654 |
| 超过 1024 的条数 | **12** |

这 12 条会被截断到 1024 token，但**不是静默截断**：`--rebuild-vectors` 会逐条打印 `chunk_id / document / token_length`，并在阶段一结束前给一个汇总。想彻底避免得改切块器按 token 而非字符切，列在「后续优化」里。

`LOCAL_EMBEDDING_MAX_LENGTH` 也写进模型指纹——改了它就等于换了向量空间，Retriever 会拒绝用旧库。

### 第一次运行会自动下载

第一次用到 `local_semantic` 时会联网下载模型（约 2.17 GB），之后走本地缓存，完全离线。

**缓存命中不等于加载快**：新进程每次仍要读 391 个权重文件、初始化 tokenizer 并分配显存，本机实测 **40~68 秒**（四次独立测量 61.0 / 39.6 / 42.2 / 68.2 秒，`Loading weights` 那一步本身只占不到 1 秒，时间花在模型对象构造和 CUDA 初始化上）。所以全量重建的总耗时里，模型加载大约占 40 秒量级。

模型缓存位置：

```
%USERPROFILE%\.cache\huggingface\hub\models--BAAI--bge-m3\
```

下载中断了没关系，重新跑会从断点续传。网络环境不好时可以先设镜像：

```bash
$env:HF_ENDPOINT="https://hf-mirror.com"     # PowerShell
# export HF_ENDPOINT="https://hf-mirror.com"  # macOS/Linux
```

国内网络下 `huggingface.co` 经常不通，镜像站通常能直接解决。

### CPU 还是 GPU

`.env` 里 `LOCAL_EMBEDDING_DEVICE=auto`：有可用 CUDA 就用 GPU，否则自动用 CPU。

启动时会打印实际用的设备，务必看一眼：

```
Embedding provider: local_semantic / Model: BAAI/bge-m3 / Device: cuda
```

想强制指定就改成 `cpu` 或 `cuda`。

**显存不够会怎样？** 模型权重 fp16 下约 1.1 GB（fp32 2.17 GB）。bge-m3 是纯 Transformer，推理时不用留反向传播的激活，6 GB 显存放得下模型本体。

真正吃显存的是 `EMBEDDING_BATCH_SIZE`。`encode()` 会把一整批 padding 到**批内最长**那条，注意力开销按 `batch × heads × seq² × dtype` 增长，所以一批长文本比十批短文本危险得多。实测（RTX 4050 Laptop 6 GB / fp16 / max_length=1024 / 一批接近 1024 token 的真实 chunk，即全库最长的 16 条）：

| batch | 峰值 allocated | 峰值 reserved | 耗时 | 单条 |
|---|---|---|---|---|
| 2 | 1137 MiB | 1162 MiB | 0.09s | 43.5 ms |
| 4 | 1183 MiB | 1226 MiB | 0.16s | 39.4 ms |
| **8（默认）** | **1275 MiB** | **1354 MiB** | **0.31s** | **38.6 ms** |
| 16 | 1459 MiB | 1610 MiB | 0.60s | 37.5 ms |

`batch=16` 比 `batch=8` 只多占 256 MiB，但单条只快 2.9%，所以默认取 **8**——离 6 GB 上限留了四五倍余量。反例：fp32 + batch=32 在全量重建跑到 2048 / 14303 时 CUDA OOM 挂死，那个组合不要用。

注意 batch 变大时单条速度几乎不涨（43.5 → 37.5 ms）：压力测试里每条都是近 1024 token 的长文本，每批都被 pad 到最长那条，算力被最长那条决定，batch 只影响显存。**这是本项目没有做 dynamic batching（按 token 长度分桶）的原因**——收益上限就是这几个百分点，不值得为此加复杂度。

显存紧张就把 `EMBEDDING_BATCH_SIZE` 调更小，或者 `LOCAL_EMBEDDING_DEVICE=cpu`。

另外留意 fp16：本项目在 CUDA 上默认用 fp16（省一半显存和带宽）。代价是归一化后向量的 L2 范数不再是精确的 1，实测最大偏差 **2.615e-04**，fp32 下是 7.83e-08。对检索排序没有影响，但如果你写「范数必须严格等于 1」的断言，得给 fp16 留容差。

内存（不是显存）也要留意：1024 维的向量在 Python 里是 1024 个 float 对象，一个 chunk 约 40 KB，14303 个 chunk 一次装完要 500 MB 以上。本项目的 `rebuild_vectors` 是按 512 条一页分批处理的，一页约 20 MB，就是为这个问题准备的。

### 怎么换模型

只改 `.env`，不动代码：

```ini
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

换完必须重建向量库（见下面）。

| 模型 | 参数 | 维度 | 上下文 | 说明 |
|---|---|---|---|---|
| `BAAI/bge-m3` | 568M | 1024 | 8192 | 本项目默认，中英文混排最优 |
| `BAAI/bge-large-zh-v1.5` | 326M | 512 | **512** | 纯中文强，但装不下本项目的 chunk |
| `BAAI/bge-small-zh-v1.5` | 33M | 512 | **512** | 内存/显存很紧张时的降级选项 |

后两者都要把 `CHUNK_SIZE` 降到 300 以下才不截断。

**关于查询前缀**：`bge-m3` 官方要求查询和文档**都不加**前缀，所以 `LOCAL_EMBEDDING_QUERY_PREFIX` 默认留空。但 `bge-large-zh-v1.5` 要求查询侧加指令 `为这个句子生成表示以用于检索相关文章：`，换到那个模型时要填进去。文档侧永远不能加前缀——加了就把索引里的原文污染了。

### 重建向量库（换了模型必做）

```bash
python ingest.py --rebuild-vectors
```

它会：**分两个阶段**，正式索引全程不动，直到新索引验证通过。

```
阶段一（构建）      从 SQLite 读 14303 个 chunk
                     → 算向量
                     → 写入 data/chroma_building/   ← 临时目录
                     → 校验 count == 14303
                     → 校验一条真实查询能命中自己
                        │
                        ├─ 失败：删除 chroma_building/，正式 data/chroma/ 原样保留
                        ▼ 成功
阶段二（替换）      关闭临时索引句柄 → 删掉旧的 data/chroma/
                     → 把 chroma_building/ 改名为 chroma/
                     → 最后才写 data/embedding_index.json
```

为什么要临时目录？早期版本是原地 `store.clear()` 再重写，中途失败就留下一个只有 2048 / 14303 条的半成品索引——`search.py` 看起来还能跑，只是结果明显少了。改成两阶段后，**半成品永远不会冒充正式索引**：写指纹这一步只在 count 校验和真实查询校验都通过之后发生。

**只动 Chroma，绝不碰 SQLite。** SQLite 是唯一的原文存档，`knowledge.db` 一字不改。也不需要重新解析那 911 个 Markdown 文件——chunk 原文没变，没必要再切一遍。

进度按页打印：`Embedding: 512 / 14303` … `14303 / 14303`。超过 `LOCAL_EMBEDDING_MAX_LENGTH` 的 chunk 会逐条打 warning。

跑完会打印本次重建用的模型、设备、维度、batch size、耗时和显存峰值。

本机实测全量重建一次：**364.2 秒（约 6 分 4 秒）**，14303 条，平均 0.03 秒/条，峰值显存 1275 MiB。配置为 RTX 4050 Laptop 6 GB / fp16 / batch=8 / max_length=1024，含约 40 秒模型加载。重建结束时 12 条超长 chunk 被截断到 1024 token（逐条 warning 见上），数量校验 `14303 / 14303` 与自查询校验都通过。

### 模型指纹：防止"昨天 bge-small、今天 bge-m3"却留着旧库

建索引时会在 `data/embedding_index.json` 写下模型指纹：

```json
{
  "model": "BAAI/bge-m3",
  "dimensions": 1024,
  "dtype": "fp16",
  "max_length": 1024
}
```

四个键都是刻意的。除了换模型，**改 `dtype` 或 `max_length` 也等于换了向量空间**：fp16 和 fp32 的数值不同，长文本截到 1024 还是截到 2048 编码出来的向量也不同。这两项如果不在指纹里，就会出现"改了参数但没重建"的静默不一致。

Retriever 每次检索前都会比对这个指纹和当前配置的模型。不一致就直接报错，**不会静默继续**：

```
Embedding model mismatch. Please rebuild vector index.
向量库是用 bge-small-zh-v1.5 (dim=512) 建的，当前模型是 BAAI/bge-m3 (dim=1024)。
请运行 `python ingest.py --rebuild-vectors` 重建向量库。
```

如果继续跑下去，两个不同向量空间里的相似度数字毫无意义，返回的 Top-5 基本是随机的，而且**表面上看服务完全正常**——这是最容易踩的坑之一。指纹检查让它在第一次检索时就炸出来。

---

## 9. 命令行问答（日常用法）

不启服务，直接在终端问自己的笔记：

```bash
python ask.py "合同违约需要承担什么责任？"   # 单次问答
python ask.py --k 8 "什么是 MVP"            # 覆盖检索条数
python ask.py                               # 交互模式，反复提问
```

首次运行要加载 bge-m3（权重约 2.17 GB，十几秒），所以先打印一行"正在加载知识库（约十几秒）…"，加载完报一下库里的 chunk 数和本次的 top_k。

输出分两块：

```
============================================================
……模型的回答，句末带 [1][2][3] 上标……
============================================================

引用 3 条 | 耗时 <N> 秒

[1] 相似度 <score>
    文档：合同法 / 违约责任的承担
    文件：Documents/Obsidian/…/合同法.md
    原文：……开头 200 字预览，剩下的自己打开文件看……
```

耗时和相似度都以你本次实际输出为准：耗时取决于上游 LLM 当时的负载和时段，
相似度没有统一的合格线（见「评测结果怎么读」）。

`[1][2][3]` 和引用列表的编号严格一一对应——模型编不出检索不到的编号。来源只打印开头 200 字，全文打开对应 `.md` 文件看。

交互模式复用同一个引擎，只加载一次模型，问第二题不用再等十几秒；`exit` 退出。

**答案不对时的排查顺序**：先 `python search.py "问题"` 看检索召回了什么（不花钱、不调 LLM）→ 检索对了再查 LLM 环节。

---

## 10. 启动服务（需要 HTTP 接口时才做）

只想在终端里问问题，上一节就够了，不用起 uvicorn。这一节是给需要程序化调用的脚本或工具用的。

```bash
uvicorn app.main:create_app --factory --reload
```

注意 `--factory`：这个参数告诉 uvicorn `create_app` 是一个**函数**，要调它拿到 app 实例。

为什么要多这一步？因为启动时会加载 embedding 模型（bge-m3 约 2.17 GB）。如果写成老式的 `app.main:app`，`app` 变量得在模块末尾 `app = create_app()`，而模块末尾的代码在 `import app.main` 时就会执行——于是**任何 import 都会触发一次模型下载和加载**，`pytest` 收集测试时也不例外。用工厂模式把启动权交给 uvicorn，测试就能用临时的数据库和向量库构造自己的 app 实例，互不干扰。

然后浏览器打开 **http://127.0.0.1:8000/docs** —— 这是 Swagger 自动生成的交互文档，每个接口都可以直接在页面上填参数点 Try it out。

| 接口 | 说明 |
|---|---|
| `GET /health` | 健康检查，附带库里的文档数/chunk 数/当前模型 |
| `GET /api/documents?limit=50&offset=0` | 分页列出已入库文档（不含全文） |
| `GET /api/documents/{id}` | 单篇文档全文 + 元信息 |
| `GET /api/search?q=关键词&k=5` | 只检索不调 LLM |
| `POST /api/chat` | 完整 RAG 问答 |
| `POST /api/ingest` | 触发导入 |

---

## 11. 用 HTTP 接口调 RAG

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 POVM？", "top_k": 5}'
```

返回：

```json
{
  "answer": "……POVM 只确定结果概率[1]……",
  "used_chunks": 3,
  "sources": [
    {"index": 1, "title": "POVM（正算符值测度）", "file_path": "quantum-mechanics/.../povm.md",
     "heading": "POVM 与测量后的状态", "score": 0.91, "chunk": "……"}
  ]
}
```

`answer` 里的上标 `[1][2][3]` 和 `sources` 里的 `index` 严格一一对应——没有检索到来源的编号不会出现，模型编不出来的编号也不会出现。这个不变式有测试守着（`tests/test_api.py::test_api_chat_returns_answer_and_citations`）。

离线模式下 `LLM_PROVIDER=mock` 会回显实际拼好的 Prompt，正好用来确认引用编号和参考资料拼接是否正确。

---

## 12. 在 Claude Code 里直接问（MCP）

`mcp_server.py` 是第三条入口：Claude Code 把它当成自己的一个 MCP 工具服务器，你在对话里提问时它可以自己检索你的知识库，不需要你打开终端或起 uvicorn。

```bash
claude mcp add --scope user obsidian-rag -- \
    "C:\Users\<你的用户名>\obsidian-rag-backend\.venv\Scripts\python.exe" \
    "C:\Users\<你的用户名>\obsidian-rag-backend\mcp_server.py"

claude mcp get obsidian-rag     # 确认 Status: √ Connected
```

暴露两个工具，都只读：

| 工具 | 用途 |
|---|---|
| `ask` | 自然语言提问，返回带引用来源的答案。默认用它。 |
| `search` | 只检索不调 LLM，返回最相近的原文片段。用来排查"答案来源不对"或想直接看笔记原文。 |

**为什么没有 ingest 工具？** 因为读和写要分开：MCP 里开 ingest 等于让 AI 在对话里随手改你的知识库。改知识库必须显式跑 `python ingest.py`。

### 首次调用会很慢，这是故意的

模型加载是懒的，不在进程启动时做。原因是 MCP 握手有 **30 秒超时**，而 bge-m3 权重要十几秒起：把加载放在启动阶段，Claude Code 会直接判定 `connection timed out after 30000ms`，整个服务器连不上。

本机实测三段耗时（RTX 4050 Laptop，首次冷启动）：

| 阶段 | 耗时 |
|---|---|
| 进程启动 → 握手应答 | **4.94 秒**（远低于 30 秒上限） |
| 第一次工具调用（含 bge-m3 加载） | **89.7 秒** |
| 第二次工具调用（模型已驻留） | **0.11 秒** |

代价就是第一问要等约一分钟。之后同一个进程里反复问都不再加载——和 `ask.py` 交互模式复用同一个引擎是同一个道理。

`search` 工具默认走知识域路由（`ask.py` / `search.py` 命令行不默认开，因为终端里可以加 `--router` 自己决定）：不加路由的话，问"海因里希法则"会被量子力学的相似词（变分法、绝热近似）挤到前面，检索到完全无关的领域。这条行为有个测试守着——`tests/test_mcp_server.py::test_routed_domains_*`。

### 证据不足的警告不会被吃掉

`ask` 判定证据不足时，返回文本的**第一行**就是"警告：知识库不足以回答这个问题"，并明确禁止基于那些语义相近的资料编造。这是本项目立 evidence gate 的全部意义：不把这条信息带给调用方，Claude 就会拿相近资料当依据往下编。

---

## 运行测试

```bash
pytest -q
```

298 个测试（刚跑过：`298 passed, 1 warning`），全部用临时目录造假 Vault 和假数据库，**不碰你的真实 Vault，也不碰 `data/`**。
用项目自带的解释器跑，不要用手边的系统 Python：

```bash
.venv\Scripts\python.exe -m pytest -q     # Windows
.venv/bin/python -m pytest -q             # macOS / Linux
```

**测试不需要下载模型**：`tests/test_embedding_semantic.py` 用 `monkeypatch` 把一个假的 `sentence_transformers` 模块塞进 `sys.modules`，模型加载、编码、归一化全部走假的实现。所以跑测试既快又不耗流量，`LocalSemanticEmbedding` 自己的逻辑（分批、缓存、维度探测、前缀、指纹校验）照样被逐个验证到。

**mock 全绿不等于真实模型能跑**——本项目自己就踩过这个坑：第一版实现把 `max_length=1024` 直接传进 `SentenceTransformer.encode()`，85 个测试一路全绿，因为假模型接受任意关键字参数；换成真实的 sentence-transformers 6.0.1 立刻 `ValueError`，因为 `encode()` 对参数做白名单校验，`max_length` 不在名单里。这类错误 mock 永远测不出来。

所以真实模型单独用三个脚本验，都不在默认 pytest 里（否则每次 `pytest` 都要重新加载一次 2.17 GB 的模型）：

```bash
python scripts/diagnose_embedding_determinism.py   # 同一文本重复编码是否可复现（A/B/C/D/E 五组）
python scripts/smoke_real_embedding.py             # 维度、归一化、NaN、语义区分度
python scripts/retrieval_check.py                  # 300 个真实 chunk 建临时索引，跑真实 Top-K
```

覆盖的点：Markdown 能读、SQLite 能存、chunk 数量正常、相同文档不会重复写入、Search 能返回相关 chunk、`/health` 正常、`/api/search` 正常，外加切块边界落在句读上、入库后的下标能精确切回原文（`原文[start:end] == chunk`）、引用编号连续、Vault 前后字节完全一致；本地语义 embedding 的分批/单次加载/归一化/查询前缀/设备选择；换模型被检测并报 `EmbeddingMismatchError`；哈希 fallback 和 OpenAI provider 不受影响；`--rebuild-vectors` 只动 Chroma、SQLite 快照前后完全一致。

`tests/test_ingest_sync.py` 专门覆盖增量导入的四种变化（新增 / 修改 / 删除 / 未变化），
以及三个容易出事的边界：文档改短后旧 chunk 必须清掉、正文被清空后必须清空、
`--path` / `--limit` 下删除同步必须自动跳过、文件读失败不能被当成已删除。

---

## 评测结果怎么读（重要口径警告）

新增的 `doc_recall_at_k` / `precision_at_k` / `ndcg_at_k` 是**文档级**排名指标：同一篇文档的多个 chunk 只算一次。原 `recall_at_k` 实际是“至少命中一篇相关文档”的 hit@k（按原始 chunk 位次计算），为确保历史 A/B 报告可比，保留旧字段与口径；`recall_all_at_5` 已修复多文档题只命中部分文档却误判为全命中的问题。新旧口径不可混用。指标实现及独立测试见 `app/eval_metrics.py`、`tests/test_eval_metrics.py`，开发说明见 [RANKING_METRICS.md](RANKING_METRICS.md)。


`eval/retrieval_questions.json` 里有 32 道题，**全部 `verified: false`，还一道都没有人工确认过。**

`eval_retrieval.py` 默认会跳过未确认的题目，所以直接跑会得到 0 道可计分题目，并提示
「没有可计分的题目」。加 `--all` 可以连未确认的一起算，但那算的是对**我猜测的答案文档**的召回，
不是对确定 ground truth 的召回。`RETRIEVAL_BASELINE.md` 和 `eval/results/` 下的所有数字
都是用 `--all` 跑出来的。

因此，在人工确认之前：

- **Recall@K / MRR 只能当作内部实验数据**，用来做同一系统改前改后的相对 A/B 对比；
  **不能当作正式 ground truth 结果**对外引用或写进结论。
- 某道题标错了期望文档，那道题的失败会算到系统的头上。这是当前数字最大的不确定性来源。
- **绝对余弦相似度分数不是统一的质量标准**。0.60 和 0.63 的差距没有意义，
  同一个分数在不同语料、不同问题上能对应完全不同的检索质量。
  质量判断以「期望文档是否排在前面 / Recall@K / MRR / 人工读一遍结果」为准，不要拿单次 score 去调参。

确认方式是手动的：读完一道题、对照真实笔记核对期望文档之后，
把那道题的 `verified` 改成 `true`。这个环节刻意不做自动化——它本来就需要人来判断。

---

## 后续优化（本轮刻意不做，不是遗漏）

**token-aware chunk splitting** —— 当前切块按字符数（`CHUNK_SIZE=800`），中文约 0.71 token/字符，所以全库有 12 / 14303 个 chunk 超过 1024 token，尾部内容没进向量。彻底的做法是按 tokenizer 的 token 边界切块而不是按字符数。本轮没做：它要改 chunker，并且会让 SQLite 里已有的 14303 个 chunk 全部作废重新切一遍。现在的取舍是截断但**不静默**——逐条打印 `chunk_id / document / token_length`，结束时汇总数量。

**dynamic batching** —— 按 token 长度分桶，短文本用大 batch、长文本用小 batch。做过实测才决定不做：压力测试里每条都是近 1024 token 的长文本，batch 从 8 加到 16，单条只快 2.9%（38.6 → 37.5 ms），显存多占 256 MiB。收益上限就这几个百分点，不值得为它加复杂度。

**导航/链接类 chunk 污染 Top-K** —— 真实检索里发现：《Harsanyi-1967-贝叶斯博弈》的「知识库链接」章节（纯 `[[wiki-link]]` 列表，没有实质内容）能排进 Top-5，score 0.5656。这类 chunk 本身是合法内容，只是对问答没价值。可以在入库时按 heading 名过滤，但会丢掉一部分索引，本轮保留原样。

**reranker（已实现、已实测、默认关闭）** —— 全量 14303 chunk 上跑真实检索后，
「章节标题同名」的干扰才暴露出来：问 Cournot 竞争时，《柠檬市场》和《Stackelberg竞争》的
「模型设定」章节靠通用标题 + 产量相关词排进 Top-2 / Top-3，真正的 Cournot竞争 掉到 #5。
300 chunk 的临时索引里没这个问题（语料小、干扰项少），**只有全量规模才看得见**。

于是 `app/reranker.py` 做了出来，并且用 A/B 四组实测过（结果在 `RETRIEVAL_BASELINE.md` 第 10 节）：
Recall@5 从 0.8077 升到 0.8846，MRR 从 0.6551 升到 0.7500，改善集中在最弱的
paraphrase / hard_negative 两类；代价是检索耗时 ×41。

**所以 `RERANKER_ENABLED` 仍然默认 `false`**——理由从「没验证过」变成了
「验证过、确实有效、但延迟代价和收益不匹配」。需要时把它改成 `true` 即可，
代码、测试、评测口径都已经就位。换更强排序器也不会解决的下一处是文档层级问题，
不在本轮范围内。

**vault 文档里的 JSON 残留** —— 《纳什均衡存在性定理》的「历史与扩展」章节原文尾部混有 `", "file_path": "/c/Users/<你的用户名>/Desk…` 这类字段残留。切块器照原样搬运、不清洗：检索排序不受影响（该 chunk 语义命中仍然正确），但内容摘要里能看见。修它要改 vault 原文，而本项目只读 vault，不在本轮范围内。

**鉴权、多用户** —— 接口和命令行都本机直连，没有登录也没有权限划分，不要暴露到公网。Web 界面已按需求删掉，问答入口就是 `ask.py` 和 HTTP 接口这两条。

## 常见问题

**Q: 答案引用了错误的来源？**
先跑 `python search.py "你的问题"` 看检索返回了什么。90% 的情况是检索没找到对的 chunk，而不是 LLM 的错。

**Q: 相似度数字看起来都很低？**
如果是 `local` 伪向量（`EMBEDDING_PROVIDER=local`），那是哈希出来的假向量，**分数完全没有意义**，
别拿它当依据。换成真模型后，**绝对分数也没有统一的"合格线"**——同一个分数在不同语料、
不同问题上可以对应完全不同的检索质量，给不出"0.X 以上就算好"这种判断标准。
看质量请直接看排名：`python search.py "你的问题"` 看期望的文档排第几，
或者跑 `eval_retrieval.py` 看 Recall@K / MRR。

**Q: 导入了 900 多篇要多久？**
全库 911 篇 / 14303 个 chunk。本机实测 `BAAI/bge-m3` 在 RTX 4050 Laptop（6 GB 显存）上用 GPU 跑 `--rebuild-vectors` 重建全量向量 **364.2 秒（约 6 分 4 秒）**，平均 0.03 秒/条，峰值显存 1275 MiB——这个数字已经包含约 40 秒的模型加载时间。平时不用跑：chunk 原文没变时增量导入会整篇跳过，`--rebuild-vectors` 只在换 embedding 模型后才需要。纯 CPU 上会明显更慢，但本项目没在 CPU 上实测过，不给数字。`EMBEDDING_BATCH_SIZE` 默认 **8** 就是实测取的值，显存紧张才往下调，不建议调更大。

**Q: 想重新导入全部内容？**
分两种情况，别搞混：

```bash
# 情况一：只换了 embedding 模型 —— 只重建向量，保留 SQLite 里的原文和 chunk
python ingest.py --rebuild-vectors

# 情况二：真的想从头再来（比如改了 CHUNK_SIZE 想重新切块）
# 这才会删 SQLite。删之前确认你不需要回滚。
rm -rf data/chroma data/knowledge.db*
python ingest.py
```

`CHUNK_SIZE` / `CHUNK_OVERLAP` 变了必须走情况二——chunk 的切分方式变了，旧的 chunk 全部作废。

**Q: 报 `Embedding model mismatch`？**
说明 `.env` 里的模型和 `data/chroma/` 里已有的向量不是同一个模型建的。跑 `python ingest.py --rebuild-vectors` 即可。这是刻意设计成报错的——两个向量空间的相似度数字没有可比性，硬跑只会得到随机结果，而且服务看起来完全正常。

**Q: 向量库和 SQLite 对不上了怎么办？**
Chroma 只是派生物，SQLite 才是原文。任何时候删掉 `data/chroma` 重来都不损失信息：
```bash
rm -rf data/chroma
python ingest.py --rebuild-vectors
```

**Q: 想加新功能而不把代码搞乱？**
保持"一个文件一个职责"。新增检索策略改 `retriever.py`，新增模型改 `embedding.py` 或 `llm.py` 加一个类 + 在工厂函数里加分支，不要在端点里写业务逻辑。
