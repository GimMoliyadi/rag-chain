# LangChain 1.x 迁移记录

目标：把现有 `rag-chain` 的通用 RAG 能力逐步迁移到成熟的 LangChain 生态，**不是 Fork 其他完整产品、也不是重做一个业务网站**。

## 迁移阶段

### 第一阶段：标准接口与检索、Prompt（本次 PR）

- 使用 `langchain_chroma.Chroma` 作为**检索执行接口**，复用现有 Chroma client、collection、向量与元数据。
- 使用 `langchain_core.embeddings.Embeddings` 适配已有本地和 API Embedding provider；保留查询侧前缀、模型指纹、GPU 配置。
- 使用 `langchain_core.prompts.ChatPromptTemplate` 生成普通/顾问模式的 system 与 user 消息。
- 保留原有 `/api/search`、`/api/chat`、CLI、MCP 的外部数据结构，检索分数依然按照余弦距离解释。
- 现有数据库、文档来源、引用编号、证据评审、救援、Router、Reranker 继续有效。

### 为什么暂时保留部分自研模块？

- `app/chunker.py` 维护 `original[start:end] == chunk.content` 的**原文精确偏移量**，直接换通用 splitter 会改变引用定位和 14k+ chunks 的存储格式。
- SQLite 是唯一原文事实来源，Chroma 是可以重建的派生索引；现有两阶段重建提供出错回滚能力。
- `ChromaStore` 的写入、删除、元数据回填仍调用原生 Chroma：已有预计算 Embedding 和文档 ID，直接替换成 `add_documents` 会重复嵌入并带来数据迁移风险。
- RAG 引擎的证据裁决、引用编号、个人顾问模式、流式回答暂时保留。**还不能声称整套编排已全部迁至 LCEL。**

### 下一阶段（需通过回归测试后再做）

1. 审计生成端 provider 与流式兼容性，评估迁移 `langchain_openai` 及 LCEL Runnable 链的成本。
2. 用经过人工核验的相同测试集对比迁移前后 Recall@K、MRR、P95 延迟；原则上本阶段不应改变指标。
3. 只有实现收益明确时才迁移分块/入库，不能破坏来源偏移、SQLite 原文、模型指纹。
4. 之后在 LangChain 生态内评估混合检索、结果多样性与 reranker 优化。

## 验收

```bash
python -m pip install -r requirements.txt
python -m pytest tests/test_langchain_integration.py tests/test_retrieval.py tests/test_chunker.py -q
python -m pytest -q
```

完成这些静态与 mock 测试之后，仍需要本地真实语料的检索 A/B 及实际 LLM 问答验证。测试前备份 `data/`；不要直接覆盖或清空已有 SQLite / Chroma 数据。

**本次迁移还没有改变模型、Chunk、检索策略或业务需求，因此性能提升不是预设结论。**
