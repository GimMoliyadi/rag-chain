# Experimental BM25 + Dense + RRF retrieval

This opt-in CLI mode combines lexical BM25 ranking with the existing dense
Chroma retriever using Reciprocal Rank Fusion (RRF).

## Usage

First ingest your Obsidian notes using the existing instructions. Then:

```bash
python search.py "债券价格为什么下降" --k 5
python search.py "债券价格为什么下降" --k 5 --hybrid
python search.py "货币政策" --hybrid --router
```

The first command is the unchanged dense baseline. The second reads a
**read-only snapshot** of chunks from the existing SQLite database and builds
an in-memory inverted BM25 index; nothing is written to the Vault, SQLite or
Chroma. No embedding recomputation is required. It uses mixed Latin word
tokens / overlapping Chinese character bigrams, then fuses the top 4*K hits
from each system with RRF (k=60).

## Limitations and next experiment

- `score` in hybrid mode represents **relative RRF rank-fusion strength**, not
  cosine similarity or calibrated evidence confidence.
- The optional mode is implemented only in `search.py`. Normal `ask.py`,
  FastAPI and MCP remain on the verified dense retrieval path. Threshold-based
  evidence grading must be separately calibrated before wiring up hybrid.
- This in-memory index is rebuilt on each CLI run; a long-running app must
  explicitly rebuild it after ingest. Peak memory and index build latency have
  not been measured on the user's full 14k+ chunks.
- Bigram BM25 is a transparent baseline, not a claim that Chinese tokenization
  is optimal. Compare with jieba/FTS-backed alternatives on real queries.
- Your private retrieval question set is unverified, so **no quality gain** is
  claimed. Build verified ground truth before drawing conclusions.
- RRF provides rank ordering only, and its numeric scale is unsuitable for
  comparing with legacy cosine-similarity scores.

Quick isolated tests:

```bash
python -m pytest --noconftest tests/test_hybrid_retrieval.py -q
```

This prepares a working experimental baseline for a later fair A/B benchmark,
without changing production answer behavior.
