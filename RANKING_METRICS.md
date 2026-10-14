# Document-level ranking metrics

This change adds testable, dependency-free ranking metrics while preserving the
legacy evaluation report fields and the 26-question historic baseline.

## Why this matters

The old `recall_at_k` is actually binary **Hit@K**: any relevant document
appearing in the first K *chunks* counts as success. It is not the same as
standard document Recall@K (the fraction of all relevant documents retrieved).
Furthermore, the former `recall_all_at_5` implementation could report 1 even
when one or more expected documents were missing entirely; that bug is fixed.

## New metrics

- `hit_at_k`: at least one relevant document in the first K distinct documents.
- `doc_recall_at_k`: relevant distinct documents found / number of relevant documents.
- `precision_at_k`: relevant distinct documents found / K.
- `ndcg_at_k`: graded ranking quality, (2^relevance - 1) / log2(rank+1),
  divided by the ideal DCG at the same K. The current private question format
  gives every expected document relevance=1.

Repeated chunks from the same document count once at their first occurrence.
This is a **document-level** view, distinct from the original chunk-level
`recall_at_k` and `mrr5` values; both families remain in JSON outputs for
A/B traceability. The script only receives its retrieved top-K chunks, so
a document appearing beyond that pool cannot be scored.

## Run

```bash
python -m pytest tests/test_eval_metrics.py -q
python eval_retrieval.py --mode raw --out eval/results/ranking-report.json
```

The second command requires an existing index, embedding configuration, and
*verified* relevance judgements. In the current repository the user's genuine
question list is intentionally ignored by git; the checked-in example is not a
benchmark. Any `--all` run with unverified questions is exploratory only.

Do not compare an old `recall_at_5` figure directly with
`doc_recall_at_5` or claim a performance improvement just because a new metric
became available. No indexing or generation behavior changes in this PR.
