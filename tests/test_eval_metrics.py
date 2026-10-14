"""Ranking metric regressions: independent of Chroma, GPU and personal Vault."""
import math

import pytest

from app.eval_metrics import document_ranking_metrics


def test_binary_ranking_relevance_and_cutoffs():
    result = document_ranking_metrics({"a": 1, "b": 1}, ["x", "a", "b"], (1, 2, 3))
    assert result["hit_at_1"] == 0
    assert result["doc_recall_at_2"] == 0.5
    assert result["doc_recall_at_3"] == 1
    assert result["precision_at_3"] == pytest.approx(2 / 3)
    assert result["ndcg_at_1"] == 0
    assert 0 < result["ndcg_at_3"] < 1


def test_duplicate_chunks_from_same_document_are_not_extra_hits():
    result = document_ranking_metrics({"a": 1, "b": 1}, ["a", "a", "x", "b"], (2, 3))
    assert result["doc_recall_at_2"] == 0.5
    assert result["doc_recall_at_3"] == 1
    assert result["precision_at_3"] == pytest.approx(2 / 3)


def test_graded_relevance_ndcg_ideal_and_reversed():
    ideal = document_ranking_metrics({"a": 3, "b": 1}, ["a", "b"], (2,))
    reversed_ranking = document_ranking_metrics({"a": 3, "b": 1}, ["b", "a"], (2,))
    assert ideal["ndcg_at_2"] == pytest.approx(1)
    assert reversed_ranking["ndcg_at_2"] < ideal["ndcg_at_2"]


def test_no_retrieval_returns_zeros():
    result = document_ranking_metrics({"a": 1}, [], (1, 5))
    assert all(value == 0 for value in result.values())


@pytest.mark.parametrize("qrels", [{}, {"x": 0}, {"x": -1}, {"x": math.nan}, {"x": math.inf}])
def test_invalid_or_unscorable_relevance_fails(qrels):
    with pytest.raises(ValueError):
        document_ranking_metrics(qrels, ["x"])


@pytest.mark.parametrize("cutoffs", [(0,), (-1,), (1.5,), (True,)])
def test_bad_cutoffs_fail(cutoffs):
    with pytest.raises(ValueError):
        document_ranking_metrics({"x": 1}, ["x"], cutoffs)
