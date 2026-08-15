import pytest

from backend.app.retrieval.ranking_metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_perfect_and_reversed_ranking():
    relevance = {"a": 3, "b": 2, "c": 1}
    assert recall_at_k(["a", "b", "c"], relevance, 3) == 1.0
    assert precision_at_k(["a", "b", "c"], relevance, 3) == 1.0
    assert mrr_at_k(["a", "b", "c"], relevance, 3) == 1.0
    assert ndcg_at_k(["a", "b", "c"], relevance, 3) == 1.0
    assert ndcg_at_k(["c", "b", "a"], relevance, 3) < 1.0


def test_partial_empty_and_nonpositive_k():
    assert recall_at_k(["x", "a"], {"a", "b"}, 2) == 0.5
    assert precision_at_k(["x", "a"], {"a", "b"}, 2) == 0.5
    assert mrr_at_k(["x", "a"], {"a"}, 2) == 0.5
    for metric in (recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k):
        assert metric([], {"a"}, 10) == 0.0
        assert metric(["a"], set(), 10) == 0.0
        assert metric(["a"], {"a"}, 0) == 0.0


def test_duplicates_do_not_inflate_metrics():
    predictions = ["a", "a", "x"]
    assert recall_at_k(predictions, {"a", "b"}, 3) == 0.5
    assert precision_at_k(predictions, {"a", "b"}, 3) == pytest.approx(1 / 3)
    assert ndcg_at_k(predictions, {"a": 2, "b": 1}, 3) < 1.0
