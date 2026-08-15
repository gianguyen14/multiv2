import pytest

from backend.app.retrieval.hybrid_ranker import HybridRanker


def test_normalization_and_configurable_weights():
    rows = [
        {"frame_id": "retrieval", "retrieval_score": 1.0, "semantic_score": 0.0},
        {"frame_id": "semantic", "retrieval_score": 0.0, "semantic_score": 2.0},
    ]
    assert HybridRanker({"retrieval_score": 1, "semantic_score": 0}).rank(rows)[0]["frame_id"] == "retrieval"
    assert HybridRanker({"retrieval_score": 1, "semantic_score": 3}).rank(rows)[0]["frame_id"] == "semantic"
    assert sum(HybridRanker({"retrieval_score": 1, "semantic_score": 3}).weights.values()) == pytest.approx(1)


def test_zero_variance_dedup_and_determinism():
    rows = [
        {"frame_id": "b", "retrieval_score": 1, "semantic_score": 1},
        {"frame_id": "a", "retrieval_score": 1, "semantic_score": 1},
        {"frame_id": "a", "retrieval_score": 0, "semantic_score": 10},
    ]
    result = HybridRanker().rank(rows, 2)
    assert [item["frame_id"] for item in result] == ["a", "b"]
    assert len(result) == len({item["frame_id"] for item in result})
    assert all(item["score"] == 0 for item in result)


def test_invalid_weights_and_nonpositive_top_k():
    with pytest.raises(ValueError):
        HybridRanker({"retrieval_score": 0})
    assert HybridRanker().rank([], 0) == []
