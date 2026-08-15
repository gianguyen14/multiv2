import numpy as np

from backend.app.retrieval.semantic_reranker import SemanticReranker


def candidates():
    return [
        {"frame_id": "b", "score": 0.9, "embedding": np.array([1, 0]), "metadata": {"keep": True}},
        {"frame_id": "a", "score": 0.8, "embedding": np.array([0, 1]), "metadata": {"keep": True}},
    ]


def test_callback_is_used_and_metadata_preserved():
    calls = []
    reranker = SemanticReranker(scorer=lambda query, candidate: calls.append(candidate["frame_id"]) or (1 if candidate["frame_id"] == "a" else 0))
    result = reranker.rerank("query", candidates(), 2)
    assert calls == ["b", "a"]
    assert [item["frame_id"] for item in result] == ["a", "b"]
    assert result[0]["metadata"] == {"keep": True}
    assert result[0]["rank"] == 1


def test_dot_product_fallback_and_top_k():
    result = SemanticReranker().rerank(np.array([0, 1]), candidates(), 1)
    assert [item["frame_id"] for item in result] == ["a"]
    assert result[0]["embedding"].tolist() == [0, 1]
    assert SemanticReranker().last_diagnostics == {}


def test_batch_scoring_and_equal_score_ties_are_deterministic():
    reranker = SemanticReranker(batch_scorer=lambda query, rows: [1.0] * len(rows))
    first = reranker.rerank("q", candidates(), None)
    second = reranker.rerank("q", reversed(candidates()), None)
    assert [item["frame_id"] for item in first] == ["a", "b"]
    assert [item["frame_id"] for item in second] == ["a", "b"]


def test_model_failure_falls_back_to_retrieval_order():
    def fail(query, candidate):
        raise RuntimeError("backend unavailable")

    reranker = SemanticReranker(scorer=fail)
    result = reranker.rerank("q", candidates(), 2)
    assert [item["frame_id"] for item in result] == ["b", "a"]
    assert reranker.last_diagnostics["fallback_used"] is True
    assert reranker.last_diagnostics["reranker_backend"] == "retrieval_order_fallback"
    assert reranker.rerank("q", candidates(), 0) == []
