import numpy as np

from backend.app.retrieval.candidate_resolver import MappingCandidateResolver
from backend.app.retrieval.m13_pipeline import M13RetrievalPipeline
from backend.app.retrieval.model_scorer import DeterministicTestScorer, ModelScorer


class FakeEncoder:
    def encode_text(self, texts):
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class FakeRetriever:
    def __init__(self, candidates):
        self.candidates = candidates
        self.requested_k = None

    def search(self, query, top_k, top_n=None):
        self.requested_k = top_k
        return [dict(candidate) for candidate in self.candidates[:top_k]]


class FailingScorer(ModelScorer):
    backend_name = "failing_test_scorer"

    def score_batch(self, query, candidates):
        raise RuntimeError("unavailable")


def fixtures():
    candidates = [
        {"frame_id": "a", "score": 0.9},
        {"frame_id": "b", "score": 0.8},
        {"frame_id": "c", "score": 0.7},
    ]
    payloads = {
        "a": {"relevance": 0, "caption": "negative"},
        "b": {"relevance": 3, "caption": "positive"},
        "c": {"relevance": 1, "caption": "secondary"},
    }
    return candidates, payloads


def test_retrieve_resolve_batch_score_and_same_candidate_invariant():
    candidates, payloads = fixtures()
    retriever = FakeRetriever(candidates)
    scorer = DeterministicTestScorer()
    pipeline = M13RetrievalPipeline(FakeEncoder(), retriever, MappingCandidateResolver(payloads), scorer, candidate_k=3, final_k=2)
    results, diagnostics = pipeline.search_text("query", include_diagnostics=True)
    assert retriever.requested_k == 3
    assert scorer.seen_candidate_ids == ["a", "b", "c"]
    assert diagnostics["candidate_ids"] == ["a", "b", "c"]
    assert [result["frame_id"] for result in results] == ["b", "c"]
    assert diagnostics["resolved_candidate_count"] == 3
    assert diagnostics["fallback_used"] is False


def test_model_failure_fails_open_to_retrieval_order():
    candidates, payloads = fixtures()
    pipeline = M13RetrievalPipeline(FakeEncoder(), FakeRetriever(candidates), MappingCandidateResolver(payloads), FailingScorer(), candidate_k=3, final_k=2)
    results, diagnostics = pipeline.search_text("query", include_diagnostics=True)
    assert [result["frame_id"] for result in results] == ["a", "b"]
    assert diagnostics["fallback_used"] is True
    assert diagnostics["fallback_reason"] == "RuntimeError"


def test_missing_payload_fails_open_without_dropping_results():
    candidates, payloads = fixtures()
    del payloads["b"]
    pipeline = M13RetrievalPipeline(FakeEncoder(), FakeRetriever(candidates), MappingCandidateResolver(payloads), DeterministicTestScorer(), candidate_k=3, final_k=2)
    results, diagnostics = pipeline.search_text("query", include_diagnostics=True)
    assert [result["frame_id"] for result in results] == ["a", "b"]
    assert diagnostics["unresolved_candidate_count"] == 1
    assert diagnostics["fallback_used"] is True
