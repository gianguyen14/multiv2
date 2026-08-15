import numpy as np

from backend.app.retrieval.candidate_resolver import MappingCandidateResolver
from backend.app.retrieval.m14_pipeline import M14RetrievalPipeline
from backend.app.retrieval.model_scorer import ModelScorer


class Encoder:
    def encode_text(self, texts):
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class Retriever:
    def __init__(self):
        self.calls = 0
        self.rows = [{"frame_id": "a", "score": .9}, {"frame_id": "b", "score": .8}, {"frame_id": "c", "score": .7}]

    def search(self, query, top_k):
        self.calls += 1
        return [dict(row) for row in self.rows[:top_k]]


class Scorer(ModelScorer):
    backend_name = "test_m14"
    signal_type = "joint_multimodal_relevance"

    def __init__(self, scores):
        self.scores = scores
        self.ids = None

    def score_batch(self, query, candidates):
        self.ids = [candidate["candidate_id"] for candidate in candidates]
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores


def pipeline(scores):
    retriever = Retriever()
    resolver = MappingCandidateResolver({key: {"image_path": f"{key}.png", "metadata": {"keep": key}} for key in "abc"})
    scorer = Scorer(scores)
    return M14RetrievalPipeline(Encoder(), retriever, resolver, scorer, candidate_k=3, final_k=2), retriever, scorer


def test_retrieves_once_scores_same_pool_preserves_metadata_and_stable_ties():
    subject, retriever, scorer = pipeline([10, 90, 90])
    results, diagnostics = subject.search_text("query", include_diagnostics=True)
    assert retriever.calls == 1
    assert scorer.ids == diagnostics["candidate_ids"] == ["a", "b", "c"]
    assert [row["frame_id"] for row in results] == ["b", "c"]
    assert results[0]["metadata"] == {"keep": "b"}
    assert diagnostics["fallback_used"] is False


def test_invalid_score_or_exception_fails_open_for_entire_query():
    for scores in ([1, float("nan"), 3], RuntimeError("down")):
        subject, retriever, _ = pipeline(scores)
        results, diagnostics = subject.search_text("query", include_diagnostics=True)
        assert [row["frame_id"] for row in results] == ["a", "b"]
        assert diagnostics["fallback_used"] is True
        assert retriever.calls == 1


def test_hybrid_uses_same_candidate_pool_and_final_k():
    subject, _, _ = pipeline([10, 90, 20])
    results = subject.search_text("query", ranking="hybrid")
    assert len(results) == 2
    assert {row["frame_id"] for row in results}.issubset({"a", "b", "c"})
