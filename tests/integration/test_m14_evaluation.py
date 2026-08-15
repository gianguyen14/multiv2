import numpy as np

from backend.app.retrieval.model_scorer import ModelScorer
from eval.m14_9router_quality import hard_negative_error, run_quality


class Encoder:
    model_name = "fake"
    def encode_image(self, images, batch_size=16):
        rows = np.eye(len(images), 32, dtype=np.float32)
        return rows
    def encode_text(self, texts, batch_size=1):
        rows = []
        for text in texts:
            row = np.zeros(32, dtype=np.float32)
            row[sum(text.encode()) % 24] = 1
            rows.append(row)
        return np.asarray(rows)


class Scorer(ModelScorer):
    backend_name = "deterministic_m14_test_scorer"
    signal_type = "joint_multimodal_relevance"
    model = "fake-joint-model"
    base_url = "mock://9router"
    max_concurrency = 2

    def __init__(self):
        self._diagnostics = {}
    def score_batch(self, query, candidates):
        self._diagnostics = {"request_count": len(candidates), "retry_count": 0, "failed_request_count": 0, "input_tokens": 0, "output_tokens": 0}
        return [100 - int(candidate["candidate_id"].split("_")[1]) for candidate in candidates]
    @property
    def diagnostics(self):
        return self._diagnostics


def test_m14_same_candidate_quality_signal_and_bootstrap_reporting():
    result = run_quality(candidate_ks=(10,), encoder=Encoder(), scorer=Scorer())
    report = result["results_by_candidate_k"]["10"]
    assert set(report["metrics"]) == {"faiss", "siglip_exact", "m14", "hybrid"}
    assert set(report["deltas_from_faiss@10"]["m14"]) == {"mrr", "ndcg"}
    assert "faiss_vs_m14" in report["signal_comparisons"]
    assert sum(report["query_outcomes"].values()) == 6
    assert set(report["bootstrap"]) == {"mrr", "ndcg"}
    assert set(report["hard_negative_error_rate"]) == {"faiss", "siglip_exact", "m14", "hybrid"}
    for query in report["per_query"]:
        assert len(query["candidate_ids"]) == 10
        assert len(set(query["candidate_ids"])) == 10


def test_hard_negative_error_detects_negative_above_grade_three():
    relevance = {"positive": 3, "negative": 0}
    assert hard_negative_error([{"candidate_id": "negative"}, {"candidate_id": "positive"}], relevance, {"negative"}) == 1
    assert hard_negative_error([{"candidate_id": "positive"}, {"candidate_id": "negative"}], relevance, {"negative"}) == 0
