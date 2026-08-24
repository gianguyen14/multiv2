import numpy as np

from eval.m13_5_corpus import load_corpus
from eval.m13_5_ground_truth_benchmark import bootstrap_delta, run_benchmark


class FakeEncoder:
    model_name = "deterministic-fixture-encoder"
    device = "cpu"
    dtype = np.float32

    def _vector(self, value):
        vector = np.zeros(8, dtype=np.float32)
        vector[value % 8] = 1.0
        return vector

    def encode_image(self, images, batch_size=32):
        return np.asarray([self._vector(index) for index, _ in enumerate(images)], dtype=np.float32)

    def encode_text(self, texts, batch_size=32):
        return np.asarray([self._vector(sum(text.encode()) % 24) for text in texts], dtype=np.float32)


def test_same_candidate_evaluation_reports_all_metrics():
    result = run_benchmark("eval/data/m13_5", candidate_ks=(10, 24), final_ks=(5, 10), batch_size=4, repetitions=1, encoder=FakeEncoder())
    assert result["run_metadata"]["dataset_fingerprint"] == load_corpus("eval/data/m13_5").fingerprint
    assert result["run_metadata"]["dataset_kind"] == "synthetic_contract_fixture"
    assert result["run_metadata"]["quality_claims_allowed"] is False
    for candidate_k in ("10", "24"):
        report = result["results_by_candidate_k"][candidate_k]
        assert set(report["metrics"]) == {"faiss", "dot_product", "siglip2", "hybrid"}
        assert "candidate_recall" in report
        assert "faiss_vs_siglip2" in report["comparisons"]
        assert "mean_absolute_difference" in report["score_differences"]["faiss_vs_siglip"]
        for query in report["per_query"]:
            expected = query["candidate_ids"]
            assert all(len(expected) == len(set(expected)) for _ in [0])
            assert query["outcome"] in {"improved", "unchanged", "regressed"}


def test_bootstrap_is_deterministic():
    first = bootstrap_delta([0.1, 0.2, 0.3], [0.2, 0.2, 0.4], repetitions=100, seed=7)
    second = bootstrap_delta([0.1, 0.2, 0.3], [0.2, 0.2, 0.4], repetitions=100, seed=7)
    assert first == second
