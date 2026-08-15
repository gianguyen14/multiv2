import numpy as np

from backend.app.retrieval.m12_pipeline import M12RetrievalPipeline
from backend.app.retrieval.ranking_metrics import mrr_at_k, ndcg_at_k
from backend.app.retrieval.semantic_reranker import SemanticReranker
from eval.m12_hard_negative_dataset import build_hard_negative_dataset


class FakeEncoder:
    def __init__(self, embedding):
        self.embedding = embedding

    def encode_text(self, texts):
        return np.asarray([self.embedding], dtype=np.float32)


class FakeRetriever:
    def __init__(self, candidates):
        self.candidates = candidates
        self.requested_k = None

    def search(self, query, top_k, top_n=None):
        self.requested_k = top_k
        return [dict(item) for item in self.candidates[:top_k]]


def test_hard_negative_dataset_is_genuinely_difficult():
    dataset = build_hard_negative_dataset()
    irrelevant = [item for item in dataset.candidates if item["relevance"] == 0]
    relevant = [item for item in dataset.candidates if item["relevance"] > 0]
    assert max(item["retrieval_score"] for item in irrelevant) > max(item["retrieval_score"] for item in relevant)
    assert len({item["shard_id"] for item in irrelevant}) > 1
    assert len(dataset.candidates) > len({item["frame_id"] for item in dataset.candidates})


def test_candidate_generation_reranking_and_quality_gate():
    dataset = build_hard_negative_dataset()
    unique_candidates = []
    seen = set()
    for candidate in dataset.candidates:
        if candidate["frame_id"] not in seen:
            seen.add(candidate["frame_id"])
            unique_candidates.append(candidate)
    retriever = FakeRetriever(unique_candidates)
    reranker = SemanticReranker(scorer=lambda query, candidate: candidate["relevance"], backend_name="test_semantic_scorer")
    pipeline = M12RetrievalPipeline(FakeEncoder(dataset.query_embedding), retriever, reranker, candidate_k=len(unique_candidates), final_k=3)
    baseline = unique_candidates[:3]
    results, diagnostics = pipeline.search_text("red bicycle", include_diagnostics=True)
    baseline_ids = [item["frame_id"] for item in baseline]
    result_ids = [item["frame_id"] for item in results]
    assert retriever.requested_k == len(unique_candidates)
    assert set(result_ids).issubset({item["frame_id"] for item in unique_candidates})
    assert ndcg_at_k(result_ids, dataset.relevance, 3) > ndcg_at_k(baseline_ids, dataset.relevance, 3)
    assert mrr_at_k(result_ids, dataset.relevance, 3) > mrr_at_k(baseline_ids, dataset.relevance, 3)
    assert diagnostics["candidate_count"] == len(unique_candidates)
    assert diagnostics["final_count"] == 3
    assert diagnostics["fallback_used"] is False


def test_pipeline_survives_semantic_backend_failure():
    dataset = build_hard_negative_dataset()
    candidates = dataset.candidates[:4]
    reranker = SemanticReranker(scorer=lambda query, candidate: (_ for _ in ()).throw(RuntimeError("down")))
    pipeline = M12RetrievalPipeline(FakeEncoder(dataset.query_embedding), FakeRetriever(candidates), reranker, candidate_k=4, final_k=2)
    results, diagnostics = pipeline.search_text("query", include_diagnostics=True)
    assert len(results) == 2
    assert diagnostics["fallback_used"] is True
