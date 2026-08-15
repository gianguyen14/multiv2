import numpy as np

from backend.app.distributed.m11_merge import merge_hybrid, merge_zscore
from backend.app.distributed.m11_router import M11SemanticRouter
from backend.app.retrieval.cross_encoder_reranker import CrossEncoderReranker


def test_multi_centroid_routing_and_recall():
    data = {0: np.array([[1, 0, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32), 1: np.array([[0, 1, 0, 0]], dtype=np.float32)}
    router = M11SemanticRouter(2, data, mode="hybrid")
    assert router.route(np.array([1, 0, 0, 0], dtype=np.float32), 1) == [0]
    assert len(router.route(np.array([1, 0, 0, 0], dtype=np.float32), 2)) == 2


def test_reranker_improves_ordering():
    reranker = CrossEncoderReranker()
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = [{"frame_id": "low", "embedding": np.array([0, 1], dtype=np.float32)}, {"frame_id": "high", "embedding": query}]
    assert reranker.rerank(query, candidates, 2)[0]["frame_id"] == "high"


def test_hybrid_merge_and_zscore_dedup():
    merged = merge_zscore([[{"frame_id": "a", "score": 1.0}], [{"frame_id": "a", "score": 2.0}, {"frame_id": "b", "score": 0.0}]], 2)
    assert len({item["frame_id"] for item in merged}) == len(merged)
    assert merge_hybrid([{"frame_id": "a", "score": 1.0}], [{"frame_id": "b", "score": 1.0}], 2)
