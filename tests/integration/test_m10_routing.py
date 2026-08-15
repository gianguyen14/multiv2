import numpy as np

from backend.app.distributed.centroid_index import ShardCentroidIndex
from backend.app.distributed.m10_cache import M10QueryCache
from backend.app.distributed.m10_merge import merge_normalized
from backend.app.distributed.routing_policy import RoutingPolicy
from backend.app.distributed.semantic_router import SemanticShardRouter


def test_semantic_routing_selects_closest_shards():
    centroids = {0: np.array([[1.0, 0.0]], dtype=np.float32), 1: np.array([[0.0, 1.0]], dtype=np.float32)}
    router = SemanticShardRouter(2, centroids, mode="hybrid")
    assert router.route(np.array([0.9, 0.1], dtype=np.float32), 1) == [0]
    assert len(router.route(np.array([0.9, 0.1], dtype=np.float32), 2)) == 2


def test_merge_normalizes_and_deduplicates():
    results = merge_normalized(
        [[{"frame_id": "a", "score": 2.0}], [{"frame_id": "a", "score": 1.0}, {"frame_id": "b", "score": 3.0}]], 2
    )
    assert len(results) == len({item["frame_id"] for item in results})
    assert results[0]["frame_id"] == "a"


def test_cache_respects_routing_configuration():
    cache = M10QueryCache()
    query = np.array([1.0, 0.0], dtype=np.float32)
    cache.set(query, 5, 1, "flat", 2, [{"frame_id": "a", "score": 1.0, "rank": 1}])
    assert cache.get(query, 5, 1, "flat", 2) is not None
    assert cache.get(query, 5, 2, "flat", 2) is None


def test_adaptive_policy_changes_behavior():
    policy = RoutingPolicy()
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert policy.choose_top_n(query, 5, latency_budget_ms=1) == 1
    assert policy.choose_top_n(query, 5, latency_budget_ms=20) >= 2
