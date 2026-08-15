from backend.app.distributed.cache import QueryCache
from backend.app.distributed.coordinator import DistributedCoordinator
from backend.app.distributed.merge import merge_results
from backend.app.distributed.router import ShardRouter
from backend.app.retrieval.distributed_retriever import DistributedRetriever


class Worker:
    def __init__(self, score, fail=False):
        self.score = score
        self.fail = fail
        self.calls = 0

    def search(self, query, top_k):
        self.calls += 1
        if self.fail:
            raise RuntimeError("failed")
        return [{"frame_id": f"frame_{self.score}", "score": self.score, "rank": 1}]


def test_routing_and_merge():
    router = ShardRouter(4)
    assert len(router.route("query")) == 1
    assert router.route("query") == router.route("query")
    assert [item["frame_id"] for item in merge_results([[{"frame_id": "a", "score": 0.2}], [{"frame_id": "b", "score": 0.9}]], 1)] == ["b"]


def test_distributed_correctness_cache_and_failure():
    workers = [Worker(0.8), Worker(0.9), Worker(0.1, fail=True)]
    cache = QueryCache()
    coordinator = DistributedCoordinator(workers, ShardRouter(3, "broadcast"), cache=cache)
    retriever = DistributedRetriever(coordinator)
    results = retriever.search("q", 2)
    assert [item["frame_id"] for item in results] == ["frame_0.9", "frame_0.8"]
    assert retriever.search("q", 2) == results
    assert cache.hits == 1
    assert workers[0].calls == 1
