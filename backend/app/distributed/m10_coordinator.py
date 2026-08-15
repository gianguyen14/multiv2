import logging
import os
from concurrent.futures import ThreadPoolExecutor

from backend.app.distributed.m10_cache import M10QueryCache
from backend.app.distributed.m10_merge import merge_normalized
from backend.app.distributed.routing_policy import RoutingPolicy

logger = logging.getLogger(__name__)


class M10Coordinator:
    def __init__(self, workers, router, policy=None, cache=None, index_type="flat"):
        self.workers = list(workers)
        self.router = router
        self.policy = policy or RoutingPolicy()
        self.cache = cache
        self.index_type = index_type
        self.max_workers = min(len(self.workers), os.cpu_count() or 1)

    def search(self, embedding, top_k, latency_budget_ms=None):
        top_n = self.policy.choose_top_n(embedding, top_k, latency_budget_ms)
        if self.router.mode == "broadcast":
            top_n = len(self.workers)
        if self.cache:
            cached = self.cache.get(embedding, top_k, top_n, self.index_type, len(self.workers))
            if cached is not None:
                return cached
        shard_ids = self.router.route(embedding, top_n)
        with ThreadPoolExecutor(max_workers=min(len(shard_ids), self.max_workers)) as executor:
            futures = {executor.submit(self.workers[shard_id].search, embedding, top_k): shard_id for shard_id in shard_ids}
            shard_results = []
            for future, shard_id in futures.items():
                try:
                    shard_results.append(future.result())
                except Exception:
                    logger.warning("Shard %s failed", shard_id, exc_info=True)
        results = merge_normalized(shard_results, top_k)
        if self.cache:
            self.cache.set(embedding, top_k, top_n, self.index_type, len(self.workers), results)
        return results
