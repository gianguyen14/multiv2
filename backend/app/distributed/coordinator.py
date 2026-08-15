import logging
import os
from concurrent.futures import ThreadPoolExecutor

from backend.app.distributed.cache import QueryCache
from backend.app.distributed.merge import merge_results
from backend.app.distributed.router import ShardRouter

logger = logging.getLogger(__name__)


class DistributedCoordinator:
    def __init__(self, workers, router=None, cache=None, max_workers=None):
        self.workers = list(workers)
        self.router = router or ShardRouter(len(self.workers))
        self.cache = cache
        self.max_workers = max_workers or min(len(self.workers), os.cpu_count() or 1)

    def search(self, query, top_k, top_n=None):
        if top_k <= 0:
            return []
        top_n = top_n or 1
        if self.cache:
            cached = self.cache.get(query, top_k, top_n)
            if cached is not None:
                return cached
        shard_ids = self.router.route(query, top_n=top_n)
        if len(shard_ids) == 1:
            try:
                results = self.workers[shard_ids[0]].search(query, top_k)
            except Exception:
                logger.warning("Shard %s failed", shard_ids[0], exc_info=True)
                results = []
        else:
            with ThreadPoolExecutor(max_workers=min(len(shard_ids), self.max_workers)) as executor:
                futures = {executor.submit(self.workers[i].search, query, top_k): i for i in shard_ids}
                shard_results = []
                for future, shard_id in futures.items():
                    try:
                        shard_results.append(future.result())
                    except Exception:
                        logger.warning("Shard %s failed", shard_id, exc_info=True)
                results = merge_results(shard_results, top_k)
        if self.cache:
            self.cache.set(query, top_k, results, top_n)
        return results
