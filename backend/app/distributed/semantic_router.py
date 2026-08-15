import numpy as np

from backend.app.distributed.centroid_index import ShardCentroidIndex
from backend.app.distributed.router import ShardRouter


class SemanticShardRouter:
    def __init__(self, shard_count, centroids=None, mode="single"):
        self.shard_count = shard_count
        self.mode = mode
        self.centroid_index = ShardCentroidIndex(centroids)
        self.hash_router = ShardRouter(shard_count)

    def route(self, embedding, top_n=1):
        if self.mode == "broadcast":
            return list(range(self.shard_count))
        count = 1 if self.mode == "single" else top_n
        if not self.centroid_index.centroids:
            return self.hash_router.route_embedding(embedding, count)
        return self.centroid_index.search(embedding, count)
