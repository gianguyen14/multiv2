import numpy as np

from backend.app.distributed.m11_centroid_index import MultiCentroidIndex
from backend.app.distributed.router import ShardRouter


class M11SemanticRouter:
    def __init__(self, shard_count, shard_embeddings=None, mode="single", k=4):
        self.shard_count = shard_count
        self.mode = mode
        self.centroids = MultiCentroidIndex(shard_embeddings, k=k)
        self.fallback = ShardRouter(shard_count)

    def route(self, embedding, top_n=1):
        if self.mode == "broadcast":
            return list(range(self.shard_count))
        count = 1 if self.mode == "single" else top_n
        if not self.centroids.centroids:
            return self.fallback.route_embedding(embedding, count)
        return self.centroids.search(embedding, count)
