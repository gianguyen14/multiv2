import numpy as np


class ShardCentroidIndex:
    def __init__(self, shard_embeddings=None):
        self.centroids = {}
        if shard_embeddings:
            for shard_id, embeddings in shard_embeddings.items():
                self.add_shard(shard_id, embeddings)

    def add_shard(self, shard_id, embeddings):
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty matrix")
        centroid = vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            raise ValueError("shard centroid must be non-zero")
        self.centroids[int(shard_id)] = centroid / norm

    def search(self, query_embedding, top_n=1):
        if not self.centroids:
            return []
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(query)
        if norm == 0:
            return sorted(self.centroids)[:top_n]
        scores = [(float(np.dot(query / norm, centroid)), shard_id) for shard_id, centroid in self.centroids.items()]
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [shard_id for _, shard_id in scores[:max(1, min(top_n, len(scores)))]]
