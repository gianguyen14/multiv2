import numpy as np


class MultiCentroidIndex:
    def __init__(self, shard_embeddings=None, k=4):
        self.k = k
        self.centroids = {}
        if shard_embeddings:
            for shard_id, embeddings in shard_embeddings.items():
                self.add_shard(shard_id, embeddings)

    def add_shard(self, shard_id, embeddings):
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) == 0:
            raise ValueError("embeddings must be a non-empty matrix")
        chunks = np.array_split(vectors, min(self.k, len(vectors)))
        shard_centroids = []
        for chunk in chunks:
            centroid = chunk.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm:
                shard_centroids.append(centroid / norm)
        self.centroids[int(shard_id)] = np.asarray(shard_centroids, dtype=np.float32)

    def search(self, query_embedding, top_n=1):
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(query)
        if norm == 0:
            return sorted(self.centroids)[:top_n]
        query = query / norm
        scores = [(float((centroids @ query).max()), shard_id) for shard_id, centroids in self.centroids.items()]
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [shard_id for _, shard_id in scores[:max(1, min(top_n, len(scores)))]]
