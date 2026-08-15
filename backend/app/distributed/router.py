import hashlib

import numpy as np


class ShardRouter:
    def __init__(self, shard_count, strategy="hash"):
        if shard_count <= 0:
            raise ValueError("shard_count must be positive")
        if strategy not in ("hash", "broadcast"):
            raise ValueError("strategy must be 'hash' or 'broadcast'")
        self.shard_count = shard_count
        self.strategy = strategy

    def route(self, query, top_n=1):
        if self.strategy == "broadcast":
            return list(range(self.shard_count))
        top_n = max(1, min(int(top_n), self.shard_count))
        if isinstance(query, np.ndarray):
            key = np.asarray(query).tobytes()
        elif isinstance(query, bytes):
            key = query
        else:
            key = str(query).encode("utf-8")
        digest = int(hashlib.sha256(key).hexdigest(), 16)
        base = digest % self.shard_count
        return [(base + offset) % self.shard_count for offset in range(top_n)]

    def route_embedding(self, embedding, top_n=1):
        return self.route(np.asarray(embedding), top_n=top_n)

    def route_hybrid(self, embedding, top_n=2):
        return self.route_embedding(embedding, top_n=top_n)

    def route_broadcast(self, embedding):
        return list(range(self.shard_count))
