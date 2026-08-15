import numpy as np


class RoutingPolicy:
    def __init__(self, default_top_n=2):
        self.default_top_n = default_top_n

    def choose_top_n(self, query_embedding, top_k, latency_budget_ms=None):
        if latency_budget_ms is not None and latency_budget_ms < 5:
            return 1
        if top_k <= 5:
            return max(2, self.default_top_n)
        embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        variance = float(np.var(np.abs(embedding)))
        return min(3, max(1, self.default_top_n + (1 if variance > 0.01 else 0)))
