from typing import Dict, List

import numpy as np


class SimpleCosineReranker:
    def rerank(
        self, query_embedding: np.ndarray, candidates: List[Dict], top_k: int
    ) -> List[Dict]:
        if top_k <= 0 or not candidates:
            return []
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return candidates[:top_k]
        scored = []
        for candidate in candidates:
            embedding = np.asarray(candidate["embedding"], dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(embedding)
            score = float(np.dot(query, embedding) / (query_norm * norm)) if norm else 0.0
            result = {key: value for key, value in candidate.items() if key != "embedding"}
            result["score"] = score
            scored.append(result)
        scored.sort(key=lambda item: item["score"], reverse=True)
        for rank, result in enumerate(scored[:top_k], start=1):
            result["rank"] = rank
        return scored[:top_k]
