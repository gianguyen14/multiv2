from typing import Dict, List, Optional

import numpy as np


class CrossEncoderReranker:
    def __init__(self, model=None):
        self.model = model

    def rerank(self, query, candidates: List[Dict], top_k: int) -> List[Dict]:
        if top_k <= 0:
            return []
        scored = []
        for candidate in candidates:
            embedding = candidate.get("embedding")
            if self.model is not None:
                score = float(self.model(query, candidate))
            elif embedding is not None:
                score = float(np.dot(np.asarray(query).reshape(-1), np.asarray(embedding).reshape(-1)))
            else:
                score = float(candidate.get("score", 0.0))
            result = dict(candidate)
            result.pop("embedding", None)
            result["score"] = score
            scored.append(result)
        scored.sort(key=lambda item: item["score"], reverse=True)
        for rank, result in enumerate(scored[:top_k], start=1):
            result["rank"] = rank
        return scored[:top_k]
