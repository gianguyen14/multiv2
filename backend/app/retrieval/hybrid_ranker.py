from collections import defaultdict
from typing import Mapping, Optional

import numpy as np


class HybridRanker:
    def __init__(self, weights: Optional[Mapping[str, float]] = None):
        weights = dict(weights or {"retrieval_score": 0.4, "semantic_score": 0.6})
        if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("weights must be non-negative and have a positive sum")
        total = sum(weights.values())
        self.weights = {key: value / total for key, value in weights.items()}

    @staticmethod
    def _id(candidate: dict) -> str:
        return str(candidate.get("candidate_id", candidate.get("document_id", candidate.get("frame_id", ""))))

    def rank(self, candidates, top_k=None):
        dedup = {}
        for candidate in candidates:
            candidate_id = self._id(candidate)
            retrieval = float(candidate.get("retrieval_score", candidate.get("score", 0.0)))
            if candidate_id not in dedup or retrieval > float(dedup[candidate_id].get("retrieval_score", dedup[candidate_id].get("score", 0.0))):
                dedup[candidate_id] = dict(candidate)
        rows = list(dedup.values())
        normalized = defaultdict(dict)
        for signal in self.weights:
            values = np.asarray([float(row.get(signal, row.get("score", 0.0) if signal == "retrieval_score" else 0.0)) for row in rows])
            mean = float(values.mean()) if len(values) else 0.0
            std = float(values.std()) if len(values) else 0.0
            for index, value in enumerate(values):
                normalized[index][signal] = (float(value) - mean) / std if std else 0.0
        output = []
        for index, row in enumerate(rows):
            result = dict(row)
            result["score"] = sum(self.weights[signal] * normalized[index][signal] for signal in self.weights)
            result["normalized_scores"] = dict(normalized[index])
            output.append(result)
        output.sort(key=lambda item: (-item["score"], self._id(item)))
        if top_k is not None:
            output = output[:max(0, top_k)]
        for rank, result in enumerate(output, start=1):
            result["rank"] = rank
        return output
