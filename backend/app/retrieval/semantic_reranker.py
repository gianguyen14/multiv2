import time
from typing import Any, Callable, Iterable, Optional

import numpy as np


class SemanticReranker:
    def __init__(self, scorer: Optional[Callable] = None, batch_scorer: Optional[Callable] = None, backend_name: Optional[str] = None):
        self.scorer = scorer
        self.batch_scorer = batch_scorer
        self.backend_name = backend_name or ("semantic_callback" if scorer or batch_scorer else "dot_product_baseline")
        self.last_diagnostics = {}

    @staticmethod
    def _id(candidate: dict) -> str:
        return str(candidate.get("candidate_id", candidate.get("document_id", candidate.get("frame_id", ""))))

    @staticmethod
    def _baseline_score(query: Any, candidate: dict) -> float:
        embedding = candidate.get("embedding")
        if embedding is None:
            return float(candidate.get("retrieval_score", candidate.get("score", 0.0)))
        query_vector = np.asarray(query, dtype=np.float32).reshape(-1)
        candidate_vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if query_vector.shape != candidate_vector.shape:
            return float(candidate.get("retrieval_score", candidate.get("score", 0.0)))
        return float(np.dot(query_vector, candidate_vector))

    def rerank(self, query: Any, candidates: Iterable[dict], top_k: Optional[int] = None) -> list[dict]:
        candidates = [dict(candidate) for candidate in candidates]
        if top_k is not None and top_k <= 0:
            return []
        started = time.perf_counter()
        fallback_used = False
        backend = self.backend_name
        try:
            if self.batch_scorer is not None:
                scores = list(self.batch_scorer(query, candidates))
                if len(scores) != len(candidates):
                    raise ValueError("batch scorer returned the wrong number of scores")
            elif self.scorer is not None:
                scores = [self.scorer(query, candidate) for candidate in candidates]
            else:
                scores = [self._baseline_score(query, candidate) for candidate in candidates]
                fallback_used = True
        except Exception:
            scores = [float(candidate.get("retrieval_score", candidate.get("score", 0.0))) for candidate in candidates]
            fallback_used = True
            backend = "retrieval_order_fallback"
        scored = []
        for index, (candidate, score) in enumerate(zip(candidates, scores)):
            result = dict(candidate)
            result.setdefault("retrieval_score", float(candidate.get("score", 0.0)))
            result["semantic_score"] = float(score)
            result["score"] = float(score)
            result["_input_order"] = index
            scored.append(result)
        scored.sort(key=lambda item: (-item["score"], self._id(item), item["_input_order"]))
        limit = len(scored) if top_k is None else min(top_k, len(scored))
        output = scored[:limit]
        for rank, result in enumerate(output, start=1):
            result.pop("_input_order", None)
            result["rank"] = rank
        self.last_diagnostics = {
            "candidate_count": len(candidates),
            "final_count": len(output),
            "rerank_time_ms": (time.perf_counter() - started) * 1000,
            "reranker_backend": backend,
            "fallback_used": fallback_used,
        }
        return output
