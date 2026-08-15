import time

from backend.app.retrieval.candidate_resolver import resolve_candidates
from backend.app.retrieval.hybrid_ranker import HybridRanker
from backend.app.retrieval.model_scorer import validate_scores


class M14RetrievalPipeline:
    def __init__(self, encoder, retriever, candidate_resolver, relevance_scorer, candidate_k=10, final_k=10, rerank_weight=0.6):
        if candidate_k <= 0 or final_k <= 0 or not 0 <= rerank_weight <= 1:
            raise ValueError("invalid M14 pipeline configuration")
        self.encoder = encoder
        self.retriever = retriever
        self.candidate_resolver = candidate_resolver
        self.relevance_scorer = relevance_scorer
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.hybrid_ranker = HybridRanker({"retrieval_score": 1 - rerank_weight, "m14_relevance_score": rerank_weight})
        self.last_diagnostics = {}

    @staticmethod
    def _id(row):
        return str(row.get("candidate_id", row.get("frame_id", "")))

    def search_text(self, text, candidate_k=None, final_k=None, ranking="m14", include_diagnostics=False):
        started = time.perf_counter()
        candidate_k = self.candidate_k if candidate_k is None else candidate_k
        final_k = self.final_k if final_k is None else final_k
        query_embedding = self.encoder.encode_text([text])[0]
        candidates = [dict(row) for row in self.retriever.search(query_embedding, candidate_k)]
        candidate_ids = [self._id(row) for row in candidates]
        resolution_start = time.perf_counter()
        try:
            resolved, missing = resolve_candidates(candidates, self.candidate_resolver)
        except Exception as exc:
            resolved, missing, error = [], candidate_ids, exc
        else:
            error = None
        resolution_ms = (time.perf_counter() - resolution_start) * 1000
        fallback = False
        reason = type(error).__name__ if error else None
        fusion_ms = 0.0
        try:
            if error:
                raise error
            if missing:
                raise ValueError("candidate payload unavailable")
            scores = validate_scores(self.relevance_scorer.score_batch(text, resolved), len(resolved))
            if [self._id(row) for row in resolved] != candidate_ids:
                raise AssertionError("same-candidate invariant violated")
            scored = []
            for order, (row, score) in enumerate(zip(resolved, scores)):
                result = dict(row)
                result["retrieval_score"] = float(result.get("score", 0.0))
                result["m14_relevance_score"] = score
                result["_faiss_order"] = order
                scored.append(result)
            fusion_start = time.perf_counter()
            if ranking == "hybrid":
                results = self.hybrid_ranker.rank(scored)
            else:
                results = sorted(scored, key=lambda row: (-row["m14_relevance_score"], row["_faiss_order"]))
                for result in results:
                    result["score"] = result["m14_relevance_score"]
            fusion_ms = (time.perf_counter() - fusion_start) * 1000
            results = results[:final_k]
            for rank, result in enumerate(results, 1):
                result.pop("_faiss_order", None)
                result["rank"] = rank
        except Exception as exc:
            fallback = True
            reason = type(exc).__name__
            results = candidates[:final_k]
            for rank, result in enumerate(results, 1):
                result["rank"] = rank
        scorer_diagnostics = dict(getattr(self.relevance_scorer, "diagnostics", {}))
        diagnostics = {
            "reranker_backend": getattr(self.relevance_scorer, "backend_name", "unknown"),
            "signal_type": getattr(self.relevance_scorer, "signal_type", "unknown"),
            "candidate_count": len(candidates),
            "resolved_candidate_count": len(resolved),
            "candidate_ids": candidate_ids,
            "payload_resolution_ms": resolution_ms,
            "fusion_ms": fusion_ms,
            "total_rerank_ms": (time.perf_counter() - started) * 1000,
            "fallback_used": fallback,
            "fallback_reason": reason,
            **scorer_diagnostics,
        }
        self.last_diagnostics = diagnostics
        return (results, diagnostics) if include_diagnostics else results
