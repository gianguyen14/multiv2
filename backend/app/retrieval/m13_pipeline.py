import time

from backend.app.retrieval.candidate_resolver import resolve_candidates
from backend.app.retrieval.hybrid_ranker import HybridRanker
from backend.app.retrieval.model_scorer import validate_scores


class M13RetrievalPipeline:
    def __init__(self, encoder, retriever, candidate_resolver, model_scorer, candidate_k=50, final_k=10, hybrid_ranker=None):
        if candidate_k <= 0 or final_k <= 0:
            raise ValueError("candidate_k and final_k must be positive")
        self.encoder = encoder
        self.retriever = retriever
        self.candidate_resolver = candidate_resolver
        self.model_scorer = model_scorer
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.hybrid_ranker = hybrid_ranker or HybridRanker()
        self.last_diagnostics = {}

    @staticmethod
    def _identifier(candidate):
        return str(candidate.get("candidate_id", candidate.get("frame_id", "")))

    def search_text(self, text, candidate_k=None, final_k=None, top_n=None, include_diagnostics=False, ranking="model"):
        started = time.perf_counter()
        encoding_start = time.perf_counter()
        query_embedding = self.encoder.encode_text([text])[0]
        encoding_ms = (time.perf_counter() - encoding_start) * 1000
        candidate_k = candidate_k or self.candidate_k
        final_k = final_k or self.final_k
        retrieval_start = time.perf_counter()
        if top_n is None:
            candidates = self.retriever.search(query_embedding, candidate_k)
        else:
            candidates = self.retriever.search(query_embedding, candidate_k, top_n=top_n)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        candidates = [dict(candidate) for candidate in candidates]
        candidate_ids = [self._identifier(candidate) for candidate in candidates]
        resolution_start = time.perf_counter()
        try:
            resolved, missing_ids = resolve_candidates(candidates, self.candidate_resolver)
        except Exception as exc:
            resolved, missing_ids = [], candidate_ids
            resolution_error = type(exc).__name__
        else:
            resolution_error = None
        resolution_ms = (time.perf_counter() - resolution_start) * 1000
        fallback_used = False
        fallback_reason = resolution_error
        model_start = time.perf_counter()
        try:
            if missing_ids:
                raise ValueError("candidate_payload_unavailable")
            scores = validate_scores(self.model_scorer.score_batch(text, resolved), len(resolved))
            scored = []
            for candidate, score in zip(resolved, scores):
                row = dict(candidate)
                row.setdefault("retrieval_score", float(row.get("score", 0.0)))
                row["semantic_score"] = float(score)
                row["score"] = float(score)
                scored.append(row)
            scored.sort(key=lambda item: (-item["score"], self._identifier(item)))
            if ranking == "hybrid":
                scored = self.hybrid_ranker.rank(scored)
            results = scored[:final_k]
            for rank, result in enumerate(results, start=1):
                result["rank"] = rank
        except Exception as exc:
            fallback_used = True
            fallback_reason = type(exc).__name__
            results = candidates[:final_k]
            for rank, result in enumerate(results, start=1):
                result["rank"] = rank
        model_ms = (time.perf_counter() - model_start) * 1000
        scorer_diagnostics = dict(getattr(self.model_scorer, "diagnostics", {}))
        diagnostics = {
            "candidate_count": len(candidates),
            "resolved_candidate_count": len(resolved),
            "unresolved_candidate_count": len(missing_ids),
            "candidate_ids": candidate_ids,
            "final_count": len(results),
            "model_backend": getattr(self.model_scorer, "backend_name", "unknown"),
            "encoding_time_ms": encoding_ms,
            "retrieval_time_ms": retrieval_ms,
            "payload_resolution_time_ms": resolution_ms,
            "rerank_model_time_ms": model_ms,
            "total_rerank_time_ms": resolution_ms + model_ms,
            "total_time_ms": (time.perf_counter() - started) * 1000,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            **scorer_diagnostics,
        }
        self.last_diagnostics = diagnostics
        return (results, diagnostics) if include_diagnostics else results
