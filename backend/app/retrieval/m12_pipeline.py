import time


class M12RetrievalPipeline:
    def __init__(self, encoder, retriever, reranker, candidate_resolver=None, candidate_k=50, final_k=10):
        if candidate_k <= 0 or final_k <= 0:
            raise ValueError("candidate_k and final_k must be positive")
        self.encoder = encoder
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_resolver = candidate_resolver
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.last_diagnostics = {}

    def search_text(self, text, candidate_k=None, final_k=None, top_n=None, include_diagnostics=False):
        started = time.perf_counter()
        query = self.encoder.encode_text([text])[0]
        candidate_k = candidate_k or self.candidate_k
        final_k = final_k or self.final_k
        retrieval_start = time.perf_counter()
        if top_n is None:
            candidates = self.retriever.search(query, candidate_k)
        else:
            candidates = self.retriever.search(query, candidate_k, top_n=top_n)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        if self.candidate_resolver is not None:
            candidates = self.candidate_resolver(candidates)
        results = self.reranker.rerank(query, candidates, final_k)
        diagnostics = {
            "candidate_count": len(candidates),
            "final_count": len(results),
            "retrieval_time_ms": retrieval_ms,
            "total_time_ms": (time.perf_counter() - started) * 1000,
            **self.reranker.last_diagnostics,
        }
        self.last_diagnostics = diagnostics
        return (results, diagnostics) if include_diagnostics else results
