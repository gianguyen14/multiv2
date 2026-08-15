from typing import Callable, Dict, Optional

import numpy as np

from backend.app.retrieval.reranker import SimpleCosineReranker


class RetrievalPipeline:
    def __init__(self, encoder, retriever, planner=None, reranker=None, use_reranker=False):
        self.encoder = encoder
        self.retriever = retriever
        self.planner = planner
        self.reranker = reranker or SimpleCosineReranker()
        self.use_reranker = use_reranker

    def search_text(self, text: str, top_k: int, metadata_filter: Optional[Callable[[Dict], bool]] = None):
        embedding = self.encoder.encode_text([text])[0]
        results = self.retriever.search(embedding, top_k * 2 if self.use_reranker else top_k)
        if metadata_filter:
            results = [result for result in results if metadata_filter(result)]
        if self.use_reranker:
            candidates = [dict(result, embedding=embedding) for result in results]
            results = self.reranker.rerank(embedding, candidates, top_k)
        return results[:top_k]
