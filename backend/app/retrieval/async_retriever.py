from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np

from backend.app.retrieval.batch_retriever import BatchFaissRetriever


class AsyncFaissRetriever:
    def __init__(self, batch_retriever: BatchFaissRetriever, max_workers: int = 4, batch_size: int = 8):
        self.batch_retriever = batch_retriever
        self.max_workers = max_workers
        self.batch_size = batch_size

    def search(self, embeddings: np.ndarray, top_k: int) -> List[List[Dict]]:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (N, D)")
        if len(embeddings) < 16:
            return [self.batch_retriever.index.search(embedding, top_k) for embedding in embeddings]
        if len(embeddings) < 32:
            return self.batch_retriever.search(embeddings, top_k)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.batch_retriever.index.search, embedding, top_k)
                for embedding in embeddings
            ]
            return [future.result() for future in futures]


AsyncRetriever = AsyncFaissRetriever
