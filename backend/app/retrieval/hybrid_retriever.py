from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from backend.app.retrieval.reranker import SimpleCosineReranker
from backend.app.retrieval.retriever import SigLIPFaissRetriever


class HybridSigLIPRetriever:
    def __init__(
        self,
        retriever: SigLIPFaissRetriever,
        reranker: Optional[SimpleCosineReranker] = None,
        alpha: float = 0.5,
        expand_factor: int = 3,
    ):
        self.retriever = retriever
        self.reranker = reranker or SimpleCosineReranker()
        self.alpha = alpha
        self.expand_factor = expand_factor

    def _candidates(self, embedding: np.ndarray, top_k: int) -> List[Dict]:
        raw = self.retriever.index.search(embedding.reshape(-1), top_k * self.expand_factor)
        candidates = []
        for result in raw:
            vector_id = next(
                key for key, value in self.retriever.index.frame_id_mapping.items()
                if value == result["frame_id"]
            )
            candidate = dict(result)
            candidate["embedding"] = self.retriever.index.index.reconstruct(int(vector_id))
            candidates.append(candidate)
        return self.reranker.rerank(embedding, candidates, top_k)

    def retrieve_texts(self, texts: List[str], top_k: int) -> List[List[Dict]]:
        if not texts or top_k <= 0:
            return [[] for _ in texts]
        embeddings = self.retriever.encoder.encode_text(texts)
        return [self._candidates(embedding, top_k) for embedding in embeddings]

    def retrieve_images(self, images: List[Image.Image], top_k: int) -> List[List[Dict]]:
        if not images or top_k <= 0:
            return [[] for _ in images]
        embeddings = self.retriever.encoder.encode_image(images)
        return [self._candidates(embedding, top_k) for embedding in embeddings]

    def retrieve_hybrid(
        self,
        texts: Optional[List[str]],
        images: Optional[List[Image.Image]],
        top_k: int,
    ) -> List[List[Dict]]:
        if top_k <= 0:
            return []
        text_embeddings = self.retriever.encoder.encode_text(texts) if texts else None
        image_embeddings = self.retriever.encoder.encode_image(images) if images else None
        if text_embeddings is None and image_embeddings is None:
            return []
        count = len(text_embeddings) if text_embeddings is not None else len(image_embeddings)
        results = []
        for i in range(count):
            text = text_embeddings[i] if text_embeddings is not None else None
            image = image_embeddings[i] if image_embeddings is not None else None
            embedding = text if image is None else image if text is None else self.alpha * text + (1 - self.alpha) * image
            results.append(self._candidates(embedding.astype(np.float32), top_k))
        return results
