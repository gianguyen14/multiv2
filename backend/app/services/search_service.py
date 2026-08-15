"""[DEPRECATED / INACTIVE STACK]
This service is a legacy M5 prototype and is NOT used by the active runtime architecture.
The authoritative service is backend.app.services.configured_search.ConfiguredSearch.
"""

from typing import Dict, List


from PIL import Image

from backend.app.retrieval.retriever import SigLIPFaissRetriever


class SearchService:
    def __init__(self, retriever: SigLIPFaissRetriever, semantic_pipeline=None, enable_semantic_reranker: bool = False):
        self.retriever = retriever
        self.semantic_pipeline = semantic_pipeline
        self.enable_semantic_reranker = enable_semantic_reranker

    def search_by_text(self, text: str, top_k: int) -> List[Dict]:
        if not isinstance(text, str) or not text or top_k <= 0:
            return []
        if self.enable_semantic_reranker and self.semantic_pipeline is not None:
            return self.semantic_pipeline.search_text(text, final_k=top_k)
        return self.retriever.retrieve_texts([text], top_k)[0]

    def search_by_image(self, image: Image.Image, top_k: int) -> List[Dict]:
        if not isinstance(image, Image.Image) or top_k <= 0:
            return []
        return self.retriever.retrieve_images([image], top_k)[0]
