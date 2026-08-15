"""[DEPRECATED / INACTIVE STACK]
This service is a legacy M6 prototype and is NOT used by the active runtime architecture.
The authoritative service is backend.app.services.configured_search.ConfiguredSearch.
"""

from typing import Dict, List, Optional


from PIL import Image

from backend.app.retrieval.hybrid_retriever import HybridSigLIPRetriever


class AdvancedSearchService:
    def __init__(self, retriever: HybridSigLIPRetriever):
        self.retriever = retriever

    def search_text(self, text: str, top_k: int) -> List[Dict]:
        if not text or top_k <= 0:
            return []
        return self.retriever.retrieve_texts([text], top_k)[0]

    def search_image(self, image: Image.Image, top_k: int) -> List[Dict]:
        if image is None or top_k <= 0:
            return []
        return self.retriever.retrieve_images([image], top_k)[0]

    def search_hybrid(
        self,
        text: Optional[str],
        image: Optional[Image.Image],
        top_k: int,
    ) -> List[Dict]:
        texts = [text] if text else None
        images = [image] if image is not None else None
        results = self.retriever.retrieve_hybrid(texts, images, top_k)
        return results[0] if results else []
