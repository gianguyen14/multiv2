from typing import Dict, List

from PIL import Image

from backend.app.retrieval.retriever import SigLIPFaissRetriever


class SearchService:
    def __init__(self, retriever: SigLIPFaissRetriever):
        self.retriever = retriever

    def search_images(
        self, images: List[Image.Image], top_k: int
    ) -> List[List[Dict]]:
        return self.retriever.search_by_image(images, top_k)

    def search_text(self, texts: List[str], top_k: int) -> List[List[Dict]]:
        return self.retriever.search_by_text(texts, top_k)
