"""[DEPRECATED / INACTIVE STACK]
This retriever is a legacy M5 prototype and is NOT used by the active runtime architecture.
The authoritative search layer is backend.app.services.configured_search.ConfiguredSearch.
"""

from typing import Dict, List


import numpy as np
from PIL import Image

from backend.app.embeddings.siglip2 import SigLIP2Encoder
from backend.app.indexes.faiss_siglip_index import FaissSigLIPIndex


class SigLIPFaissRetriever:
    def __init__(self, index: FaissSigLIPIndex, encoder: SigLIP2Encoder):
        self.index = index
        self.encoder = encoder

    def retrieve_images(
        self, images: List[Image.Image], top_k: int
    ) -> List[List[Dict]]:
        if not images or top_k <= 0:
            return [[] for _ in images]
        embeddings = self.encoder.encode_image(images)
        return [self.index.search(vector, top_k) for vector in embeddings]

    def retrieve_texts(self, texts: List[str], top_k: int) -> List[List[Dict]]:
        if not texts or top_k <= 0:
            return [[] for _ in texts]
        embeddings = self.encoder.encode_text(texts)
        return [self.index.search(vector, top_k) for vector in embeddings]
