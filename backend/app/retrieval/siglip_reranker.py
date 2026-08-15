import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from backend.app.retrieval.model_scorer import ModelScorer
from backend.app.retrieval.semantic_reranker import SemanticReranker


class SigLIPSemanticScorer(ModelScorer):
    def __init__(self, encoder=None, model_scorer: Optional[Callable] = None, batch_model_scorer: Optional[Callable] = None, batch_size: int = 16):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.encoder = encoder
        self.model_scorer = model_scorer
        self.batch_model_scorer = batch_model_scorer
        self.batch_size = batch_size
        self.backend_name = "siglip_model_scorer" if self.is_model_backed else "siglip_embedding_dot_product_baseline"
        self._diagnostics = {}

    @classmethod
    def dual_encoder(cls, encoder, batch_size: int = 16):
        scorer = cls(encoder=encoder, batch_size=batch_size)
        scorer.backend_name = "siglip2_dual_encoder_similarity"
        return scorer

    @staticmethod
    def capabilities() -> dict:
        return {
            "dual_encoder": True,
            "joint_pairwise_scoring": False,
            "supports_batch_text": True,
            "supports_batch_image": True,
        }

    def score(self, query, candidate) -> float:
        if self.model_scorer is not None:
            return float(self.model_scorer(query, candidate))
        embedding = candidate.get("embedding")
        if embedding is None:
            return float(candidate.get("retrieval_score", candidate.get("score", 0.0)))
        return float(np.dot(np.asarray(query).reshape(-1), np.asarray(embedding).reshape(-1)))

    def _candidate_images(self, candidates):
        images = []
        for candidate in candidates:
            image = candidate.get("image", candidate.get("image_path", candidate.get("path")))
            if image is None:
                raise ValueError("candidate image payload is required")
            if isinstance(image, str):
                image = Path(image)
            images.append(image)
        return images

    def score_batch(self, query, candidates):
        candidates = list(candidates)
        started = time.perf_counter()
        if self.batch_model_scorer is not None:
            scores = list(self.batch_model_scorer(query, candidates))
        elif self.model_scorer is not None:
            scores = [self.model_scorer(query, candidate) for candidate in candidates]
        elif self.backend_name == "siglip2_dual_encoder_similarity":
            query_embedding = self.encoder.encode_text([query], batch_size=1)[0]
            candidate_embeddings = self.encoder.encode_image(self._candidate_images(candidates), batch_size=self.batch_size)
            scores = (candidate_embeddings @ query_embedding).astype(np.float32).tolist()
        else:
            scores = [self.score(query, candidate) for candidate in candidates]
        self._diagnostics = {
            "model_backend": self.backend_name,
            "model_device": getattr(self.encoder, "device", None),
            "batch_size": self.batch_size,
            "batch_count": math.ceil(len(candidates) / self.batch_size) if candidates else 0,
            "model_inference_time_ms": (time.perf_counter() - started) * 1000,
        }
        return [float(score) for score in scores]

    @property
    def diagnostics(self) -> dict:
        return dict(self._diagnostics)

    @property
    def is_model_backed(self) -> bool:
        return self.model_scorer is not None or self.batch_model_scorer is not None


def create_siglip_reranker(encoder=None, model_scorer=None, batch_model_scorer=None, batch_size=16) -> SemanticReranker:
    scorer = SigLIPSemanticScorer(encoder, model_scorer, batch_model_scorer, batch_size=batch_size)
    return SemanticReranker(batch_scorer=scorer.score_batch, backend_name=scorer.backend_name)
