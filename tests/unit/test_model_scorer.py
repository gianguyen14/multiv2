import numpy as np
from PIL import Image

from backend.app.retrieval.model_scorer import DeterministicTestScorer
from backend.app.retrieval.siglip_reranker import SigLIPSemanticScorer


class FakeEncoder:
    device = "cpu"

    def __init__(self):
        self.image_batch_sizes = []

    def encode_text(self, texts, batch_size=32):
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    def encode_image(self, images, batch_size=32):
        self.image_batch_sizes.append(batch_size)
        return np.asarray([[index / 10.0, 1.0] for index in range(len(images))], dtype=np.float32)


def test_deterministic_scorer_batch_association_and_backend():
    scorer = DeterministicTestScorer()
    candidates = [{"frame_id": "b", "relevance": 2}, {"frame_id": "a", "relevance": 1}]
    assert scorer.score_batch(None, candidates) == [2.0, 1.0]
    assert scorer.seen_candidate_ids == ["b", "a"]
    assert scorer.backend_name == "deterministic_test_scorer"


def test_siglip_dual_encoder_batches_images_and_names_backend():
    encoder = FakeEncoder()
    scorer = SigLIPSemanticScorer.dual_encoder(encoder, batch_size=2)
    candidates = [{"image": Image.new("RGB", (1, 1))} for _ in range(3)]
    assert scorer.score_batch("query", candidates) == [0.0, 0.10000000149011612, 0.20000000298023224]
    assert encoder.image_batch_sizes == [2]
    assert scorer.backend_name == "siglip2_dual_encoder_similarity"
    assert scorer.capabilities()["joint_pairwise_scoring"] is False
    assert scorer.diagnostics["batch_count"] == 2
