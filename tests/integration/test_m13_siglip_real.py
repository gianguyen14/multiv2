import os

import pytest
from PIL import Image

from backend.app.embeddings.siglip2 import SigLIP2Encoder
from backend.app.retrieval.siglip_reranker import SigLIPSemanticScorer


pytestmark = [
    pytest.mark.slow,
    pytest.mark.real_model,
    pytest.mark.skipif(os.getenv("RUN_M13_REAL_MODEL") != "1", reason="set RUN_M13_REAL_MODEL=1 when SigLIP2 weights are locally available"),
]


def test_real_siglip_dual_encoder_scores_local_images():
    encoder = SigLIP2Encoder(force_download=False)
    scorer = SigLIPSemanticScorer.dual_encoder(encoder, batch_size=2)
    candidates = [{"image": Image.new("RGB", (224, 224), color)} for color in ("red", "blue")]
    scores = scorer.score_batch("a red image", candidates)
    assert len(scores) == 2
    assert scorer.capabilities() == {"dual_encoder": True, "joint_pairwise_scoring": False, "supports_batch_text": True, "supports_batch_image": True}
