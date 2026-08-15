import math
import os

import pytest

from backend.app.retrieval.ninerouter_vision_scorer import NineRouterVisionScorer


pytestmark = [
    pytest.mark.slow,
    pytest.mark.real_model,
    pytest.mark.network,
    pytest.mark.skipif(os.getenv("RUN_M14_REAL_MODEL") != "1", reason="set RUN_M14_REAL_MODEL=1 for the local 9Router smoke test"),
]


def test_real_9router_model_accepts_image_and_returns_relevance_score():
    scorer = NineRouterVisionScorer()
    scores = scorer.score_batch("red circle on white background", [{"candidate_id": "img_001", "image_path": "eval/data/m13_5/images/img_001.png"}])
    assert len(scores) == 1 and math.isfinite(scores[0]) and 0 <= scores[0] <= 100
    assert scorer.diagnostics["requested_model"] == "cx/gpt-5.6-sol"
