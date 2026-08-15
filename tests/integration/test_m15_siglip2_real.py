import os

import numpy as np
import pytest

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.embeddings.siglip2 import SigLIP2Encoder
from backend.app.video.frame_index import load_current_frame_index
from backend.app.video.ingest import ingest_path
from tests.m15_support import create_video


pytestmark = [
    pytest.mark.slow,
    pytest.mark.real_model,
    pytest.mark.skipif(os.getenv("RUN_M15_REAL_MODEL") != "1",
        reason="set RUN_M15_REAL_MODEL=1 when SigLIP2 weights are locally available"),
]


def test_real_siglip2_multivideo_search_mapping(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    create_video(corpus / "synthetic_red_car.mp4", "red_car")
    create_video(corpus / "synthetic_blue_object.mp4", "blue_object")
    create_video(corpus / "synthetic_white_cup.mp4", "white_cup")
    output = tmp_path / "processed"
    encoder = SigLIP2Encoder(device="cpu", force_download=False)
    report = ingest_path(corpus, encoder, VideoIngestConfig(processed_root=output, embed_batch_size=8))
    assert report["videos_succeeded"] == 3 and report["indexed_frames"] == 6
    bundle = load_current_frame_index(output / "index")
    query = encoder.encode_text(["red car on a road"])[0]
    hits = bundle.index.search(query, 6)
    assert hits
    resolved = [bundle.resolver.resolve(hit["frame_id"]) for hit in hits]
    assert all(payload is not None for payload in resolved)
    assert all(payload["video_id"] in {"synthetic_red_car", "synthetic_blue_object", "synthetic_white_cup"} for payload in resolved)
    assert all(payload["source_frame_index_zero_based"] in {0, 4} for payload in resolved)
    assert len({payload["frame_uid"] for payload in resolved}) == 6
    for video_id in {payload["video_id"] for payload in resolved}:
        embeddings = np.load(output / video_id / "embeddings.npy")
        assert embeddings.shape == (2, 768)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
