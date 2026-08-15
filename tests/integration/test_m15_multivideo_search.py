import numpy as np

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.video.frame_index import load_current_frame_index
from backend.app.video.ingest import ingest_path
from tests.m15_support import MeanRGBEncoder, create_video


def test_multivideo_search_resolves_original_frames(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    create_video(corpus / "synthetic_red_car.mp4", "red_car")
    create_video(corpus / "synthetic_blue_object.mp4", "blue_object")
    create_video(corpus / "synthetic_white_cup.mp4", "white_cup")
    output = tmp_path / "processed"
    encoder = MeanRGBEncoder()
    report = ingest_path(corpus, encoder, VideoIngestConfig(processed_root=output))
    assert report["videos_succeeded"] == 3 and report["videos_failed"] == 0
    bundle = load_current_frame_index(output / "index")
    assert bundle.index.index.ntotal == 6
    assert len(bundle.resolver.payloads) == len(set(bundle.resolver.payloads)) == 6
    query = np.array([1, 0, 0, 0], dtype=np.float32)
    hit = bundle.index.search(query, 1)[0]
    payload = bundle.resolver.resolve(hit["frame_id"])
    assert payload["video_id"] in {"synthetic_red_car", "synthetic_blue_object", "synthetic_white_cup"}
    assert payload["source_frame_index_zero_based"] in {0, 4}
    assert payload["timestamp_seconds"] is not None
