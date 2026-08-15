from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.video.frame_index import load_current_frame_index
from backend.app.video.ingest import ingest_path
from tests.m15_support import MeanRGBEncoder, create_video


def test_corrupt_video_does_not_destroy_valid_dataset_ingestion(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    create_video(corpus / "valid_video_A.mp4", "red_car")
    (corpus / "corrupt_video.mp4").write_bytes(b"not an mp4")
    create_video(corpus / "valid_video_B.mp4", "blue_object")
    output = tmp_path / "processed"
    report = ingest_path(corpus, MeanRGBEncoder(), VideoIngestConfig(processed_root=output))
    assert report["videos_succeeded"] == 2 and report["videos_failed"] == 1
    assert "VideoDecodeError" in report["failures"][0]["error"]
    assert (output / "corrupt_video/manifest.json").is_file()
    bundle = load_current_frame_index(output / "index")
    assert bundle.index.index.ntotal == 4
    assert {payload["video_id"] for payload in bundle.resolver.payloads.values()} == {"valid_video_A", "valid_video_B"}
