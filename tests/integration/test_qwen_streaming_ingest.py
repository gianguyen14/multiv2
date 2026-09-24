import numpy as np

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.video.ingest import ingest_path
from backend.app.video.m15_ingestion_pipeline import VideoIngestionPipeline


class FakeQwenEncoder:
    embedding_dim = 1024

    def __init__(self):
        self.calls = 0

    def identity(self):
        return {
            "backend": "qwen3_vl",
            "model_name": "Qwen/Qwen3-VL-Embedding-2B",
            "revision": "test-revision",
            "instruction": "Retrieve the video frame that best matches the described visual scene.",
            "embedding_dim": 1024,
            "normalization": "l2",
            "output_dtype": "float32",
            "contract_version": "qwen3-vl-image-ingest-v1",
        }

    def encode_image(self, images, batch_size=1, normalize=True):
        assert batch_size == 1
        assert normalize is True
        self.calls += len(images)
        values = np.zeros((len(images), 1024), dtype=np.float32)
        values[:, 0] = 1.0
        return values


def test_qwen_streaming_pipeline_persists_contract_and_source_identity(tmp_path):
    encoder = FakeQwenEncoder()
    config = VideoIngestConfig(
        processed_root=tmp_path,
        sample_interval_seconds=10.0,
        ingest_backend="qwen3_vl",
        qwen_embedding_dim=1024,
        embed_batch_size=1,
        qwen_image_width=896,
        decode_threads=6,
        ingest_queue_depth=8,
    )
    result = VideoIngestionPipeline(encoder, config).ingest_video("tests/fixtures/test_5s.mp4")
    assert result["sampled_frame_count"] == 1
    assert result["streaming_progress"]["embedded_frames"] == 1
    records = VideoIngestionPipeline(encoder, config).store.load_records("test_5s")
    embeddings = VideoIngestionPipeline(encoder, config).store.load_embeddings("test_5s")
    assert embeddings.shape == (1, 1024)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert np.isclose(np.linalg.norm(embeddings[0]), 1.0)
    assert records[0].source_frame_index_zero_based == 0
    assert records[0].submission_frame_id == 0
    assert records[0].pts is not None
    assert records[0].timestamp_seconds is not None


def test_qwen_directory_ingest_builds_index_once_after_videos(tmp_path, monkeypatch):
    encoder = FakeQwenEncoder()
    config = VideoIngestConfig(
        processed_root=tmp_path,
        sample_interval_seconds=10.0,
        ingest_backend="qwen3_vl",
        qwen_embedding_dim=1024,
        embed_batch_size=1,
    )
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    for name in ("a.mp4", "b.mp4"):
        (source_dir / name).write_bytes((__import__("pathlib").Path("tests/fixtures/test_5s.mp4").read_bytes()))
    import backend.app.video.ingest as ingest_module
    calls = []
    original = ingest_module.build_frame_index
    monkeypatch.setattr(ingest_module, "build_frame_index", lambda *args, **kwargs: (calls.append(1) or original(*args, **kwargs)))
    report = ingest_path(source_dir, encoder, config)
    assert report["videos_succeeded"] == 2
    assert report["videos_failed"] == 0
    assert len(calls) == 1
    assert encoder.calls == 2
