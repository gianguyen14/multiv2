from dataclasses import replace

import numpy as np

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.video.frame_index import load_current_frame_index
from backend.app.video.ingest import ingest_path
from backend.app.video.m15_ingestion_pipeline import VideoIngestionPipeline
from tests.m15_support import MeanRGBEncoder


def test_config_invalidation_is_dependency_aware(tmp_path):
    source = "tests/fixtures/test_5s.mp4"
    base = VideoIngestConfig(processed_root=tmp_path)
    first_encoder = MeanRGBEncoder()
    VideoIngestionPipeline(first_encoder, base).ingest_video(source)
    assert VideoIngestionPipeline(first_encoder, replace(base, embed_batch_size=2, device="other")).ingest_video(source)["start_stage"] == "complete"
    interval = VideoIngestionPipeline(first_encoder, replace(base, sample_interval_seconds=0.5)).ingest_video(source)
    assert interval["start_stage"] == "frames" and interval["sampled_frame_count"] == 10
    policy = VideoIngestionPipeline(first_encoder, replace(base, sample_interval_seconds=0.5, frame_id_policy="one_based")).ingest_video(source)
    assert policy["start_stage"] == "frames"
    second_encoder = MeanRGBEncoder("v2")
    embedding = VideoIngestionPipeline(second_encoder, replace(base, sample_interval_seconds=0.5, frame_id_policy="one_based")).ingest_video(source)
    assert embedding["start_stage"] == "embeddings"
    assert second_encoder.calls == 1


def test_manifest_claim_is_rejected_when_embedding_is_missing(tmp_path):
    config = VideoIngestConfig(processed_root=tmp_path)
    encoder = MeanRGBEncoder()
    VideoIngestionPipeline(encoder, config).ingest_video("tests/fixtures/test_5s.mp4")
    (tmp_path / "test_5s/embeddings.npy").unlink()
    result = VideoIngestionPipeline(encoder, config).ingest_video("tests/fixtures/test_5s.mp4")
    assert result["start_stage"] == "embeddings"


def test_index_type_change_rebuilds_only_index(tmp_path, monkeypatch):
    source = "tests/fixtures/test_5s.mp4"
    flat_config = VideoIngestConfig(processed_root=tmp_path, index_type="flat")
    ingest_path(source, MeanRGBEncoder(), flat_config)
    flat = load_current_frame_index(tmp_path / "index")
    store = VideoIngestionPipeline(MeanRGBEncoder(), flat_config).store
    metadata = store.load_metadata("test_5s").to_dict()
    records = [record.to_dict() for record in store.load_records("test_5s")]
    embeddings = store.load_embeddings("test_5s").copy()

    def unexpected(*args, **kwargs):
        raise AssertionError("per-video artifact rebuilt")

    monkeypatch.setattr("backend.app.video.m15_ingestion_pipeline.inspect_video", unexpected)
    monkeypatch.setattr("backend.app.video.m15_ingestion_pipeline.iter_frames", unexpected)
    monkeypatch.setattr("backend.app.video.frame_store.FrameStore.save_image", unexpected)
    encoder = MeanRGBEncoder()
    report = ingest_path(source, encoder, replace(flat_config, index_type="hnsw"))

    assert report["results"][0]["start_stage"] == "complete"
    assert encoder.calls == 0
    assert store.load_metadata("test_5s").to_dict() == metadata
    assert [record.to_dict() for record in store.load_records("test_5s")] == records
    np.testing.assert_array_equal(store.load_embeddings("test_5s"), embeddings)
    hnsw = load_current_frame_index(tmp_path / "index")
    assert hnsw.generation_id != flat.generation_id
    assert hnsw.metadata["index_type"] == "hnsw"
    assert set(hnsw.resolver.payloads) == set(flat.resolver.payloads)
    hit = hnsw.index.search(np.array([1, 0, 0, 0], dtype=np.float32), 1)[0]
    assert hnsw.resolver.resolve(hit["frame_id"])["frame_uid"] == hit["frame_id"]
