import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.video.frame_index import build_frame_index
from backend.app.video.m15_ingestion_pipeline import VideoIngestionPipeline
from tests.m15_support import MeanRGBEncoder


def logical_artifacts(root):
    pipeline = VideoIngestionPipeline(MeanRGBEncoder(), VideoIngestConfig(processed_root=root))
    store = pipeline.store
    manifest = store.load_manifest("test_5s")
    bundle = build_frame_index(store, [manifest], root / "index", 4)
    records = [record.to_dict() for record in store.load_records("test_5s")]
    for record in records:
        record["image_path"] = Path(record["image_path"]).relative_to(root).as_posix()
    generation = {key: bundle.metadata[key] for key in ("index_type", "embedding_dim", "vector_count", "video_ids")}
    mapping = json.loads((bundle.generation_path / "mapping.json").read_text())
    payloads = json.loads((bundle.generation_path / "payloads.json").read_text())
    for payload in payloads["payloads"].values():
        payload["image_path"] = Path(payload["image_path"]).relative_to(root).as_posix()
    return store.load_metadata("test_5s").to_dict(), records, store.load_embeddings("test_5s"), generation, mapping, payloads


@pytest.mark.parametrize("stage,expected_start,expected_calls", [
    ("after_metadata", "frames", 1),
    ("after_frames", "embeddings", 1),
    ("after_embeddings", "complete", 1),
])
def test_forced_interruption_resumes_from_highest_valid_stage(tmp_path, stage, expected_start, expected_calls):
    source = "tests/fixtures/test_5s.mp4"
    clean_root = tmp_path / "clean"
    clean_config = VideoIngestConfig(processed_root=clean_root)
    VideoIngestionPipeline(MeanRGBEncoder(), clean_config).ingest_video(source)
    clean = logical_artifacts(clean_root)

    resumed_root = tmp_path / "resumed"
    config = VideoIngestConfig(processed_root=resumed_root)
    encoder = MeanRGBEncoder()

    def failpoint(name, context):
        if name == stage:
            raise RuntimeError(f"forced {name}")

    with pytest.raises(RuntimeError, match="forced"):
        VideoIngestionPipeline(encoder, config, failpoint).ingest_video(source)
    interrupted = VideoIngestionPipeline(encoder, config).store.load_manifest("test_5s")
    assert interrupted.status == "failed"
    result = VideoIngestionPipeline(encoder, config).ingest_video(source)
    assert result["start_stage"] == expected_start
    assert encoder.calls == expected_calls
    healed = VideoIngestionPipeline(encoder, config).store.load_manifest("test_5s")
    assert healed.status == "embeddings_ready"
    assert healed.failed_stage is None and healed.error is None

    resumed = logical_artifacts(resumed_root)
    assert clean[:2] == resumed[:2]
    np.testing.assert_array_equal(clean[2], resumed[2])
    assert clean[3:] == resumed[3:]
    mapping = resumed[4]["frame_id_mapping"]
    payloads = resumed[5]["payloads"]
    assert list(map(int, mapping)) == list(range(len(mapping)))
    assert all(payloads[frame_uid]["candidate_id"] == payloads[frame_uid]["frame_uid"] == frame_uid
        for frame_uid in mapping.values())
