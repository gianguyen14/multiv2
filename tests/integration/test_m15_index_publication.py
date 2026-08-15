import os
from pathlib import Path

import numpy as np
import pytest

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.video.frame_index import build_frame_index, current_generation_id, load_current_frame_index
from backend.app.video.m15_ingestion_pipeline import VideoIngestionPipeline
from tests.m15_support import MeanRGBEncoder


@pytest.mark.parametrize("failure", ["after_staging_write", "before_validation", "after_validation", "index_pre_publish"])
def test_failed_publication_preserves_previous_generation(tmp_path, failure):
    config = VideoIngestConfig(processed_root=tmp_path)
    encoder = MeanRGBEncoder()
    pipeline = VideoIngestionPipeline(encoder, config)
    pipeline.ingest_video("tests/fixtures/test_5s.mp4")
    manifest = pipeline.store.load_manifest("test_5s")
    first = build_frame_index(pipeline.store, [manifest], tmp_path / "index", 4)
    def failpoint(name, context):
        if name == failure:
            raise RuntimeError(name)
    with pytest.raises(RuntimeError, match=failure):
        build_frame_index(pipeline.store, [manifest], tmp_path / "index", 4, failpoint=failpoint)
    active = load_current_frame_index(tmp_path / "index")
    assert active.generation_id == first.generation_id
    assert active.index.search(np.array([1, 0, 0, 0], dtype=np.float32), 1)
    retry = build_frame_index(pipeline.store, [manifest], tmp_path / "index", 4)
    assert retry.generation_id != first.generation_id
    assert (tmp_path / "index/generations" / first.generation_id).is_dir()


def test_current_replace_failure_preserves_active_generation(tmp_path, monkeypatch):
    config = VideoIngestConfig(processed_root=tmp_path)
    pipeline = VideoIngestionPipeline(MeanRGBEncoder(), config)
    pipeline.ingest_video("tests/fixtures/test_5s.mp4")
    manifest = pipeline.store.load_manifest("test_5s")
    index_root = tmp_path / "index"
    first = build_frame_index(pipeline.store, [manifest], index_root, 4)
    real_replace = os.replace

    def fail_current_replace(source, destination):
        if Path(destination) == index_root / "CURRENT":
            raise OSError("forced CURRENT replacement failure")
        return real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr("backend.app.video.atomic_io.os.replace", fail_current_replace)
        with pytest.raises(OSError, match="CURRENT replacement"):
            build_frame_index(pipeline.store, [manifest], index_root, 4)

    assert current_generation_id(index_root) == first.generation_id
    active = load_current_frame_index(index_root)
    assert active.generation_id == first.generation_id
    hit = active.index.search(np.array([1, 0, 0, 0], dtype=np.float32), 1)[0]
    assert active.resolver.resolve(hit["frame_id"])["frame_uid"] == hit["frame_id"]

    retry = build_frame_index(pipeline.store, [manifest], index_root, 4)
    assert retry.generation_id != first.generation_id
    assert current_generation_id(index_root) == retry.generation_id
    assert (index_root / "generations" / first.generation_id).is_dir()
