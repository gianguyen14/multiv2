import json
import math
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock
import numpy as np
import pytest
from PIL import Image

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.video.frame_dedup import filter_near_duplicate_frames
from backend.app.video.frame_index import build_frame_index, validate_generation
from backend.app.video.frame_record import FrameRecord
from backend.app.video.frame_sampler import (
    sample_sparse_shot_frames_with_protection,
    iter_sample_frames,
)
from backend.app.video.frame_store import FrameStore
from backend.app.video.ingest_manifest import IngestManifest
from backend.app.video.video_decoder import DecodedFrame


def _create_synthetic_frames(duration_seconds: float, fps: float = 30.0) -> List[DecodedFrame]:
    total_frames = int(round(duration_seconds * fps)) + 1
    frames = []
    for i in range(total_frames):
        ts = i / fps
        frames.append(DecodedFrame(
            source_frame_index_zero_based=i,
            pts=i,
            timestamp_seconds=ts,
            width=2,
            height=2,
            image=Image.new("RGB", (2, 2)),
        ))
    return frames


def test_sampling_reason_periodic_only():
    """Frame selected only by periodic sampling has reason='periodic' and shot_id=None."""
    frames = _create_synthetic_frames(duration_seconds=10.0, fps=30.0)
    # No shots provided -> pure periodic sampling
    sampled, protected = sample_sparse_shot_frames_with_protection(frames, interval_seconds=5.0, shot_boundaries=[])
    
    assert len(sampled) == 3  # t=0s (0), t=5s (150), t=10s (300)
    assert len(protected) == 0
    for s in sampled:
        assert s.sampling_reason == "periodic"
        assert s.shot_id is None


def test_sampling_reason_shot_only():
    """Shot midpoint not coinciding with periodic sample has reason='shot' and correct shot_id."""
    frames = _create_synthetic_frames(duration_seconds=10.0, fps=30.0)
    # Shot [1.5, 2.5] midpoint = 2.0s (frame 60). Periodic targets are 0.0s (0), 5.0s (150), 10.0s (300).
    shots = [(1.5, 2.5)]
    sampled, protected = sample_sparse_shot_frames_with_protection(frames, interval_seconds=5.0, shot_boundaries=shots)

    assert len(sampled) == 4
    # Frame 60 (t=2.0s) is shot-only
    frame_60 = next(s for s in sampled if s.frame.source_frame_index_zero_based == 60)
    assert frame_60.sampling_reason == "shot"
    assert frame_60.shot_id == 0
    assert 60 in protected


def test_sampling_reason_periodic_plus_shot_overlap():
    """When periodic sample and shot midpoint choose the exact same frame, reason='periodic+shot'."""
    frames = _create_synthetic_frames(duration_seconds=10.0, fps=30.0)
    # Periodic at 5s selects frame 0 (t=0s), frame 150 (t=5s), frame 300 (t=10s)
    # Shot #0 [4.0, 6.0] has midpoint 5.0s (frame 150)
    shots = [(4.0, 6.0)]
    sampled, protected = sample_sparse_shot_frames_with_protection(frames, interval_seconds=5.0, shot_boundaries=shots)

    assert len(sampled) == 3  # Frame 150 appears only once
    frame_150 = next(s for s in sampled if s.frame.source_frame_index_zero_based == 150)
    assert frame_150.sampling_reason == "periodic+shot"
    assert frame_150.shot_id == 0
    assert 150 in protected


def test_multiple_shots_assigns_stable_zero_based_shot_ids():
    """Multiple detected shots assign distinct zero-based integer shot IDs 0, 1, 2, ..."""
    frames = _create_synthetic_frames(duration_seconds=20.0, fps=30.0)
    shots = [
        (1.0, 2.0),   # Shot #0 -> midpoint 1.5s (frame 45)
        (3.0, 4.0),   # Shot #1 -> midpoint 3.5s (frame 105)
        (7.0, 9.0),   # Shot #2 -> midpoint 8.0s (frame 240)
    ]
    sampled, protected = sample_sparse_shot_frames_with_protection(frames, interval_seconds=5.0, shot_boundaries=shots)

    # Check shot_id on shot representatives
    shot_frames = [s for s in sampled if s.sampling_reason in {"shot", "periodic+shot"}]
    assert len(shot_frames) == 3
    assert [s.shot_id for s in shot_frames] == [0, 1, 2]
    assert protected == {45, 105, 240}


def test_dedup_alignment_with_retained_metadata():
    """Verify that when dedup drops non-protected frames, metadata remains strictly 1-to-1 aligned with retained embeddings."""
    # Sequence:
    # A (t=0.0s, periodic): retained
    # B (t=1.0s, periodic): duplicate of A -> dropped
    # C (t=2.0s, shot #0): duplicate of A but protected -> retained
    # D (t=5.0s, periodic+shot #1): duplicate of C but protected -> retained
    rec_a = FrameRecord.create(
        video_id="vid1", source_frame_index_zero_based=0, submission_frame_id=0,
        timestamp_seconds=0.0, pts=0, width=2, height=2, image_path="/path/0.jpg",
        sample_interval_seconds=5.0, ingestion_version="m15-v1",
        shot_id=None, sampling_reason="periodic",
    )
    rec_b = FrameRecord.create(
        video_id="vid1", source_frame_index_zero_based=30, submission_frame_id=30,
        timestamp_seconds=1.0, pts=30, width=2, height=2, image_path="/path/30.jpg",
        sample_interval_seconds=5.0, ingestion_version="m15-v1",
        shot_id=None, sampling_reason="periodic",
    )
    rec_c = FrameRecord.create(
        video_id="vid1", source_frame_index_zero_based=60, submission_frame_id=60,
        timestamp_seconds=2.0, pts=60, width=2, height=2, image_path="/path/60.jpg",
        sample_interval_seconds=5.0, ingestion_version="m15-v1",
        shot_id=0, sampling_reason="shot",
    )
    rec_d = FrameRecord.create(
        video_id="vid1", source_frame_index_zero_based=150, submission_frame_id=150,
        timestamp_seconds=5.0, pts=150, width=2, height=2, image_path="/path/150.jpg",
        sample_interval_seconds=5.0, ingestion_version="m15-v1",
        shot_id=1, sampling_reason="periodic+shot",
    )
    records = [rec_a, rec_b, rec_c, rec_d]

    # All embeddings identical (cosine = 1.0)
    v = np.zeros(768, dtype=np.float32)
    v[0] = 1.0
    embeddings = np.stack([v, v, v, v])

    # Protected indices: C (60) and D (150)
    protected = {60, 150}

    ret_recs, ret_embs, ret_indices = filter_near_duplicate_frames(
        records, embeddings, protected_source_frame_indices=protected, threshold=0.97
    )

    # Retained should be exactly A (0), C (60), D (150)
    assert ret_indices == [0, 2, 3]
    assert len(ret_recs) == 3
    assert len(ret_embs) == 3
    assert [r.source_frame_index_zero_based for r in ret_recs] == [0, 60, 150]
    assert [r.sampling_reason for r in ret_recs] == ["periodic", "shot", "periodic+shot"]
    assert [r.shot_id for r in ret_recs] == [None, 0, 1]


def test_old_artifact_backward_compatibility():
    """Deserializing old records/payloads without shot_id and sampling_reason must succeed gracefully."""
    old_record_dict = {
        "frame_uid": "vid1:000000000",
        "video_id": "vid1",
        "source_frame_index_zero_based": 0,
        "submission_frame_id": 0,
        "timestamp_seconds": 0.0,
        "pts": 0,
        "width": 1920,
        "height": 1080,
        "image_path": "/path/0.jpg",
        "embedding_id": "vid1:000000000",
        "sample_strategy": "temporal_coarse",
        "sample_interval_seconds": 1.0,
        "ingestion_version": "m15-v1",
    }

    record = FrameRecord.from_dict(old_record_dict)
    assert record.frame_uid == "vid1:000000000"
    assert record.shot_id is None
    assert record.sampling_reason is None


def test_index_publication_payload_preserves_sampling_metadata(tmp_path):
    """Publishing index creates payloads.json with matching video_id, timestamp, sampling_reason, and shot_id."""
    store = FrameStore(tmp_path / "processed")
    video_id = "test_vid"
    
    rec_0 = FrameRecord.create(
        video_id=video_id, source_frame_index_zero_based=0, submission_frame_id=0,
        timestamp_seconds=0.0, pts=0, width=2, height=2, image_path=str(tmp_path / "0.jpg"),
        sample_interval_seconds=5.0, ingestion_version="m15-v1",
        shot_id=None, sampling_reason="periodic",
    )
    rec_1 = FrameRecord.create(
        video_id=video_id, source_frame_index_zero_based=75, submission_frame_id=75,
        timestamp_seconds=2.5, pts=75, width=2, height=2, image_path=str(tmp_path / "75.jpg"),
        sample_interval_seconds=5.0, ingestion_version="m15-v1",
        shot_id=0, sampling_reason="shot",
    )
    records = [rec_0, rec_1]
    store.save_records(video_id, records)

    v0 = np.zeros(768, dtype=np.float32)
    v0[0] = 1.0
    v1 = np.zeros(768, dtype=np.float32)
    v1[1] = 1.0
    embeddings = np.stack([v0, v1])
    store.save_embeddings(video_id, embeddings)

    manifest = IngestManifest(
        video_id=video_id,
        source_path="/tmp/fake.mp4",
        source_size=100,
        source_mtime_ns=100,
        source_hash="fakehash",
        ingestion_version="m15-v1",
        status="embeddings_ready",
        completed_stage="embeddings",
    )
    store.save_manifest(manifest)

    # Build frame index
    output_root = tmp_path / "index_out"
    bundle = build_frame_index(
        store=store,
        manifests=[manifest],
        output_root=output_root,
        embedding_dim=768,
        index_type="flat",
    )

    # Verify ntotal == mapping_count == payload_count
    assert bundle.index.index.ntotal == 2
    assert len(bundle.resolver.payloads) == 2

    # Inspect payloads
    payload_0 = bundle.resolver.resolve(f"{video_id}:000000000")
    assert payload_0 is not None
    assert payload_0["video_id"] == video_id
    assert payload_0["source_frame_index_zero_based"] == 0
    assert payload_0["timestamp_seconds"] == 0.0
    assert payload_0["sampling_reason"] == "periodic"
    assert payload_0["shot_id"] is None

    payload_1 = bundle.resolver.resolve(f"{video_id}:000000075")
    assert payload_1 is not None
    assert payload_1["video_id"] == video_id
    assert payload_1["source_frame_index_zero_based"] == 75
    assert payload_1["timestamp_seconds"] == 2.5
    assert payload_1["sampling_reason"] == "shot"
    assert payload_1["shot_id"] == 0
