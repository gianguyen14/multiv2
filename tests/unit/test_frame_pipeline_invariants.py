"""Zero-model invariant audit suite for AIC Frame Sampling & Deduplication."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.video.frame_dedup import filter_near_duplicate_frames
from backend.app.video.frame_index import load_current_frame_index
from backend.app.video.frame_record import FrameRecord
from backend.app.video.frame_sampler import (
    FrameSamplingError,
    iter_sample_frames,
    sample_frames,
    sample_sparse_shot_frames_with_protection,
)
from backend.app.video.m15_ingestion_pipeline import VideoIngestionPipeline
from backend.app.video.video_decoder import DecodedFrame


def _create_synthetic_timeline(duration_seconds: float, fps: float = 30.0) -> list[DecodedFrame]:
    total_frames = int(round(duration_seconds * fps)) + 1
    return [
        DecodedFrame(
            source_frame_index_zero_based=i,
            pts=i,
            timestamp_seconds=i / fps,
            width=2,
            height=2,
            image=Image.new("RGB", (2, 2)),
        )
        for i in range(total_frames)
    ]


# ============================================================================
# 1. CONFIGURATION ROLLBACK & INVALID VALUE REJECTION
# ============================================================================

def test_config_rollback_legacy():
    config = VideoIngestConfig(
        visual_sampling_mode="legacy",
        sample_interval_seconds=1.0,
        visual_dedup_enabled=False,
    )
    assert config.visual_sampling_mode == "legacy"
    assert config.effective_sample_interval_seconds == 1.0
    assert config.visual_dedup_enabled is False


def test_config_sparse_shot_resolution():
    config = VideoIngestConfig(
        visual_sampling_mode="sparse_shot",
        visual_global_sample_seconds=5.0,
        visual_dedup_enabled=True,
        visual_dedup_threshold=0.97,
    )
    assert config.visual_sampling_mode == "sparse_shot"
    assert config.effective_sample_interval_seconds == 5.0
    assert config.visual_dedup_enabled is True
    assert config.visual_dedup_threshold == 0.97


def test_config_invalid_values_rejection():
    # Unknown sampling mode
    with pytest.raises(ValueError):
        VideoIngestConfig(visual_sampling_mode="unsupported_mode")

    # Invalid global sample interval (0, negative, NaN, Inf)
    for invalid_interval in [0, -1.0, float("nan"), float("inf"), -float("inf"), "abc"]:
        with pytest.raises(ValueError):
            VideoIngestConfig(visual_global_sample_seconds=invalid_interval)

    # Invalid dedup threshold (<0, >1, NaN, Inf)
    for invalid_thresh in [-0.1, 1.05, float("nan"), float("inf"), "abc"]:
        with pytest.raises(ValueError):
            VideoIngestConfig(visual_dedup_threshold=invalid_thresh)


# ============================================================================
# 2. SAMPLER INVARIANT MATRIX
# ============================================================================

def test_sampler_empty_input_raises_error():
    with pytest.raises(FrameSamplingError):
        list(iter_sample_frames([], interval_seconds=1.0))
    with pytest.raises(FrameSamplingError):
        sample_sparse_shot_frames_with_protection([], interval_seconds=5.0)


def test_sampler_single_frame():
    frames = _create_synthetic_timeline(duration_seconds=0.0, fps=30.0)  # 1 frame at t=0.0
    sampled = list(iter_sample_frames(frames, interval_seconds=5.0))
    assert len(sampled) == 1
    assert sampled[0].frame.source_frame_index_zero_based == 0
    assert sampled[0].target_timestamp_seconds == 0.0


def test_sampler_interval_shorter_than_duration():
    frames = _create_synthetic_timeline(duration_seconds=20.0, fps=30.0)
    sampled = list(iter_sample_frames(frames, interval_seconds=5.0))
    # Targets: 0, 5, 10, 15, 20
    assert len(sampled) == 5
    assert [s.frame.source_frame_index_zero_based for s in sampled] == [0, 150, 300, 450, 600]
    # Check chronological ordering and no duplicates
    indices = [s.frame.source_frame_index_zero_based for s in sampled]
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))


def test_sampler_interval_longer_than_duration():
    frames = _create_synthetic_timeline(duration_seconds=3.0, fps=30.0)
    sampled = list(iter_sample_frames(frames, interval_seconds=5.0))
    assert len(sampled) == 1
    assert sampled[0].frame.source_frame_index_zero_based == 0


def test_sampler_indivisible_duration():
    frames = _create_synthetic_timeline(duration_seconds=7.3, fps=30.0)
    sampled = list(iter_sample_frames(frames, interval_seconds=5.0))
    # Targets: 0, 5
    assert len(sampled) == 2
    assert [s.frame.source_frame_index_zero_based for s in sampled] == [0, 150]


# ============================================================================
# 3. SHOT BOUNDARY INVARIANTS & COVERAGE
# ============================================================================

def test_shot_boundary_matrix_and_filtering():
    frames = _create_synthetic_timeline(duration_seconds=20.0, fps=30.0)
    shots = [
        (2.0, 4.0),       # Valid shot 0: midpoint 3.0s -> frame 90
        (5.0, 5.05),      # Very short shot 1: midpoint 5.025s -> frame 151
        (0.0, 1.0),       # Start shot 2: midpoint 0.5s -> frame 15
        (18.0, 20.0),     # End shot 3: midpoint 19.0s -> frame 570
        (4.0, 6.0),       # Shot 4 midpoint 5.0s -> frame 150 (exact collision with periodic target 5.0s)
        (-2.0, 3.0),      # Malformed: negative start -> skipped
        (8.0, 6.0),       # Malformed: reversed start/end -> skipped
        (float("nan"), 5.0),  # Malformed: NaN -> skipped
    ]

    sampled, protected = sample_sparse_shot_frames_with_protection(frames, interval_seconds=5.0, shot_boundaries=shots)

    # Valid shots: (2.0, 4.0), (5.0, 5.05), (0.0, 1.0), (18.0, 20.0), (4.0, 6.0) -> exactly 5 valid shots
    # Periodic samples: 0, 150, 300, 450, 600
    # Midpoints: 90, 151, 15, 570, 150
    # Unique indices: [0, 15, 90, 150, 151, 300, 450, 570, 600]
    expected_indices = [0, 15, 90, 150, 151, 300, 450, 570, 600]
    actual_indices = [s.frame.source_frame_index_zero_based for s in sampled]
    assert actual_indices == expected_indices
    assert actual_indices == sorted(actual_indices)
    assert len(actual_indices) == len(set(actual_indices))
    assert protected == {15, 90, 150, 151, 570}


def test_synthetic_shot_coverage_100_percent():
    frames = _create_synthetic_timeline(duration_seconds=20.0, fps=30.0)
    shots = [
        (1.0, 2.0),
        (5.0, 7.0),
        (11.0, 13.0),
    ]
    sampled, protected = sample_sparse_shot_frames_with_protection(frames, interval_seconds=5.0, shot_boundaries=shots)

    # Synthetic embeddings: distinct orthogonal vectors
    embs = np.eye(len(sampled), 768, dtype=np.float32)
    records = [
        FrameRecord.create(
            video_id="syn_v", source_frame_index_zero_based=s.frame.source_frame_index_zero_based,
            submission_frame_id=s.frame.source_frame_index_zero_based, timestamp_seconds=s.frame.timestamp_seconds,
            pts=s.frame.pts, width=2, height=2, image_path="/syn.jpg", sample_interval_seconds=5.0,
            ingestion_version="m15-v1", shot_id=s.shot_id, sampling_reason=s.sampling_reason,
        )
        for s in sampled
    ]

    ret_recs, ret_embs, ret_indices = filter_near_duplicate_frames(
        records, embs, protected_source_frame_indices=protected, threshold=0.97
    )

    # Coverage verification
    represented_shots = {r.shot_id for r in ret_recs if r.shot_id is not None}
    assert represented_shots == {0, 1, 2}
    assert len(represented_shots) == len(shots)  # 100% coverage


# ============================================================================
# 4. PROVENANCE INVARIANTS & FORBIDDEN STATES
# ============================================================================

def test_provenance_rules_and_forbidden_states():
    # Valid periodic record
    r_p = FrameRecord.create(
        video_id="v", source_frame_index_zero_based=0, submission_frame_id=0,
        timestamp_seconds=0.0, pts=0, width=2, height=2, image_path="/0.jpg",
        sample_interval_seconds=5.0, ingestion_version="m15-v1",
        shot_id=None, sampling_reason="periodic",
    )
    assert r_p.sampling_reason == "periodic"
    assert r_p.shot_id is None

    # Valid shot record
    r_s = FrameRecord.create(
        video_id="v", source_frame_index_zero_based=30, submission_frame_id=30,
        timestamp_seconds=1.0, pts=30, width=2, height=2, image_path="/30.jpg",
        sample_interval_seconds=5.0, ingestion_version="m15-v1",
        shot_id=0, sampling_reason="shot",
    )
    assert r_s.sampling_reason == "shot"
    assert r_s.shot_id == 0

    # Valid periodic+shot record
    r_ps = FrameRecord.create(
        video_id="v", source_frame_index_zero_based=150, submission_frame_id=150,
        timestamp_seconds=5.0, pts=150, width=2, height=2, image_path="/150.jpg",
        sample_interval_seconds=5.0, ingestion_version="m15-v1",
        shot_id=1, sampling_reason="periodic+shot",
    )
    assert r_ps.sampling_reason == "periodic+shot"
    assert r_ps.shot_id == 1

    # Forbidden: periodic with non-None shot_id
    with pytest.raises(ValueError, match="periodic sampling reason must have shot_id=None"):
        FrameRecord.create(
            video_id="v", source_frame_index_zero_based=0, submission_frame_id=0,
            timestamp_seconds=0.0, pts=0, width=2, height=2, image_path="/0.jpg",
            sample_interval_seconds=5.0, ingestion_version="m15-v1",
            shot_id=0, sampling_reason="periodic",
        )

    # Forbidden: shot with None shot_id
    with pytest.raises(ValueError, match="sampling reason must have a non-negative integer shot_id"):
        FrameRecord.create(
            video_id="v", source_frame_index_zero_based=0, submission_frame_id=0,
            timestamp_seconds=0.0, pts=0, width=2, height=2, image_path="/0.jpg",
            sample_interval_seconds=5.0, ingestion_version="m15-v1",
            shot_id=None, sampling_reason="shot",
        )

    # Forbidden: invalid sampling_reason string
    with pytest.raises(ValueError, match="invalid sampling_reason"):
        FrameRecord.create(
            video_id="v", source_frame_index_zero_based=0, submission_frame_id=0,
            timestamp_seconds=0.0, pts=0, width=2, height=2, image_path="/0.jpg",
            sample_interval_seconds=5.0, ingestion_version="m15-v1",
            shot_id=None, sampling_reason="unknown_strategy",
        )


# ============================================================================
# 5. MULTIPLE-SHOT SAME-FRAME EDGE CASE AUDIT
# ============================================================================

def test_multi_shot_same_frame_edge_case():
    """Audit behavior when two synthetic shots resolve to the exact same frame index."""
    frames = _create_synthetic_timeline(duration_seconds=5.0, fps=30.0)
    # Both shots have identical boundaries [1.0, 1.01], midpoint 1.005s -> both resolve to frame 30
    shots = [(1.0, 1.01), (1.0, 1.01)]
    sampled, protected = sample_sparse_shot_frames_with_protection(frames, interval_seconds=5.0, shot_boundaries=shots)

    # Frame 30 is chosen by both shots. In dictionary merge, the later shot ordinal overwrites the earlier.
    frame_30 = next(s for s in sampled if s.frame.source_frame_index_zero_based == 30)
    assert frame_30.sampling_reason == "shot"
    assert frame_30.shot_id == 1  # Later shot ordinal retained in current dictionary merge
    assert 30 in protected
    # Single consolidated frame emitted for frame 30
    assert len([s for s in sampled if s.frame.source_frame_index_zero_based == 30]) == 1


# ============================================================================
# 6. DEDUP INVARIANTS
# ============================================================================

def test_dedup_threshold_and_previous_retained_behavior():
    v0 = np.zeros(768, dtype=np.float32); v0[0] = 1.0
    v1_eq = np.zeros(768, dtype=np.float32); v1_eq[0] = 0.97; v1_eq[1] = np.sqrt(1 - 0.97**2)
    v2_below = np.zeros(768, dtype=np.float32); v2_below[0] = 0.96; v2_below[1] = np.sqrt(1 - 0.96**2)
    v3_above_v2 = np.array(v2_below, copy=True)
    v4_protected = np.array(v2_below, copy=True)

    embeddings = np.stack([v0, v1_eq, v2_below, v3_above_v2, v4_protected])
    records = [
        FrameRecord.create(video_id="v", source_frame_index_zero_based=i*30, submission_frame_id=i*30,
                           timestamp_seconds=float(i), pts=i*30, width=2, height=2, image_path=f"/{i}.jpg",
                           sample_interval_seconds=5.0, ingestion_version="m15-v1",
                           shot_id=0 if i == 4 else None, sampling_reason="shot" if i == 4 else "periodic")
        for i in range(5)
    ]
    protected = {4 * 30}  # frame index 120 (record 4)

    ret_recs, ret_embs, ret_indices = filter_near_duplicate_frames(
        records, embeddings, protected_source_frame_indices=protected, threshold=0.97
    )

    assert ret_indices == [0, 2, 4]
    assert [r.source_frame_index_zero_based for r in ret_recs] == [0, 60, 120]


def test_dedup_malformed_embeddings_rejection():
    records = [
        FrameRecord.create(video_id="v", source_frame_index_zero_based=0, submission_frame_id=0,
                           timestamp_seconds=0.0, pts=0, width=2, height=2, image_path="/0.jpg",
                           sample_interval_seconds=5.0, ingestion_version="m15-v1",
                           shot_id=None, sampling_reason="periodic")
    ]
    # Mismatched counts
    embs_wrong_len = np.zeros((2, 768), dtype=np.float32)
    with pytest.raises(ValueError, match="Mismatched records count"):
        filter_near_duplicate_frames(records, embs_wrong_len)

    # 1D array instead of 2D
    embs_1d = np.zeros(1, dtype=np.float32)
    with pytest.raises(ValueError, match="embeddings must be a 2D float32 array"):
        filter_near_duplicate_frames(records, embs_1d)


# ============================================================================
# 7. METADATA ROUNDTRIP & UNKNOWN PAYLOAD COMPATIBILITY
# ============================================================================

def test_metadata_roundtrip_new_and_old_payloads():
    rec = FrameRecord.create(
        video_id="L22_V001", source_frame_index_zero_based=150, submission_frame_id=150,
        timestamp_seconds=5.0, pts=150, width=1280, height=720, image_path="/path/150.jpg",
        sample_interval_seconds=5.0, ingestion_version="m15-v1",
        shot_id=2, sampling_reason="periodic+shot",
    )
    d = rec.to_dict()
    assert d["shot_id"] == 2
    assert d["sampling_reason"] == "periodic+shot"

    # Exact roundtrip
    rec_restored = FrameRecord.from_dict(d)
    assert rec_restored == rec

    # Unknown payload fields (e.g., candidate_id, extra_metric)
    d_with_unknown = dict(d)
    d_with_unknown["candidate_id"] = "L22_V001:000000150"
    d_with_unknown["extra_metric"] = 0.999
    d_with_unknown["arbitrary_tag"] = "test"

    rec_unknown = FrameRecord.from_dict(d_with_unknown)
    assert rec_unknown == rec
    assert rec_unknown.shot_id == 2
    assert rec_unknown.sampling_reason == "periodic+shot"


# ============================================================================
# 8. EXISTING INDEX READ-ONLY CONTRACT & FAISS SELF-QUERY
# ============================================================================

def test_existing_index_read_only_contract():
    index_path = Path("data/processed-rc2-smoke/index")
    if not index_path.exists():
        pytest.skip("data/processed-rc2-smoke/index not present locally")

    bundle = load_current_frame_index(index_path)
    ntotal = bundle.index.index.ntotal
    m_count = len(bundle.index.frame_id_mapping)
    p_count = len(bundle.resolver.payloads)

    assert ntotal == m_count == p_count == 3578
    assert bundle.index.embedding_dim == 768

    # Frame-ID resolution audit for first, middle, last entries
    first_fid = bundle.index.frame_id_mapping[0]
    mid_fid = bundle.index.frame_id_mapping[len(bundle.index.frame_id_mapping) // 2]
    last_fid = bundle.index.frame_id_mapping[len(bundle.index.frame_id_mapping) - 1]

    for fid in [first_fid, mid_fid, last_fid]:
        p = bundle.resolver.resolve(fid)
        assert p is not None
        assert p["frame_uid"] == fid
        assert p["video_id"] in {"L22_V001", "L22_V002", "L22_V003"}
        assert p["timestamp_seconds"] >= 0.0

    # FAISS self-query for 3 stored vectors
    for vec_idx in [0, len(bundle.index.frame_id_mapping) // 2, len(bundle.index.frame_id_mapping) - 1]:
        q_vec = bundle.index.index.reconstruct(vec_idx)
        results = bundle.index.search(q_vec, top_k=3)
        assert len(results) >= 1
        top_res = results[0]
        assert np.isfinite(top_res["score"])
        assert top_res["score"] >= 0.9999
        assert bundle.resolver.resolve(top_res["frame_id"]) is not None


# ============================================================================
# 9. STEP-6 PRODUCTION CHANGES AUDIT (MOCKED FACTORY CALL PATH)
# ============================================================================

def test_shot_detector_factory_call_path_mocked():
    mock_encoder = MagicMock()
    mock_encoder.embedding_dim = 768
    mock_encoder.identity.return_value = {"embedding_dim": 768, "model_name": "mock"}

    with patch("backend.app.video.m15_ingestion_pipeline.get_shot_detector") as mock_get_detector:
        mock_detector_inst = MagicMock()
        mock_get_detector.return_value = mock_detector_inst

        # 1. Legacy mode -> must NOT call get_shot_detector()
        cfg_legacy = VideoIngestConfig(visual_sampling_mode="legacy")
        pipe_legacy = VideoIngestionPipeline(mock_encoder, cfg_legacy)
        assert pipe_legacy.shot_detector is None
        mock_get_detector.assert_not_called()

        # 2. Sparse_shot mode with None shot_detector -> calls get_shot_detector()
        cfg_sparse = VideoIngestConfig(visual_sampling_mode="sparse_shot")
        pipe_sparse = VideoIngestionPipeline(mock_encoder, cfg_sparse)
        assert pipe_sparse.shot_detector is mock_detector_inst
        mock_get_detector.assert_called_once()
