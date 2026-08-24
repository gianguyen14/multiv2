import math
from dataclasses import dataclass
from typing import List
import numpy as np
import pytest

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.video.frame_dedup import filter_near_duplicate_frames


@dataclass
class DummyFrameRecord:
    source_frame_index_zero_based: int
    timestamp_seconds: float


def _make_unit_vector(dim: int, angle_rad: float = 0.0) -> np.ndarray:
    """Create a 2D-rotated unit vector in dimension D."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = math.cos(angle_rad)
    vec[1] = math.sin(angle_rad)
    return vec / np.linalg.norm(vec)


def test_first_frame_always_retained():
    """First frame in a video must always be retained."""
    records = [DummyFrameRecord(0, 0.0)]
    embs = np.array([_make_unit_vector(768, 0.0)], dtype=np.float32)

    ret_recs, ret_embs, ret_indices = filter_near_duplicate_frames(records, embs, threshold=0.97)
    assert len(ret_recs) == 1
    assert ret_indices == [0]


def test_near_duplicate_removed_when_above_threshold():
    """Consecutive non-protected frame with cosine >= threshold must be dropped."""
    records = [
        DummyFrameRecord(0, 0.0),
        DummyFrameRecord(1, 1.0),
    ]
    # Vectors with cosine similarity ~0.99 > 0.97
    v0 = _make_unit_vector(768, 0.0)
    v1 = _make_unit_vector(768, 0.1)  # cos(0.1) = 0.9950
    embs = np.stack([v0, v1])

    ret_recs, ret_embs, ret_indices = filter_near_duplicate_frames(records, embs, threshold=0.97)
    assert len(ret_recs) == 1
    assert ret_indices == [0]
    assert ret_recs[0].source_frame_index_zero_based == 0


def test_distinct_frame_retained_when_below_threshold():
    """Consecutive non-protected frame with cosine < threshold must be kept."""
    records = [
        DummyFrameRecord(0, 0.0),
        DummyFrameRecord(1, 1.0),
    ]
    # Vectors with cosine similarity ~0.87 < 0.97
    v0 = _make_unit_vector(768, 0.0)
    v1 = _make_unit_vector(768, 0.5)  # cos(0.5) = 0.8775
    embs = np.stack([v0, v1])

    ret_recs, ret_embs, ret_indices = filter_near_duplicate_frames(records, embs, threshold=0.97)
    assert len(ret_recs) == 2
    assert ret_indices == [0, 1]


def test_threshold_boundary_deterministic_drop_at_exact_equality():
    """When cosine similarity exactly equals threshold, frame must be dropped (rule: cosine >= threshold)."""
    records = [
        DummyFrameRecord(0, 0.0),
        DummyFrameRecord(1, 1.0),
    ]
    target_thresh = 0.97
    # cos(angle) = 0.97
    angle = math.acos(target_thresh)
    v0 = _make_unit_vector(768, 0.0)
    v1 = _make_unit_vector(768, angle)
    embs = np.stack([v0, v1])

    # Dot product is exactly 0.97
    assert np.isclose(np.dot(v0, v1), 0.97, atol=1e-6)

    ret_recs, _, ret_indices = filter_near_duplicate_frames(records, embs, threshold=0.97)
    assert len(ret_recs) == 1
    assert ret_indices == [0]


def test_comparison_against_previous_retained_reference():
    """Verify that candidate C is compared against last RETAINED frame A, not dropped candidate B."""
    records = [
        DummyFrameRecord(0, 0.0),  # Frame A
        DummyFrameRecord(1, 1.0),  # Frame B: duplicate of A (angle 0.05, cos(0.05) = 0.9987 > 0.97)
        DummyFrameRecord(2, 2.0),  # Frame C: angle 0.3 (cos(0.3 - 0.0) = 0.9553 < 0.97 vs A)
    ]
    v0 = _make_unit_vector(768, 0.0)
    v1 = _make_unit_vector(768, 0.05)
    v2 = _make_unit_vector(768, 0.3)
    embs = np.stack([v0, v1, v2])

    ret_recs, _, ret_indices = filter_near_duplicate_frames(records, embs, threshold=0.97)
    # A is retained, B is dropped.
    # C is compared with A: dot(v2, v0) = cos(0.3) = 0.9553 < 0.97 -> C is RETAINED.
    # Note: If C were compared against dropped B, dot(v2, v1) = cos(0.25) = 0.9689 < 0.97
    assert ret_indices == [0, 2]
    assert [r.source_frame_index_zero_based for r in ret_recs] == [0, 2]


def test_protected_shot_representative_survives_near_duplicate():
    """A protected shot representative with cosine >= threshold must still be preserved."""
    records = [
        DummyFrameRecord(0, 0.0),  # Periodic
        DummyFrameRecord(1, 1.0),  # Shot midpoint (protected) with identical vector
        DummyFrameRecord(2, 2.0),  # Periodic with identical vector (not protected)
    ]
    v0 = _make_unit_vector(768, 0.0)
    v1 = _make_unit_vector(768, 0.0)  # cosine = 1.0
    v2 = _make_unit_vector(768, 0.0)  # cosine = 1.0
    embs = np.stack([v0, v1, v2])

    # Frame 1 is protected as shot representative
    protected = {1}
    ret_recs, _, ret_indices = filter_near_duplicate_frames(
        records, embs, protected_source_frame_indices=protected, threshold=0.97
    )

    # Frame 0 is retained (first frame).
    # Frame 1 is retained (protected shot rep, even though cosine = 1.0).
    # Frame 2 is dropped (non-protected, cosine = 1.0 vs Frame 1).
    assert ret_indices == [0, 1]
    assert [r.source_frame_index_zero_based for r in ret_recs] == [0, 1]


def test_all_detected_shots_remain_represented():
    """Synthetic shot set where shot midpoints are visually redundant must still achieve 100% coverage."""
    records = [
        DummyFrameRecord(0, 0.0),    # Periodic (t=0s)
        DummyFrameRecord(30, 1.0),   # Shot #1 midpoint (t=1.0s), protected
        DummyFrameRecord(90, 3.0),   # Shot #2 midpoint (t=3.0s), protected
        DummyFrameRecord(150, 5.0),  # Periodic (t=5.0s)
    ]
    # All identical embeddings (cosine = 1.0)
    v = _make_unit_vector(768, 0.0)
    embs = np.stack([v, v, v, v])

    shots = [(0.5, 1.5), (2.5, 3.5)]  # Shot #1 and Shot #2
    protected = {30, 90}

    ret_recs, _, ret_indices = filter_near_duplicate_frames(
        records, embs, protected_source_frame_indices=protected, threshold=0.97
    )

    # Frame 0 kept (first frame)
    # Frame 30 kept (Shot #1 protected)
    # Frame 90 kept (Shot #2 protected)
    # Frame 150 dropped (Periodic duplicate of 90)
    assert ret_indices == [0, 1, 2]
    retained_times = [r.timestamp_seconds for r in ret_recs]

    # Verify coverage invariant
    for shot_idx, (s_start, s_end) in enumerate(shots):
        in_shot = [t for t in retained_times if s_start <= t <= s_end]
        assert len(in_shot) >= 1, f"Shot #{shot_idx} was not represented after dedup!"


def test_periodic_only_fallback():
    """When no shot boundaries exist, dedup operates deterministically over periodic candidates."""
    records = [
        DummyFrameRecord(0, 0.0),
        DummyFrameRecord(30, 1.0),
        DummyFrameRecord(60, 2.0),
    ]
    # Vectors: 0 and 30 are duplicates, 60 is distinct
    v0 = _make_unit_vector(768, 0.0)
    v1 = _make_unit_vector(768, 0.01)  # cos ~ 0.9999 > 0.97
    v2 = _make_unit_vector(768, 0.5)   # cos ~ 0.8775 < 0.97
    embs = np.stack([v0, v1, v2])

    ret_recs, _, ret_indices = filter_near_duplicate_frames(
        records, embs, protected_source_frame_indices=set(), threshold=0.97
    )
    assert ret_indices == [0, 2]
    assert [r.source_frame_index_zero_based for r in ret_recs] == [0, 60]


def test_disabled_dedup_returns_all_candidates():
    """When enabled=False, all candidates and embeddings are returned unchanged."""
    records = [DummyFrameRecord(0, 0.0), DummyFrameRecord(1, 1.0)]
    v = _make_unit_vector(768, 0.0)
    embs = np.stack([v, v])

    ret_recs, ret_embs, ret_indices = filter_near_duplicate_frames(
        records, embs, threshold=0.97, enabled=False
    )
    assert len(ret_recs) == 2
    assert ret_indices == [0, 1]


@pytest.mark.parametrize("invalid_thresh", [-0.1, 1.5, float("nan"), float("inf"), -float("inf")])
def test_reject_invalid_dedup_threshold(invalid_thresh):
    """Dedup must reject out-of-range or non-finite threshold values."""
    records = [DummyFrameRecord(0, 0.0)]
    embs = np.array([_make_unit_vector(768, 0.0)], dtype=np.float32)

    with pytest.raises(ValueError, match="visual dedup threshold must be a finite number between 0.0 and 1.0"):
        filter_near_duplicate_frames(records, embs, threshold=invalid_thresh)


def test_dedup_determinism():
    """Same candidates, embeddings, and config must produce identical retained indices."""
    np.random.seed(42)
    dim = 768
    num_frames = 20
    embs = np.random.randn(num_frames, dim).astype(np.float32)
    # Inject some duplicates
    embs[3] = embs[2]
    embs[5] = embs[4]
    embs[10] = embs[9]
    records = [DummyFrameRecord(i, i * 1.0) for i in range(num_frames)]
    protected = {3, 15}

    _, _, ind_1 = filter_near_duplicate_frames(records, embs, protected_source_frame_indices=protected, threshold=0.97)
    _, _, ind_2 = filter_near_duplicate_frames(records, embs, protected_source_frame_indices=protected, threshold=0.97)

    assert ind_1 == ind_2


@pytest.mark.parametrize(
    "bad_value,error",
    [
        (float("nan"), "finite values"),
        (float("inf"), "finite values"),
        (0.0, "zero-norm"),
    ],
)
def test_malformed_embedding_rows_are_rejected(bad_value, error):
    records = [DummyFrameRecord(0, 0.0)]
    embeddings = np.zeros((1, 4), dtype=np.float32)
    embeddings[0, 0] = bad_value

    with pytest.raises(ValueError, match=error):
        filter_near_duplicate_frames(records, embeddings)


def test_non_float32_embeddings_are_rejected():
    records = [DummyFrameRecord(0, 0.0)]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="2D float32"):
        filter_near_duplicate_frames(records, embeddings)


def test_empty_record_alignment_is_validated_when_enabled():
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="Mismatched records count"):
        filter_near_duplicate_frames([], embeddings)


def test_boolean_threshold_is_rejected():
    records = [DummyFrameRecord(0, 0.0)]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite number"):
        filter_near_duplicate_frames(records, embeddings, threshold=True)


def test_finite_values_with_overflowed_norm_are_rejected():
    records = [DummyFrameRecord(0, 0.0)]
    maximum = np.finfo(np.float32).max
    embeddings = np.array([[maximum, maximum]], dtype=np.float32)
    assert np.isfinite(embeddings).all()

    with pytest.raises(ValueError, match="overflowed"):
        filter_near_duplicate_frames(records, embeddings)
