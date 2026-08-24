"""Focused unit tests for Step 8 Local Dense Candidate Refinement."""

from dataclasses import dataclass
import math
from typing import Any, List
from unittest.mock import MagicMock
import numpy as np
import pytest

from backend.app.config.local_refine_config import LocalRefineConfig
from backend.app.retrieval.local_refinement import (
    CoarseCandidate,
    MergedRegion,
    RefinedCandidate,
    generate_candidate_regions,
    generate_local_timestamps,
    refine_coarse_candidates,
)


@dataclass
class SyntheticLocalFrame:
    timestamp_seconds: float
    source_frame_index_zero_based: int
    frame_uid: str


# ============================================================================
# 1. PURE REGION GENERATION & MERGING TESTS
# ============================================================================

def test_region_generation_single_candidate():
    """T=20, W=10 -> [10, 30]"""
    c = CoarseCandidate(video_id="v1", timestamp_seconds=20.0, coarse_score=0.9, coarse_rank=0)
    regions = generate_candidate_regions([c], window_seconds=10.0)
    assert len(regions) == 1
    assert regions[0].video_id == "v1"
    assert regions[0].start_seconds == 10.0
    assert regions[0].end_seconds == 30.0
    assert regions[0].best_coarse_rank == 0


def test_region_generation_clamp_start():
    """T=4, W=10 -> start clamped to 0.0 -> [0.0, 14.0]"""
    c = CoarseCandidate(video_id="v1", timestamp_seconds=4.0, coarse_score=0.9, coarse_rank=0)
    regions = generate_candidate_regions([c], window_seconds=10.0)
    assert len(regions) == 1
    assert regions[0].start_seconds == 0.0
    assert regions[0].end_seconds == 14.0


def test_region_generation_clamp_end_with_duration():
    """T=96, W=10, duration=100 -> end clamped to 100.0 -> [86.0, 100.0]"""
    c = CoarseCandidate(video_id="v1", timestamp_seconds=96.0, coarse_score=0.9, coarse_rank=0)
    regions = generate_candidate_regions([c], window_seconds=10.0, duration_by_video={"v1": 100.0})
    assert len(regions) == 1
    assert regions[0].start_seconds == 86.0
    assert regions[0].end_seconds == 100.0


def test_region_generation_overlapping_same_video():
    """Candidate at 20s [10,30] and 25s [15,35] merge to [10,35]"""
    c1 = CoarseCandidate(video_id="v1", timestamp_seconds=20.0, coarse_score=0.9, coarse_rank=0)
    c2 = CoarseCandidate(video_id="v1", timestamp_seconds=25.0, coarse_score=0.8, coarse_rank=1)
    regions = generate_candidate_regions([c1, c2], window_seconds=10.0)
    assert len(regions) == 1
    assert regions[0].video_id == "v1"
    assert regions[0].start_seconds == 10.0
    assert regions[0].end_seconds == 35.0
    assert regions[0].best_coarse_rank == 0
    assert len(regions[0].origin_candidates) == 2


def test_region_generation_touching_windows():
    """Windows [0, 10] and [10, 20] touch -> merge to [0, 20]"""
    c1 = CoarseCandidate(video_id="v1", timestamp_seconds=5.0, coarse_score=0.9, coarse_rank=0)  # [0, 10] with W=5
    c2 = CoarseCandidate(video_id="v1", timestamp_seconds=15.0, coarse_score=0.8, coarse_rank=1)  # [10, 20] with W=5
    regions = generate_candidate_regions([c1, c2], window_seconds=5.0)
    assert len(regions) == 1
    assert regions[0].start_seconds == 0.0
    assert regions[0].end_seconds == 20.0


def test_region_generation_separate_same_video():
    """Separated windows on same video remain distinct."""
    c1 = CoarseCandidate(video_id="v1", timestamp_seconds=10.0, coarse_score=0.9, coarse_rank=0)  # [5, 15] with W=5
    c2 = CoarseCandidate(video_id="v1", timestamp_seconds=30.0, coarse_score=0.8, coarse_rank=1)  # [25, 35] with W=5
    regions = generate_candidate_regions([c1, c2], window_seconds=5.0)
    assert len(regions) == 2
    assert (regions[0].start_seconds, regions[0].end_seconds) == (5.0, 15.0)
    assert (regions[1].start_seconds, regions[1].end_seconds) == (25.0, 35.0)


def test_region_generation_cross_video_isolation():
    """Windows on different videos never merge, even with identical timestamps."""
    c1 = CoarseCandidate(video_id="v1", timestamp_seconds=10.0, coarse_score=0.9, coarse_rank=0)
    c2 = CoarseCandidate(video_id="v2", timestamp_seconds=10.0, coarse_score=0.8, coarse_rank=1)
    regions = generate_candidate_regions([c1, c2], window_seconds=10.0)
    assert len(regions) == 2
    assert {r.video_id for r in regions} == {"v1", "v2"}


# ============================================================================
# 2. MAX-REGION TESTS & MERGE-BEFORE-LIMIT ORDER
# ============================================================================

def test_max_regions_applied_after_merging():
    """6 candidates that merge into 2 regions should both be retained when max_regions=5."""
    # Group 1: 3 overlapping candidates on v1 around 20s
    v1_cands = [
        CoarseCandidate(video_id="v1", timestamp_seconds=20.0, coarse_score=0.9, coarse_rank=0),
        CoarseCandidate(video_id="v1", timestamp_seconds=22.0, coarse_score=0.85, coarse_rank=1),
        CoarseCandidate(video_id="v1", timestamp_seconds=24.0, coarse_score=0.8, coarse_rank=2),
    ]
    # Group 2: 3 overlapping candidates on v2 around 50s
    v2_cands = [
        CoarseCandidate(video_id="v2", timestamp_seconds=50.0, coarse_score=0.75, coarse_rank=3),
        CoarseCandidate(video_id="v2", timestamp_seconds=52.0, coarse_score=0.7, coarse_rank=4),
        CoarseCandidate(video_id="v2", timestamp_seconds=54.0, coarse_score=0.65, coarse_rank=5),
    ]
    candidates = v1_cands + v2_cands
    regions = generate_candidate_regions(candidates, window_seconds=10.0, max_regions=5)
    assert len(regions) == 2
    assert regions[0].video_id == "v1"
    assert regions[1].video_id == "v2"


def test_max_regions_truncates_to_strongest():
    """When >5 independent merged regions exist, keep only the 5 with best coarse rank."""
    candidates = [
        CoarseCandidate(video_id=f"v{i}", timestamp_seconds=100.0 * i, coarse_score=1.0 - 0.1 * i, coarse_rank=i)
        for i in range(8)
    ]
    regions = generate_candidate_regions(candidates, window_seconds=5.0, max_regions=5)
    assert len(regions) == 5
    assert [r.video_id for r in regions] == ["v0", "v1", "v2", "v3", "v4"]
    assert [r.best_coarse_rank for r in regions] == [0, 1, 2, 3, 4]


def test_region_generation_rejects_invalid_max_regions():
    candidate = CoarseCandidate(video_id="v1", timestamp_seconds=10.0)
    with pytest.raises(ValueError, match="max_regions must be an integer >= 1"):
        generate_candidate_regions([candidate], max_regions=0)


# ============================================================================
# 3. LOCAL TIMESTAMP GENERATION TESTS
# ============================================================================

def test_timestamp_generation_exact_intervals():
    """0..2 @ 0.5 -> 0.0, 0.5, 1.0, 1.5, 2.0"""
    ts = generate_local_timestamps(0.0, 2.0, interval_seconds=0.5)
    assert ts == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_timestamp_generation_non_divisible_end():
    """0..2.2 @ 0.5 -> 0.0, 0.5, 1.0, 1.5, 2.0 (no sample > 2.2)"""
    ts = generate_local_timestamps(0.0, 2.2, interval_seconds=0.5)
    assert ts == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert all(t <= 2.2 for t in ts)


def test_timestamp_generation_deterministic_no_duplicates():
    ts1 = generate_local_timestamps(10.0, 30.0, interval_seconds=0.5)
    ts2 = generate_local_timestamps(10.0, 30.0, interval_seconds=0.5)
    assert ts1 == ts2
    assert len(ts1) == len(set(ts1))
    assert ts1 == sorted(ts1)


# ============================================================================
# 4. REFINEMENT SCORING, BEST-FRAME SELECTION & TIE-BREAKING
# ============================================================================

def test_refinement_selects_highest_similarity_frame():
    """Verify frame with highest cosine similarity to query is selected."""
    q_vec = np.zeros(768, dtype=np.float32)
    q_vec[0] = 1.0  # target direction

    # Frames: f0 (sim 0.5), f1 (sim 0.99), f2 (sim 0.2)
    f0 = SyntheticLocalFrame(timestamp_seconds=10.0, source_frame_index_zero_based=100, frame_uid="v1:000000100")
    f1 = SyntheticLocalFrame(timestamp_seconds=10.5, source_frame_index_zero_based=105, frame_uid="v1:000000105")
    f2 = SyntheticLocalFrame(timestamp_seconds=11.0, source_frame_index_zero_based=110, frame_uid="v1:000000110")
    frames_list = [f0, f1, f2]

    e0 = np.zeros(768, dtype=np.float32); e0[0] = 0.5; e0[1] = np.sqrt(1 - 0.5**2)
    e1 = np.zeros(768, dtype=np.float32); e1[0] = 0.99; e1[1] = np.sqrt(1 - 0.99**2)
    e2 = np.zeros(768, dtype=np.float32); e2[0] = 0.2; e2[1] = np.sqrt(1 - 0.2**2)
    embs_matrix = np.stack([e0, e1, e2])

    frame_provider = MagicMock(return_value=frames_list)
    embedding_provider = MagicMock(return_value=embs_matrix)

    candidate = CoarseCandidate(video_id="v1", timestamp_seconds=10.5, coarse_score=0.8, coarse_rank=0)
    config = LocalRefineConfig(enabled=True, window_seconds=10.0, interval_seconds=0.5, max_regions=5)

    refined = refine_coarse_candidates(
        candidates=[candidate],
        query_embedding=q_vec,
        frame_provider=frame_provider,
        embedding_provider=embedding_provider,
        config=config,
    )

    assert len(refined) == 1
    assert refined[0].video_id == "v1"
    assert refined[0].refined_timestamp_seconds == 10.5
    assert refined[0].refined_source_frame_index == 105
    assert refined[0].refined_frame_id == "v1:000000105"
    assert math.isclose(refined[0].local_similarity, 0.99, abs_tol=1e-4)


def test_refinement_tie_break_earliest_timestamp():
    """When two frames have equal top similarity, earliest timestamp wins."""
    q_vec = np.zeros(768, dtype=np.float32)
    q_vec[0] = 1.0

    f0 = SyntheticLocalFrame(timestamp_seconds=10.0, source_frame_index_zero_based=100, frame_uid="v1:000000100")
    f1 = SyntheticLocalFrame(timestamp_seconds=11.0, source_frame_index_zero_based=110, frame_uid="v1:000000110")

    # Identical top similarity 0.95
    e0 = np.zeros(768, dtype=np.float32); e0[0] = 0.95; e0[1] = np.sqrt(1 - 0.95**2)
    e1 = np.zeros(768, dtype=np.float32); e1[0] = 0.95; e1[1] = np.sqrt(1 - 0.95**2)
    embs_matrix = np.stack([e0, e1])

    frame_provider = MagicMock(return_value=[f0, f1])
    embedding_provider = MagicMock(return_value=embs_matrix)

    candidate = CoarseCandidate(video_id="v1", timestamp_seconds=10.0, coarse_score=0.8, coarse_rank=0)
    config = LocalRefineConfig(enabled=True)

    refined = refine_coarse_candidates(
        candidates=[candidate],
        query_embedding=q_vec,
        frame_provider=frame_provider,
        embedding_provider=embedding_provider,
        config=config,
    )

    assert len(refined) == 1
    assert refined[0].refined_timestamp_seconds == 10.0  # Earlier timestamp selected
    assert refined[0].refined_source_frame_index == 100


def test_merged_region_uses_best_ranked_origin_candidate():
    q_vec = np.array([1.0, 0.0], dtype=np.float32)
    frame = SyntheticLocalFrame(22.0, 220, "v1:000000220")
    earlier_but_weaker = CoarseCandidate(
        video_id="v1", timestamp_seconds=20.0, coarse_rank=5
    )
    later_but_stronger = CoarseCandidate(
        video_id="v1", timestamp_seconds=25.0, coarse_rank=1
    )

    refined = refine_coarse_candidates(
        [earlier_but_weaker, later_but_stronger],
        q_vec,
        MagicMock(return_value=[frame]),
        MagicMock(return_value=np.array([[1.0, 0.0]], dtype=np.float32)),
        config=LocalRefineConfig(enabled=True),
    )

    assert refined[0].origin_candidate is later_but_stronger


# ============================================================================
# 5. PROVIDER CALL COUNT & WORK BOUND TESTS
# ============================================================================

def test_provider_called_once_per_merged_region():
    """Overlapping candidates merging to one region call frame_provider exactly once."""
    c1 = CoarseCandidate(video_id="v1", timestamp_seconds=20.0, coarse_score=0.9, coarse_rank=0)
    c2 = CoarseCandidate(video_id="v1", timestamp_seconds=25.0, coarse_score=0.85, coarse_rank=1)

    f = SyntheticLocalFrame(timestamp_seconds=20.0, source_frame_index_zero_based=200, frame_uid="v1:000000200")
    frame_provider = MagicMock(return_value=[f])
    embedding_provider = MagicMock(return_value=np.ones((1, 768), dtype=np.float32))

    config = LocalRefineConfig(enabled=True, window_seconds=10.0)
    q_vec = np.ones(768, dtype=np.float32)

    refined = refine_coarse_candidates(
        candidates=[c1, c2],
        query_embedding=q_vec,
        frame_provider=frame_provider,
        embedding_provider=embedding_provider,
        config=config,
    )

    # Frame provider called only 1 time for merged region [10, 35]
    assert frame_provider.call_count == 1
    call_args = frame_provider.call_args[0]
    assert call_args[0] == "v1"
    # Target timestamps cover 10.0 to 35.0 (span 25s @ 0.5s = 51 timestamps)
    assert len(call_args[1]) == 51


def test_max_default_work_bound():
    """5 full default regions (span 20s @ 0.5s = 41 timestamps each) bound total work to <= 205 timestamps."""
    total_timestamps_requested = 0

    def mock_frame_provider(vid: str, timestamps: List[float]):
        nonlocal total_timestamps_requested
        total_timestamps_requested += len(timestamps)
        return [SyntheticLocalFrame(ts, int(ts * 30), f"{vid}:{int(ts*30):09d}") for ts in timestamps]

    def mock_emb_provider(frames: List[Any]):
        return np.ones((len(frames), 768), dtype=np.float32)

    candidates = [
        CoarseCandidate(video_id=f"v{i}", timestamp_seconds=100.0 * (i + 1), coarse_score=1.0 - 0.1 * i, coarse_rank=i)
        for i in range(5)
    ]
    config = LocalRefineConfig(enabled=True, window_seconds=10.0, interval_seconds=0.5, max_regions=5)
    q_vec = np.ones(768, dtype=np.float32)

    refined = refine_coarse_candidates(
        candidates=candidates,
        query_embedding=q_vec,
        frame_provider=mock_frame_provider,
        embedding_provider=mock_emb_provider,
        config=config,
    )

    assert len(refined) == 5
    assert total_timestamps_requested == 5 * 41 == 205


# ============================================================================
# 6. CONFIG VALIDATION & ERROR HANDLING
# ============================================================================

def test_config_validation_rejections():
    with pytest.raises(ValueError, match="window_seconds must be a positive and finite number"):
        LocalRefineConfig(window_seconds=0.0)
    with pytest.raises(ValueError, match="window_seconds must be a positive and finite number"):
        LocalRefineConfig(window_seconds=float("nan"))
    with pytest.raises(ValueError, match="interval_seconds must be a positive and finite number"):
        LocalRefineConfig(interval_seconds=-0.5)
    with pytest.raises(ValueError, match="max_regions must be an integer >= 1"):
        LocalRefineConfig(max_regions=0)
    with pytest.raises(ValueError, match="max_regions must be an integer >= 1"):
        LocalRefineConfig(max_regions=True)  # bool is not allowed as int


def test_invalid_embedding_inputs_rejected():
    candidate = CoarseCandidate(video_id="v1", timestamp_seconds=10.0)
    config = LocalRefineConfig(enabled=True)
    f = SyntheticLocalFrame(timestamp_seconds=10.0, source_frame_index_zero_based=100, frame_uid="v1:000000100")
    frame_provider = MagicMock(return_value=[f])

    # 1. Query embedding with NaNs
    q_nan = np.full(768, float("nan"), dtype=np.float32)
    with pytest.raises(ValueError, match="query_embedding contains non-finite values"):
        refine_coarse_candidates([candidate], q_nan, frame_provider, MagicMock(), config=config)

    # 2. Query embedding with wrong shape (3D)
    q_3d = np.zeros((1, 2, 768), dtype=np.float32)
    with pytest.raises(ValueError, match="query_embedding must be a 1D vector"):
        refine_coarse_candidates([candidate], q_3d, frame_provider, MagicMock(), config=config)

    # 3. Local embedding wrong dimension (e.g. 512 vs 768)
    q_vec = np.ones(768, dtype=np.float32)
    emb_wrong_dim = MagicMock(return_value=np.ones((1, 512), dtype=np.float32))
    with pytest.raises(ValueError, match="Dimension mismatch between local embeddings"):
        refine_coarse_candidates([candidate], q_vec, frame_provider, emb_wrong_dim, config=config)

    # 4. Zero-norm local vectors do not produce meaningful cosine scores
    zero_vector = MagicMock(return_value=np.zeros((1, 768), dtype=np.float32))
    with pytest.raises(ValueError, match="zero-norm vector"):
        refine_coarse_candidates(
            [candidate], q_vec, frame_provider, zero_vector, config=config
        )

    # 5. Finite float32 vectors whose norms overflow are also invalid
    maximum = np.finfo(np.float32).max
    with pytest.raises(ValueError, match="norm is zero or invalid"):
        refine_coarse_candidates(
            [candidate],
            np.array([maximum, maximum], dtype=np.float32),
            frame_provider,
            MagicMock(),
            config=config,
        )
    overflowed_vector = MagicMock(
        return_value=np.array([[maximum] * 768], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="overflowed"):
        refine_coarse_candidates(
            [candidate], q_vec, frame_provider, overflowed_vector, config=config
        )


def test_empty_candidates_and_disabled_mode_invariants():
    q_vec = np.ones(768, dtype=np.float32)
    frame_provider = MagicMock()
    emb_provider = MagicMock()

    # Empty candidates
    res_empty = refine_coarse_candidates([], q_vec, frame_provider, emb_provider, config=LocalRefineConfig(enabled=True))
    assert res_empty == []
    frame_provider.assert_not_called()
    emb_provider.assert_not_called()

    # Disabled mode
    cand = CoarseCandidate(video_id="v1", timestamp_seconds=10.0)
    res_disabled = refine_coarse_candidates([cand], q_vec, frame_provider, emb_provider, config=LocalRefineConfig(enabled=False))
    assert res_disabled == []
    frame_provider.assert_not_called()
    emb_provider.assert_not_called()


def test_provider_returns_no_frames_gracefully_skipped():
    """When frame_provider returns empty list for a region, the region is skipped without crashing."""
    q_vec = np.ones(768, dtype=np.float32)
    frame_provider = MagicMock(return_value=[])
    emb_provider = MagicMock()

    cand = CoarseCandidate(video_id="v1", timestamp_seconds=10.0)
    config = LocalRefineConfig(enabled=True)

    refined = refine_coarse_candidates([cand], q_vec, frame_provider, emb_provider, config=config)
    assert refined == []
    emb_provider.assert_not_called()
