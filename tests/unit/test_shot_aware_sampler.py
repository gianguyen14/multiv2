import math
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock
import pytest
from PIL import Image

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.shot_detection.base import ShotDetector
from backend.app.video.m15_ingestion_pipeline import _normalize_detected_shots
from backend.app.video.frame_sampler import (
    FrameSamplingError,
    SampledFrame,
    iter_sample_frames,
    iter_sample_sparse_shot_frames,
    sample_frames,
    sample_sparse_shot_frames,
    sample_sparse_shot_frames_with_protection,
)
from backend.app.video.video_decoder import DecodedFrame


def _create_synthetic_frames(duration_seconds: float, fps: float = 30.0) -> List[DecodedFrame]:
    """Create deterministic stream of DecodedFrame objects."""
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


def _full_list_reference(frames, interval_seconds, shot_boundaries):
    """Test-only reference for the pre-streaming selection semantics."""
    timed = [frame for frame in frames if frame.timestamp_seconds is not None]
    periodic = {
        item.frame.source_frame_index_zero_based: item
        for item in iter_sample_frames(timed, interval_seconds)
    }
    shots = {}
    for shot_id, (start, end) in enumerate(shot_boundaries):
        in_window = [
            frame for frame in timed if start <= frame.timestamp_seconds <= end
        ]
        if not in_window:
            continue
        midpoint = (start + end) / 2.0
        selected = min(
            in_window,
            key=lambda frame: (
                abs(frame.timestamp_seconds - midpoint),
                frame.source_frame_index_zero_based,
            ),
        )
        shots[selected.source_frame_index_zero_based] = SampledFrame(
            selected, midpoint, "shot", shot_id
        )

    merged = {}
    for source_index in set(periodic) | set(shots):
        if source_index in periodic and source_index in shots:
            periodic_item = periodic[source_index]
            merged[source_index] = SampledFrame(
                periodic_item.frame,
                periodic_item.target_timestamp_seconds,
                "periodic+shot",
                shots[source_index].shot_id,
            )
        else:
            merged[source_index] = periodic.get(source_index, shots.get(source_index))
    return (
        sorted(
            merged.values(),
            key=lambda item: item.frame.source_frame_index_zero_based,
        ),
        set(shots),
    )


class MockShotDetector(ShotDetector):
    """Deterministic mock detector for unit tests without external models."""

    def __init__(self, shots: List[Tuple[int, int]]):
        self.shots = shots
        self.call_count = 0

    def detect_shots(self, video_path: Path) -> List[Tuple[int, int]]:
        self.call_count += 1
        return self.shots

    def detect_shot_boundaries(self, video_path: Path) -> List[int]:
        self.call_count += 1
        return [s[0] for s in self.shots]


def test_short_shot_between_periodic_samples_retained():
    """Verify that a short scene (2.0s -> 3.0s) falling between 5s periodic samples is retained."""
    frames = _create_synthetic_frames(duration_seconds=10.0, fps=30.0)
    # Periodic at 5s would only take t=0.0s (frame 0), t=5.0s (frame 150), t=10.0s (frame 300)
    # Shot [2.0, 3.0] has midpoint 2.5s (frame 75)
    shots = [(2.0, 3.0)]

    sampled = sample_sparse_shot_frames(frames, interval_seconds=5.0, shot_boundaries=shots)
    indices = [s.frame.source_frame_index_zero_based for s in sampled]
    timestamps = [s.frame.timestamp_seconds for s in sampled]

    assert 75 in indices  # midpoint frame 75 is preserved
    assert len(sampled) == 4  # 3 periodic (0, 150, 300) + 1 shot midpoint (75)
    assert indices == [0, 75, 150, 300]
    assert timestamps == [0.0, 2.5, 5.0, 10.0]


def test_midpoint_overlaps_periodic_sample_exact_dedup():
    """If a shot midpoint coincides with a periodic sample, output contains it exactly once."""
    frames = _create_synthetic_frames(duration_seconds=10.0, fps=30.0)
    # Periodic at 5s takes t=0.0s, 5.0s, 10.0s
    # Shot [4.0, 6.0] has midpoint 5.0s (frame 150)
    shots = [(4.0, 6.0)]

    sampled = sample_sparse_shot_frames(frames, interval_seconds=5.0, shot_boundaries=shots)
    indices = [s.frame.source_frame_index_zero_based for s in sampled]

    assert len(sampled) == 3
    assert indices == [0, 150, 300]
    # Frame 150 appears exactly once
    assert indices.count(150) == 1


def test_multiple_shots_coverage_invariant():
    """Verify 100% shot coverage across multiple consecutive and isolated shots."""
    frames = _create_synthetic_frames(duration_seconds=20.0, fps=30.0)
    shots = [
        (1.0, 2.0),    # midpoint 1.5s -> frame 45
        (3.0, 4.0),    # midpoint 3.5s -> frame 105
        (7.0, 9.0),    # midpoint 8.0s -> frame 240
        (12.0, 14.0),  # midpoint 13.0s -> frame 390
        (16.0, 18.0),  # midpoint 17.0s -> frame 510
    ]

    sampled = sample_sparse_shot_frames(frames, interval_seconds=5.0, shot_boundaries=shots)
    sampled_times = [s.frame.timestamp_seconds for s in sampled]
    sampled_indices = [s.frame.source_frame_index_zero_based for s in sampled]

    # Proving invariant: every valid shot has >= 1 retained sample in its temporal interval
    for shot_idx, (s_start, s_end) in enumerate(shots):
        in_shot = [t for t in sampled_times if s_start <= t <= s_end]
        assert len(in_shot) >= 1, f"Shot #{shot_idx} [{s_start}, {s_end}] was not represented in output!"

    # Verify chronological ordering
    assert sampled_indices == sorted(sampled_indices)
    assert len(sampled_indices) == len(set(sampled_indices))


def test_chronological_ordering_and_no_duplicates():
    """Verify strictly increasing frame indices and timestamps in merged output."""
    frames = _create_synthetic_frames(duration_seconds=15.0, fps=25.0)
    shots = [(1.2, 2.4), (4.8, 5.2), (8.0, 11.0), (13.5, 14.5)]

    sampled = sample_sparse_shot_frames(frames, interval_seconds=4.0, shot_boundaries=shots)
    indices = [s.frame.source_frame_index_zero_based for s in sampled]

    assert indices == sorted(indices)
    for i in range(len(indices) - 1):
        assert indices[i] < indices[i + 1]


def test_boundary_shots_near_video_start_and_end():
    """Boundary shots near t=0.0s and t=duration must resolve to valid source frames."""
    duration = 12.0
    frames = _create_synthetic_frames(duration_seconds=duration, fps=30.0)
    shots = [
        (0.0, 0.5),    # near start (midpoint 0.25s)
        (11.5, 12.0),  # near end (midpoint 11.75s)
    ]

    sampled = sample_sparse_shot_frames(frames, interval_seconds=5.0, shot_boundaries=shots)
    for s in sampled:
        assert 0 <= s.frame.source_frame_index_zero_based <= len(frames) - 1
        assert 0.0 <= s.frame.timestamp_seconds <= duration


def test_invalid_detector_output_handled_gracefully():
    """Malformed shot intervals (end < start, negative, NaN, Inf) are skipped safely."""
    frames = _create_synthetic_frames(duration_seconds=10.0, fps=30.0)
    malformed_shots = [
        (5.0, 2.0),           # end < start
        (-2.0, 3.0),          # negative start
        (float("nan"), 4.0),  # NaN
        (1.0, float("inf")),  # Inf
        (2.0, 3.0),           # valid shot (midpoint 2.5s)
    ]

    sampled = sample_sparse_shot_frames(frames, interval_seconds=5.0, shot_boundaries=malformed_shots)
    indices = [s.frame.source_frame_index_zero_based for s in sampled]

    # Valid shot [2.0, 3.0] frame 75 is retained, malformed entries skipped
    assert 75 in indices
    assert indices == [0, 75, 150, 300]


def test_malformed_shape_and_out_of_timeline_shots_are_skipped():
    frames = _create_synthetic_frames(duration_seconds=10.0, fps=30.0)
    malformed_shots = [
        None,
        (1.0,),
        (1.0, 2.0, 3.0),
        (False, 2.0),
        (20.0, 21.0),
        (2.0, 3.0),
    ]

    sampled = sample_sparse_shot_frames(
        frames, interval_seconds=5.0, shot_boundaries=malformed_shots
    )

    assert [item.frame.source_frame_index_zero_based for item in sampled] == [
        0,
        75,
        150,
        300,
    ]
    shot_item = next(item for item in sampled if item.sampling_reason == "shot")
    assert shot_item.shot_id == 5


def test_large_finite_shot_boundaries_do_not_overflow_midpoint():
    frames = _create_synthetic_frames(duration_seconds=1.0, fps=2.0)

    sampled = sample_sparse_shot_frames(
        iter(frames), interval_seconds=5.0, shot_boundaries=[(1e308, 1e308)]
    )

    assert [item.frame.source_frame_index_zero_based for item in sampled] == [0]


def test_normal_shot_midpoint_preserves_historical_float_value_exactly():
    start = 0.04091934103444243
    end = 2.331778862250263
    frames = [
        DecodedFrame(index, index, timestamp, 2, 2, Image.new("RGB", (2, 2)))
        for index, timestamp in enumerate((0.0, 1.010976, 3.0))
    ]

    sampled = sample_sparse_shot_frames(
        iter(frames), interval_seconds=100.0, shot_boundaries=[(start, end)]
    )
    shot = next(item for item in sampled if item.sampling_reason == "shot")

    assert shot.target_timestamp_seconds == (start + end) / 2.0


def test_detector_millisecond_normalization_preserves_valid_intervals():
    raw = [None, (1,), (False, 1000), (2000, 1000), (1000, 2000)]

    assert _normalize_detected_shots(raw) == [(1.0, 2.0)]


def test_sparse_shot_determinism():
    """Identical frames, interval, and shots produce identical merged frame output."""
    frames_1 = _create_synthetic_frames(duration_seconds=10.0, fps=30.0)
    frames_2 = _create_synthetic_frames(duration_seconds=10.0, fps=30.0)
    shots = [(1.5, 2.5), (6.0, 8.0)]

    sampled_1 = sample_sparse_shot_frames(frames_1, interval_seconds=5.0, shot_boundaries=shots)
    sampled_2 = sample_sparse_shot_frames(frames_2, interval_seconds=5.0, shot_boundaries=shots)

    assert len(sampled_1) == len(sampled_2)
    for s1, s2 in zip(sampled_1, sampled_2):
        assert s1.frame.source_frame_index_zero_based == s2.frame.source_frame_index_zero_based
        assert s1.frame.timestamp_seconds == s2.frame.timestamp_seconds
        assert s1.target_timestamp_seconds == s2.target_timestamp_seconds


def test_sparse_shot_sampling_does_not_retain_every_decoded_frame():
    """Sparse sampling must keep memory proportional to selected frames, not video length."""

    class TrackedFrame:
        live = 0
        peak = 0

        def __init__(self, index):
            type(self).live += 1
            type(self).peak = max(type(self).peak, type(self).live)
            self.source_frame_index_zero_based = index
            self.timestamp_seconds = index / 10.0

        def __del__(self):
            type(self).live -= 1

    def decoded_frames():
        for index in range(10_000):
            yield TrackedFrame(index)

    sampled, protected = sample_sparse_shot_frames_with_protection(
        decoded_frames(),
        interval_seconds=5.0,
        shot_boundaries=[(0.0, 999.9)],
    )

    assert len(sampled) == 201
    assert len(protected) == 1
    assert TrackedFrame.peak < 250


def test_streaming_matches_full_list_reference_on_irregular_timestamps():
    timestamps = [0.08, 0.42, 1.21, 1.82, 2.45, 3.67, 5.10]
    frames = [
        DecodedFrame(index, index, timestamp, 2, 2, Image.new("RGB", (2, 2)))
        for index, timestamp in enumerate(timestamps)
    ]
    shots = [(0.0, 0.2), (1.3, 2.3), (2.3, 4.0), (4.0, 5.2)]

    expected, expected_protected = _full_list_reference(frames, 1.0, shots)
    actual, actual_protected = sample_sparse_shot_frames_with_protection(
        (frame for frame in frames), 1.0, shots
    )

    pack = lambda rows: [
        (
            item.frame.source_frame_index_zero_based,
            item.frame.timestamp_seconds,
            item.target_timestamp_seconds,
            item.sampling_reason,
            item.shot_id,
        )
        for item in rows
    ]
    assert pack(actual) == pack(expected)
    assert actual_protected == expected_protected


def test_short_unrepresented_shot_never_selects_outside_its_window():
    frames = [
        DecodedFrame(index, index, timestamp, 2, 2, Image.new("RGB", (2, 2)))
        for index, timestamp in enumerate((0.0, 1.0, 3.0))
    ]

    sampled, protected = sample_sparse_shot_frames_with_protection(
        iter(frames), 100.0, [(1.4, 1.6)]
    )

    assert protected == set()
    assert all(item.sampling_reason == "periodic" for item in sampled)


def test_same_frame_multi_shot_collision_keeps_later_scalar_shot_id():
    frames = _create_synthetic_frames(duration_seconds=1.0, fps=2.0)

    sampled, protected = sample_sparse_shot_frames_with_protection(
        iter(frames), 100.0, [(0.4, 0.6), (0.45, 0.55)]
    )

    shot = next(item for item in sampled if item.sampling_reason == "shot")
    assert shot.frame.source_frame_index_zero_based == 1
    assert shot.shot_id == 1
    assert protected == {1}


def test_pipeline_legacy_mode_does_not_invoke_detector(tmp_path):
    """In legacy mode, VideoIngestionPipeline does NOT invoke the shot detector."""
    from backend.app.video.m15_ingestion_pipeline import VideoIngestionPipeline

    mock_encoder = MagicMock()
    mock_encoder.embedding_dim = 768
    mock_encoder.encode_image.return_value = [[0.1] * 768]
    mock_encoder.identity.return_value = {
        "provider": "test", "model_name": "mock", "embedding_dim": 768, "normalization": "l2"
    }

    mock_detector = MockShotDetector(shots=[(1000, 2000), (3000, 4000)])
    config_legacy = VideoIngestConfig(
        processed_root=tmp_path / "legacy_out",
        visual_sampling_mode="legacy",
        sample_interval_seconds=1.0,
    )

    pipeline = VideoIngestionPipeline(
        encoder=mock_encoder,
        config=config_legacy,
        shot_detector=mock_detector,
    )

    # Verify detector was not called during initialization
    assert mock_detector.call_count == 0
