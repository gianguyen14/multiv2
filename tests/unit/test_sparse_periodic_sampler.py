import math
import os
import pytest
from PIL import Image

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.video.frame_sampler import iter_sample_frames, sample_frames
from backend.app.video.video_decoder import DecodedFrame


def _create_synthetic_frames(duration_seconds: float, fps: float = 30.0):
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


def test_legacy_mode_default_1s_sampling():
    """Verify legacy 1.0s interval sampling produces 1 frame per second."""
    frames = _create_synthetic_frames(duration_seconds=5.0, fps=30.0)
    sampled = sample_frames(frames, interval_seconds=1.0)
    
    # Expected targets: 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
    assert len(sampled) == 6
    target_times = [s.target_timestamp_seconds for s in sampled]
    assert target_times == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    indices = [s.frame.source_frame_index_zero_based for s in sampled]
    assert indices == [0, 30, 60, 90, 120, 150]


def test_sparse_3s_mode_sampling():
    """Verify 3.0s sparse periodic sampling produces frames at 3s intervals."""
    frames = _create_synthetic_frames(duration_seconds=10.0, fps=30.0)
    sampled = sample_frames(frames, interval_seconds=3.0)
    
    # Expected targets: 0.0, 3.0, 6.0, 9.0
    assert len(sampled) == 4
    target_times = [s.target_timestamp_seconds for s in sampled]
    assert target_times == [0.0, 3.0, 6.0, 9.0]
    
    indices = [s.frame.source_frame_index_zero_based for s in sampled]
    assert indices == [0, 90, 180, 270]


def test_sparse_5s_mode_sampling():
    """Verify 5.0s sparse periodic sampling produces frames at 5s intervals."""
    frames = _create_synthetic_frames(duration_seconds=15.0, fps=30.0)
    sampled = sample_frames(frames, interval_seconds=5.0)
    
    # Expected targets: 0.0, 5.0, 10.0, 15.0
    assert len(sampled) == 4
    target_times = [s.target_timestamp_seconds for s in sampled]
    assert target_times == [0.0, 5.0, 10.0, 15.0]
    
    indices = [s.frame.source_frame_index_zero_based for s in sampled]
    assert indices == [0, 150, 300, 450]


def test_non_divisible_duration_behavior():
    """Verify non-divisible video durations (e.g. 12.0s duration with 5.0s interval)."""
    frames = _create_synthetic_frames(duration_seconds=12.0, fps=30.0)
    sampled = sample_frames(frames, interval_seconds=5.0)
    
    # Targets should be 0.0, 5.0, 10.0 (15.0 is beyond 12.0s and must NOT be emitted)
    assert len(sampled) == 3
    target_times = [s.target_timestamp_seconds for s in sampled]
    assert target_times == [0.0, 5.0, 10.0]
    
    # Frame indices must map correctly to t=0s, 5s, 10s
    indices = [s.frame.source_frame_index_zero_based for s in sampled]
    assert indices == [0, 150, 300]
    
    # Ensure no frame timestamp exceeds video duration
    for s in sampled:
        assert s.frame.timestamp_seconds <= 12.0


def test_sampling_determinism():
    """Same video stream and interval must produce identical sampled sequence."""
    frames_a = _create_synthetic_frames(duration_seconds=8.0, fps=25.0)
    frames_b = _create_synthetic_frames(duration_seconds=8.0, fps=25.0)
    
    sampled_a = sample_frames(frames_a, interval_seconds=3.0)
    sampled_b = sample_frames(frames_b, interval_seconds=3.0)
    
    assert len(sampled_a) == len(sampled_b)
    for sa, sb in zip(sampled_a, sampled_b):
        assert sa.target_timestamp_seconds == sb.target_timestamp_seconds
        assert sa.frame.source_frame_index_zero_based == sb.frame.source_frame_index_zero_based
        assert sa.frame.timestamp_seconds == sb.frame.timestamp_seconds


@pytest.mark.parametrize("invalid_interval", [False, True, 0, -1.0, -5.0, float("nan"), float("inf"), -float("inf")])
def test_reject_invalid_sampler_interval(invalid_interval):
    """Frame sampler must reject non-positive or non-finite intervals."""
    frames = _create_synthetic_frames(duration_seconds=5.0, fps=30.0)
    with pytest.raises(ValueError, match="sample interval must be a positive and finite number"):
        sample_frames(frames, interval_seconds=invalid_interval)


def test_video_ingest_config_modes():
    """Verify VideoIngestConfig handles legacy and sparse_shot modes with effective interval."""
    # Legacy default
    cfg_legacy = VideoIngestConfig(visual_sampling_mode="legacy", sample_interval_seconds=1.0)
    assert cfg_legacy.effective_sample_interval_seconds == 1.0
    assert cfg_legacy.visual_sampling_mode == "legacy"

    # Sparse shot mode
    cfg_sparse = VideoIngestConfig(
        visual_sampling_mode="sparse_shot",
        visual_global_sample_seconds=5.0,
        sample_interval_seconds=1.0,
    )
    assert cfg_sparse.effective_sample_interval_seconds == 5.0
    assert cfg_sparse.visual_sampling_mode == "sparse_shot"

    # Invalid mode
    with pytest.raises(ValueError, match="unsupported visual sampling mode"):
        VideoIngestConfig(visual_sampling_mode="unsupported_mode")

    # Invalid global sample seconds
    with pytest.raises(ValueError, match="visual global sample seconds must be a positive and finite number"):
        VideoIngestConfig(visual_global_sample_seconds=0.0)

    with pytest.raises(ValueError, match="visual global sample seconds must be a positive and finite number"):
        VideoIngestConfig(visual_global_sample_seconds=float("nan"))


def test_video_ingest_config_from_env(monkeypatch):
    """Verify VideoIngestConfig.from_env parses VISUAL_SAMPLING_MODE and VISUAL_GLOBAL_SAMPLE_SECONDS."""
    monkeypatch.setenv("VISUAL_SAMPLING_MODE", "sparse_shot")
    monkeypatch.setenv("VISUAL_GLOBAL_SAMPLE_SECONDS", "3.5")
    
    cfg = VideoIngestConfig.from_env()
    assert cfg.visual_sampling_mode == "sparse_shot"
    assert cfg.visual_global_sample_seconds == 3.5
    assert cfg.effective_sample_interval_seconds == 3.5

    # Invalid env value raises
    monkeypatch.setenv("VISUAL_GLOBAL_SAMPLE_SECONDS", "invalid_number")
    with pytest.raises(ValueError, match="invalid VISUAL_GLOBAL_SAMPLE_SECONDS"):
        VideoIngestConfig.from_env()


def test_video_ingest_config_rejects_invalid_boolean_environment(monkeypatch):
    monkeypatch.setenv("VISUAL_DEDUP_ENABLED", "sometimes")
    with pytest.raises(ValueError, match="VISUAL_DEDUP_ENABLED must be one of"):
        VideoIngestConfig.from_env()
