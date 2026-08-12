"""Unit tests for sampling strategies."""

import pytest
from backend.app.samplers.base import (
    SamplingConfig,
    FixedFPSStrategy,
    ShotPlusFixedFPSStrategy,
    AdaptiveDenseStrategy,
)
from backend.app.schemas.frame import FrameData


def test_sampling_config():
    """Test SamplingConfig dataclass."""
    config = SamplingConfig(
        video_id="test_video",
        duration_ms=5000,
        fps=25,
        shot_boundaries=(0, 2500),
    )
    assert config.video_id == "test_video"
    assert config.duration_ms == 5000
    assert config.fps == 25
    assert config.shot_boundaries == (0, 2500)


class TestFixedFPSStrategy:
    """Tests for FixedFPSStrategy."""

    def test_1fps_basic(self):
        """Test 1 FPS sampling."""
        config = SamplingConfig(
            video_id="test",
            duration_ms=5000,
            fps=25,
        )
        strategy = FixedFPSStrategy(target_fps=1.0)
        frames = strategy.sample(config)

        assert len(frames) == 5  # 5 seconds at 1 FPS
        assert frames[0].timestamp_ms == 0
        assert frames[1].timestamp_ms == 1000
        assert frames[4].timestamp_ms == 4000
        assert all(f.shot_id == "shot_0" for f in frames)

    def test_0_5fps(self):
        """Test 0.5 FPS sampling."""
        config = SamplingConfig(
            video_id="test",
            duration_ms=10000,
            fps=25,
        )
        strategy = FixedFPSStrategy(target_fps=0.5)
        frames = strategy.sample(config)

        assert len(frames) == 5  # 10 seconds at 0.5 FPS = 5 frames
        assert frames[0].timestamp_ms == 0
        assert frames[1].timestamp_ms == 2000

    def test_2fps(self):
        """Test 2 FPS sampling."""
        config = SamplingConfig(
            video_id="test",
            duration_ms=3000,
            fps=25,
        )
        strategy = FixedFPSStrategy(target_fps=2.0)
        frames = strategy.sample(config)

        assert len(frames) == 6  # 3 seconds at 2 FPS = 6 frames
        assert frames[0].timestamp_ms == 0
        assert frames[1].timestamp_ms == 500

    def test_zero_duration(self):
        """Test zero duration returns empty list."""
        config = SamplingConfig(
            video_id="test",
            duration_ms=0,
            fps=25,
        )
        strategy = FixedFPSStrategy(target_fps=1.0)
        frames = strategy.sample(config)
        assert frames == []

    def test_invalid_fps(self):
        """Test invalid FPS raises error."""
        with pytest.raises(ValueError):
            FixedFPSStrategy(target_fps=0)
        with pytest.raises(ValueError):
            FixedFPSStrategy(target_fps=-1)


class TestShotPlusFixedFPSStrategy:
    """Tests for ShotPlusFixedFPSStrategy."""

    def test_shot_fixed_1fps(self):
        """Test shot-aware 1 FPS sampling."""
        config = SamplingConfig(
            video_id="test",
            duration_ms=10000,
            fps=25,
            shot_boundaries=(0, 4000, 7000),
        )
        strategy = ShotPlusFixedFPSStrategy(target_fps=1.0)
        frames = strategy.sample(config)

        # Shot 0: 0-4000ms = 4 frames (0, 1000, 2000, 3000)
        # Shot 1: 4000-7000ms = 3 frames (4000, 5000, 6000)
        # Shot 2: 7000-10000ms = 3 frames (7000, 8000, 9000)
        assert len(frames) == 10

        # Check shot assignments
        shot_0_frames = [f for f in frames if f.shot_id == "shot_0"]
        shot_1_frames = [f for f in frames if f.shot_id == "shot_1"]
        shot_2_frames = [f for f in frames if f.shot_id == "shot_2"]

        assert len(shot_0_frames) == 4
        assert len(shot_1_frames) == 3
        assert len(shot_2_frames) == 3

        # Check timestamps
        assert shot_0_frames[0].timestamp_ms == 0
        assert shot_0_frames[3].timestamp_ms == 3000
        assert shot_1_frames[0].timestamp_ms == 4000
        assert shot_2_frames[0].timestamp_ms == 7000

    def test_no_shots_fallback(self):
        """Test fallback to simple fixed FPS when no shots."""
        config = SamplingConfig(
            video_id="test",
            duration_ms=5000,
            fps=25,
            shot_boundaries=(),
        )
        strategy = ShotPlusFixedFPSStrategy(target_fps=1.0)
        frames = strategy.sample(config)

        assert len(frames) == 5
        assert all(f.shot_id == "shot_0" for f in frames)


class TestAdaptiveDenseStrategy:
    """Tests for AdaptiveDenseStrategy."""

    def test_short_shot_2fps(self):
        """Test short shot (<4s) gets 2 FPS + first/middle/last."""
        config = SamplingConfig(
            video_id="test",
            duration_ms=3000,
            fps=25,
            shot_boundaries=(0,),
        )
        strategy = AdaptiveDenseStrategy()
        frames = strategy.sample(config)

        # Shot < 4s -> 2 FPS = interval 500ms
        # 3000ms at 2 FPS: 0, 500, 1000, 1500, 2000, 2500 (6 frames)
        # Plus first (0), middle (1500), last (2999) - but these may already be included
        # Expected unique frames
        assert len(frames) >= 6
        assert frames[0].timestamp_ms == 0
        assert frames[-1].timestamp_ms == 2999  # last frame

        # Check all frames belong to shot_0
        assert all(f.shot_id == "shot_0" for f in frames)

    def test_medium_shot_1fps(self):
        """Test medium shot (4-15s) gets 1 FPS + first/middle/last."""
        config = SamplingConfig(
            video_id="test",
            duration_ms=10000,
            fps=25,
            shot_boundaries=(0,),
        )
        strategy = AdaptiveDenseStrategy()
        frames = strategy.sample(config)

        # 10s at 1 FPS = 10 frames + special
        assert len(frames) >= 10
        assert frames[0].timestamp_ms == 0
        assert frames[-1].timestamp_ms == 9999

    def test_long_shot_0_5fps(self):
        """Test long shot (>15s) gets 0.5 FPS + first/middle/last."""
        config = SamplingConfig(
            video_id="test",
            duration_ms=20000,
            fps=25,
            shot_boundaries=(0,),
        )
        strategy = AdaptiveDenseStrategy()
        frames = strategy.sample(config)

        # 20s at 0.5 FPS = 10 frames + special
        assert len(frames) >= 10
        assert frames[0].timestamp_ms == 0
        assert frames[-1].timestamp_ms == 19999

    def test_multiple_shots(self):
        """Test adaptive sampling with multiple shots of different lengths."""
        config = SamplingConfig(
            video_id="test",
            duration_ms=20000,
            fps=25,
            shot_boundaries=(0, 3000, 10000),
        )
        strategy = AdaptiveDenseStrategy()
        frames = strategy.sample(config)

        # Shot 0: 3s -> 2 FPS
        # Shot 1: 7s -> 1 FPS
        # Shot 2: 10s -> 1 FPS
        assert len(frames) > 0

        shot_0_frames = [f for f in frames if f.shot_id == "shot_0"]
        shot_1_frames = [f for f in frames if f.shot_id == "shot_1"]
        shot_2_frames = [f for f in frames if f.shot_id == "shot_2"]

        assert len(shot_0_frames) >= 6  # 3s at 2 FPS
        assert len(shot_1_frames) >= 7  # 7s at 1 FPS
        assert len(shot_2_frames) >= 10  # 10s at 1 FPS

    def test_no_shots_fallback(self):
        """Test fallback to default 1 FPS when no shots."""
        config = SamplingConfig(
            video_id="test",
            duration_ms=5000,
            fps=25,
            shot_boundaries=(),
        )
        strategy = AdaptiveDenseStrategy()
        frames = strategy.sample(config)

        assert len(frames) >= 5
        assert all(f.shot_id == "shot_0" for f in frames)

    def test_duplicate_prevention(self):
        """Test that first/middle/last don't create duplicates."""
        config = SamplingConfig(
            video_id="test",
            duration_ms=2000,
            fps=25,
            shot_boundaries=(0,),
        )
        strategy = AdaptiveDenseStrategy()
        frames = strategy.sample(config)

        # Should not have duplicate timestamps
        timestamps = [f.timestamp_ms for f in frames]
        assert len(timestamps) == len(set(timestamps))


class TestFrameIdGeneration:
    """Test frame ID generation is deterministic."""

    def test_deterministic_frame_ids(self):
        """Same config produces same frame IDs."""
        config = SamplingConfig(
            video_id="test_video",
            duration_ms=5000,
            fps=25,
        )
        strategy = FixedFPSStrategy(target_fps=1.0)
        frames1 = strategy.sample(config)
        frames2 = strategy.sample(config)

        ids1 = [f.frame_id for f in frames1]
        ids2 = [f.frame_id for f in frames2]
        assert ids1 == ids2

    def test_unique_frame_ids(self):
        """All frame IDs are unique within a video."""
        config = SamplingConfig(
            video_id="test",
            duration_ms=10000,
            fps=25,
        )
        strategy = FixedFPSStrategy(target_fps=2.0)
        frames = strategy.sample(config)

        ids = [f.frame_id for f in frames]
        assert len(ids) == len(set(ids))