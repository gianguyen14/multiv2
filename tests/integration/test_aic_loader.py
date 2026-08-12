"""Integration tests for AIC Loader."""

import pytest
from pathlib import Path

from backend.app.loaders.aic_loader import create_loader, AICLoader
from backend.app.samplers.base import (
    FixedFPSStrategy,
    ShotPlusFixedFPSStrategy,
    AdaptiveDenseStrategy,
)
from backend.app.shot_detection.base import NullShotDetector


class TestAICLoaderIntegration:
    """Integration tests for the full AIC loader pipeline."""

    @pytest.fixture
    def video_path(self) -> Path:
        """Get the test video path."""
        return Path(__file__).parent.parent / "fixtures" / "test_5s.mp4"

    def test_fixed_1fps_loader(self, video_path):
        """Test loader with FixedFPSStrategy at 1 FPS."""
        loader = create_loader(sampling_type="fixed", fps=1.0, shot_detector_type="none")
        frames = loader.process_video(video_path, allow_metadata_fallback=True)

        assert len(frames) == 5  # 5 seconds at 1 FPS
        assert frames[0].timestamp_ms == 0
        assert frames[4].timestamp_ms == 4000
        assert all(f.shot_id == "shot_0" for f in frames)

    def test_fixed_0_5fps_loader(self, video_path):
        """Test loader with FixedFPSStrategy at 0.5 FPS."""
        loader = create_loader(sampling_type="fixed", fps=0.5, shot_detector_type="none")
        frames = loader.process_video(video_path, allow_metadata_fallback=True)

        assert len(frames) == 3  # 5 seconds at 0.5 FPS = 3 frames (0, 2000, 4000)
        assert frames[0].timestamp_ms == 0
        assert frames[-1].timestamp_ms == 4000

    def test_fixed_2fps_loader(self, video_path):
        """Test loader with FixedFPSStrategy at 2 FPS."""
        loader = create_loader(sampling_type="fixed", fps=2.0, shot_detector_type="none")
        frames = loader.process_video(video_path, allow_metadata_fallback=True)

        # 5 seconds at 2 FPS = 10 frames (0, 500, 1000, ..., 4500)
        assert len(frames) == 10
        assert frames[0].timestamp_ms == 0
        assert frames[-1].timestamp_ms == 4500

    def test_shot_fixed_1fps_loader(self, video_path):
        """Test loader with ShotPlusFixedFPSStrategy."""
        loader = create_loader(sampling_type="shot_fixed", fps=1.0, shot_detector_type="none")
        frames = loader.process_video(video_path, allow_metadata_fallback=True)

        assert len(frames) == 5
        assert all(f.shot_id == "shot_0" for f in frames)

    def test_adaptive_loader(self, video_path):
        """Test loader with AdaptiveDenseStrategy."""
        loader = create_loader(sampling_type="adaptive", shot_detector_type="none")
        frames = loader.process_video(video_path, allow_metadata_fallback=True)

        # 5s shot -> 1 FPS + first/middle/last
        assert len(frames) >= 5
        assert frames[0].timestamp_ms == 0
        assert frames[-1].timestamp_ms == 4999

    def test_deterministic_output(self, video_path):
        """Test that same loader produces identical output."""
        loader = create_loader(sampling_type="fixed", fps=1.0, shot_detector_type="none")
        frames1 = loader.process_video(video_path, allow_metadata_fallback=True)
        frames2 = loader.process_video(video_path, allow_metadata_fallback=True)

        ids1 = [f.frame_id for f in frames1]
        ids2 = [f.frame_id for f in frames2]
        timestamps1 = [f.timestamp_ms for f in frames1]
        timestamps2 = [f.timestamp_ms for f in frames2]

        assert ids1 == ids2
        assert timestamps1 == timestamps2

    def test_metadata_fallback(self, video_path):
        """Test metadata fallback works for test fixture."""
        loader = create_loader(sampling_type="fixed", fps=1.0, shot_detector_type="none")
        frames = loader.process_video(video_path, allow_metadata_fallback=True)

        assert len(frames) == 5
        assert frames[0].video_id == "test_5s"

    def test_shot_detection_disabled(self, video_path):
        """Test that NullShotDetector produces no shot boundaries."""
        loader = AICLoader(
            sampling_strategy=FixedFPSStrategy(1.0),
            shot_detector=NullShotDetector(),
        )
        frames = loader.process_video(video_path, allow_metadata_fallback=True)

        # Should still work, just no shot boundaries detected
        assert len(frames) == 5
        assert all(f.shot_id == "shot_0" for f in frames)

    def test_frame_ids_unique(self, video_path):
        """Test all frame IDs are unique."""
        loader = create_loader(sampling_type="fixed", fps=2.0, shot_detector_type="none")
        frames = loader.process_video(video_path, allow_metadata_fallback=True)

        ids = [f.frame_id for f in frames]
        assert len(ids) == len(set(ids))

    def test_timestamps_monotonic(self, video_path):
        """Test timestamps are monotonically increasing."""
        loader = create_loader(sampling_type="fixed", fps=2.0, shot_detector_type="none")
        frames = loader.process_video(video_path, allow_metadata_fallback=True)

        timestamps = [f.timestamp_ms for f in frames]
        assert timestamps == sorted(timestamps)

    def test_frames_with_shots_method(self, video_path):
        """Test process_video_with_shots method."""
        loader = create_loader(sampling_type="fixed", fps=1.0, shot_detector_type="none")
        frames, shots = loader.process_video_with_shots(video_path, allow_metadata_fallback=True)

        assert len(frames) == 5
        assert len(shots) == 1
        assert shots[0].shot_id == "shot_0"
        assert shots[0].start_ms == 0
        assert shots[0].end_ms == 5000