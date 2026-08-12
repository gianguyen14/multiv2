"""SamplingStrategy interface for deterministic frame extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from backend.app.schemas.frame import FrameData, ShotData


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for a sampling strategy."""
    video_id: str
    duration_ms: int
    fps: int
    width: int = 0
    height: int = 0
    shot_boundaries: Tuple[int, ...] = ()


class SamplingStrategy(ABC):
    """Abstract base class for deterministic frame sampling strategies.

    All strategies must implement the `sample` method which returns a list of
    FrameData objects with exact timestamps and shot mapping.
    """

    @abstractmethod
    def sample(self, config: SamplingConfig) -> List[FrameData]:
        """Generate deterministic frames for the given video config.

        Args:
            config: SamplingConfig with video metadata and optional shot boundaries.

        Returns:
            List of FrameData objects with exact timestamps, frame_ids, and shot_ids.
        """
        pass

    def _generate_frame_id(self, video_id: str, index: int) -> str:
        """Generate deterministic frame ID."""
        return f"{video_id}_frame_{index:06d}"

    def _assign_shots(self, frames: List[FrameData], shot_boundaries: Tuple[int, ...]) -> List[FrameData]:
        """Assign shot IDs to frames based on shot boundaries.

        Args:
            frames: List of frames with timestamps.
            shot_boundaries: Tuple of shot start timestamps in ms.

        Returns:
            Frames with shot_id populated.
        """
        if not shot_boundaries:
            # No shot boundaries - assign all to shot_0
            return [FrameData(
                video_id=f.video_id,
                frame_id=f.frame_id,
                timestamp_ms=f.timestamp_ms,
                shot_id="shot_0",
                image_path=f.image_path
            ) for f in frames]

        # Ensure boundaries are sorted
        sorted_boundaries = sorted(shot_boundaries)

        updated_frames = []
        for frame in frames:
            # Find which shot this frame belongs to
            shot_idx = 0
            for i, boundary in enumerate(sorted_boundaries):
                if frame.timestamp_ms >= boundary:
                    shot_idx = i
                else:
                    break
            shot_id = f"shot_{shot_idx}"
            updated_frames.append(FrameData(
                video_id=frame.video_id,
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                shot_id=shot_id,
                image_path=frame.image_path
            ))
        return updated_frames


class FixedFPSStrategy(SamplingStrategy):
    """Fixed FPS sampling strategy.

    Samples frames at a fixed rate (e.g., 0.25, 0.5, 1, 2 FPS) throughout
    the entire video duration.
    """

    def __init__(self, target_fps: float):
        """Initialize with target FPS.

        Args:
            target_fps: Frames per second (e.g., 0.25, 0.5, 1, 2)
        """
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.target_fps = target_fps

    def sample(self, config: SamplingConfig) -> List[FrameData]:
        """Sample frames at fixed FPS."""
        if config.duration_ms <= 0:
            return []

        interval_ms = int(1000 / self.target_fps)
        # Calculate frame count: ceil(duration / interval) = (duration + interval - 1) // interval
        frame_count = max(1, (config.duration_ms + interval_ms - 1) // interval_ms)

        frames = []
        for i in range(frame_count):
            timestamp_ms = min(i * interval_ms, config.duration_ms - 1) if config.duration_ms > 0 else 0
            frame_id = self._generate_frame_id(config.video_id, i)
            frames.append(FrameData(
                video_id=config.video_id,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                shot_id="shot_0",  # Will be reassigned by caller if shots exist
                image_path=None
            ))

        # Assign shots if boundaries provided
        return self._assign_shots(frames, config.shot_boundaries)


class ShotPlusFixedFPSStrategy(SamplingStrategy):
    """Shot-aware sampling with fixed FPS within each shot.

    First detects shots, then samples at a fixed FPS within each shot.
    """

    def __init__(self, target_fps: float = 1.0):
        """Initialize with target FPS within shots.

        Args:
            target_fps: Frames per second within each shot.
        """
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.target_fps = target_fps

    def sample(self, config: SamplingConfig) -> List[FrameData]:
        """Sample frames at fixed FPS within each shot."""
        if config.duration_ms <= 0:
            return []

        interval_ms = int(1000 / self.target_fps)
        shot_boundaries = config.shot_boundaries

        if not shot_boundaries:
            # No shot info - fall back to simple fixed FPS
            return FixedFPSStrategy(self.target_fps).sample(config)

        sorted_boundaries = list(sorted(shot_boundaries))
        # Add end as final boundary
        all_boundaries = sorted_boundaries + [config.duration_ms]

        frames = []
        frame_index = 0

        for shot_idx, (start_ms, end_ms) in enumerate(zip(sorted_boundaries, all_boundaries[1:])):
            shot_duration = end_ms - start_ms
            if shot_duration <= 0:
                continue

            # Use ceiling division: (duration + interval - 1) // interval
            shot_frame_count = max(1, (shot_duration + interval_ms - 1) // interval_ms)

            for i in range(shot_frame_count):
                timestamp_ms = min(start_ms + i * interval_ms, end_ms - 1)
                frame_id = self._generate_frame_id(config.video_id, frame_index)
                frames.append(FrameData(
                    video_id=config.video_id,
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    shot_id=f"shot_{shot_idx}",
                    image_path=None
                ))
                frame_index += 1

        return frames


class AdaptiveDenseStrategy(SamplingStrategy):
    """Shot + adaptive dense sampling strategy.

    Sampling rate varies by shot duration:
    - shot < 4s: 2 FPS
    - 4-15s: 1 FPS
    - >15s: 0.5 FPS
    Plus always include first, middle, last frame of each shot.
    """

    def __init__(self):
        # Thresholds in milliseconds
        self.fast_threshold_ms = 4000  # < 4s -> 2 FPS
        self.medium_threshold_ms = 15000  # 4-15s -> 1 FPS
        # > 15s -> 0.5 FPS

    def _get_fps_for_duration(self, duration_ms: int) -> float:
        """Get target FPS based on shot duration."""
        if duration_ms < self.fast_threshold_ms:
            return 2.0
        elif duration_ms <= self.medium_threshold_ms:
            return 1.0
        else:
            return 0.5

    def sample(self, config: SamplingConfig) -> List[FrameData]:
        """Sample frames with adaptive density per shot."""
        if config.duration_ms <= 0:
            return []

        shot_boundaries = config.shot_boundaries

        if not shot_boundaries:
            # No shot info - treat whole video as one shot and apply adaptive logic
            # Determine FPS based on total duration
            target_fps = self._get_fps_for_duration(config.duration_ms)
            interval_ms = int(1000 / target_fps)

            # Calculate regular frames using ceiling division
            regular_count = max(1, (config.duration_ms + interval_ms - 1) // interval_ms)
            regular_timestamps = [
                min(i * interval_ms, config.duration_ms - 1)
                for i in range(regular_count)
            ]

            # Always include first, middle, last
            special_timestamps = [0]
            if config.duration_ms > 0:
                special_timestamps.append(config.duration_ms // 2)
                special_timestamps.append(config.duration_ms - 1)

            # Combine and deduplicate
            all_timestamps = sorted(set(regular_timestamps + special_timestamps))

            frames = []
            for i, ts in enumerate(all_timestamps):
                if ts < 0 or ts >= config.duration_ms:
                    continue
                frame_id = self._generate_frame_id(config.video_id, i)
                frames.append(FrameData(
                    video_id=config.video_id,
                    frame_id=frame_id,
                    timestamp_ms=ts,
                    shot_id="shot_0",
                    image_path=None
                ))

            return frames

        sorted_boundaries = list(sorted(shot_boundaries))
        all_boundaries = sorted_boundaries + [config.duration_ms]

        frames = []
        frame_index = 0

        for shot_idx, (start_ms, end_ms) in enumerate(zip(sorted_boundaries, all_boundaries[1:])):
            shot_duration = end_ms - start_ms
            if shot_duration <= 0:
                continue

            target_fps = self._get_fps_for_duration(shot_duration)
            interval_ms = int(1000 / target_fps)

            # Calculate regular frames using ceiling division
            regular_count = max(1, (shot_duration + interval_ms - 1) // interval_ms)
            regular_timestamps = [
                min(start_ms + i * interval_ms, end_ms - 1)
                for i in range(regular_count)
            ]

            # Always include first, middle, last
            special_timestamps = [start_ms]
            if shot_duration > 0:
                middle_ms = start_ms + shot_duration // 2
                special_timestamps.append(middle_ms)
                special_timestamps.append(end_ms - 1)

            # Combine and deduplicate
            all_timestamps = sorted(set(regular_timestamps + special_timestamps))

            for ts in all_timestamps:
                if ts < start_ms or ts >= end_ms:
                    continue
                frame_id = self._generate_frame_id(config.video_id, frame_index)
                frames.append(FrameData(
                    video_id=config.video_id,
                    frame_id=frame_id,
                    timestamp_ms=ts,
                    shot_id=f"shot_{shot_idx}",
                    image_path=None
                ))
                frame_index += 1

        return frames