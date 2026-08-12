"""AIC Loader - integrates metadata, shot detection, and sampling to produce FrameData."""

from pathlib import Path
from typing import List, Optional, Tuple

from backend.app.loaders.metadata_loader import load_metadata, MetadataError
from backend.app.samplers.base import (
    SamplingStrategy,
    FixedFPSStrategy,
    ShotPlusFixedFPSStrategy,
    AdaptiveDenseStrategy,
    SamplingConfig,
)
from backend.app.shot_detection.base import (
    ShotDetector,
    get_shot_detector,
    NullShotDetector,
)
from backend.app.schemas.frame import FrameData, ShotData


class AICLoader:
    """Main loader that orchestrates metadata, shot detection, and sampling."""

    def __init__(
        self,
        sampling_strategy: Optional[SamplingStrategy] = None,
        shot_detector: Optional[ShotDetector] = None,
        shot_detector_type: str = "transnetv2",
    ):
        """Initialize the AIC loader.

        Args:
            sampling_strategy: Strategy for frame sampling. If None, uses FixedFPSStrategy(1.0).
            shot_detector: ShotDetector instance. If None, creates one based on shot_detector_type.
            shot_detector_type: Type of shot detector ("transnetv2", "none").
        """
        self.sampling_strategy = sampling_strategy or FixedFPSStrategy(1.0)
        self.shot_detector = shot_detector or get_shot_detector(shot_detector_type)

    def process_video(
        self,
        video_path: Path,
        *,
        allow_metadata_fallback: bool = False,
    ) -> List[FrameData]:
        """Process a video and return list of FrameData objects.

        Args:
            video_path: Path to the video file.
            allow_metadata_fallback: Allow metadata fallback for test fixtures.

        Returns:
            List of FrameData objects with deterministic IDs, timestamps, and shot mapping.

        Raises:
            FileNotFoundError: If video file doesn't exist.
            MetadataError: If metadata cannot be extracted.
        """
        # Load metadata
        manifest = load_metadata(video_path, allow_fallback=allow_metadata_fallback)

        # Detect shots
        shot_boundaries = self.shot_detector.detect_shot_boundaries(video_path)
        shot_boundaries_tuple = tuple(shot_boundaries) if shot_boundaries else ()

        # Create sampling config
        config = SamplingConfig(
            video_id=manifest["video_id"],
            duration_ms=manifest["duration_ms"],
            fps=manifest["fps"],
            shot_boundaries=shot_boundaries_tuple,
        )

        # Sample frames
        frames = self.sampling_strategy.sample(config)

        return frames

    def process_video_with_shots(
        self,
        video_path: Path,
        *,
        allow_metadata_fallback: bool = False,
    ) -> Tuple[List[FrameData], List[ShotData]]:
        """Process video and return both frames and shot data.

        Args:
            video_path: Path to the video file.
            allow_metadata_fallback: Allow metadata fallback for test fixtures.

        Returns:
            Tuple of (frames, shots) where frames is a list of FrameData
            and shots is a list of ShotData.
        """
        # Load metadata to get actual video duration
        manifest = load_metadata(video_path, allow_fallback=allow_metadata_fallback)
        duration_ms = manifest["duration_ms"]

        # Get shot boundaries from detector
        shot_boundaries = self.shot_detector.detect_shot_boundaries(video_path)
        if shot_boundaries:
            sorted_boundaries = sorted(shot_boundaries)
            shot_boundaries_with_end = sorted_boundaries + [duration_ms]
        else:
            # Single shot covering entire video
            sorted_boundaries = [0]
            shot_boundaries_with_end = [0, duration_ms]

        frames = self.process_video(video_path, allow_metadata_fallback=allow_metadata_fallback)

        # Group frames by shot_id
        shots_dict = {}
        for frame in frames:
            if frame.shot_id not in shots_dict:
                shots_dict[frame.shot_id] = []
            shots_dict[frame.shot_id].append(frame)

        shots = []
        for shot_idx, (start_ms, end_ms) in enumerate(zip(shot_boundaries_with_end[:-1], shot_boundaries_with_end[1:])):
            shot_id = f"shot_{shot_idx}"
            shot_frames = shots_dict.get(shot_id, [])
            if shot_frames:
                shot = ShotData(
                    shot_id=shot_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    frames=tuple(shot_frames),
                )
                shots.append(shot)

        return frames, shots


def create_loader(
    sampling_type: str = "fixed",
    fps: float = 1.0,
    shot_detector_type: str = "transnetv2",
) -> AICLoader:
    """Factory function to create an AICLoader with specified strategies.

    Args:
        sampling_type: One of "fixed", "shot_fixed", "adaptive"
        fps: Target FPS for fixed strategies
        shot_detector_type: "transnetv2" or "none"

    Returns:
        Configured AICLoader instance.
    """
    if sampling_type == "fixed":
        strategy = FixedFPSStrategy(fps)
    elif sampling_type == "shot_fixed":
        strategy = ShotPlusFixedFPSStrategy(fps)
    elif sampling_type == "adaptive":
        strategy = AdaptiveDenseStrategy()
    else:
        raise ValueError(f"Unknown sampling_type: {sampling_type}")

    return AICLoader(
        sampling_strategy=strategy,
        shot_detector_type=shot_detector_type,
    )