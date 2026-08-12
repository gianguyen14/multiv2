"""FrameData schema - the canonical data structure for video frames."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FrameData:
    """Immutable frame data structure.

    Attributes:
        video_id: Unique video identifier.
        frame_id: Unique frame identifier within the video.
        timestamp_ms: Exact timestamp in milliseconds from video start.
        shot_id: Shot identifier this frame belongs to.
        image_path: Optional path to the extracted frame image.
    """
    video_id: str
    frame_id: str
    timestamp_ms: int
    shot_id: str
    image_path: Optional[str] = None


@dataclass(frozen=True)
class ShotData:
    """Immutable shot data structure.

    Attributes:
        shot_id: Unique shot identifier.
        start_ms: Shot start timestamp in milliseconds.
        end_ms: Shot end timestamp in milliseconds.
        frames: List of FrameData belonging to this shot.
    """
    shot_id: str
    start_ms: int
    end_ms: int
    frames: tuple[FrameData, ...] = ()

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms