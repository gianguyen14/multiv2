from dataclasses import dataclass
from typing import Iterable

from backend.app.video.video_decoder import DecodedFrame


class FrameSamplingError(RuntimeError):
    pass


@dataclass(frozen=True)
class SampledFrame:
    frame: DecodedFrame
    target_timestamp_seconds: float


def iter_sample_frames(frames: Iterable[DecodedFrame], interval_seconds=1.0):
    if interval_seconds <= 0:
        raise ValueError("sample interval must be positive")
    previous = None
    target = 0.0
    last_selected_index = None
    last_timed = None
    for current in frames:
        if current.timestamp_seconds is None:
            continue
        last_timed = current
        if previous is None:
            previous = current
        while target <= current.timestamp_seconds:
            candidate = previous
            if abs(current.timestamp_seconds - target) < abs(previous.timestamp_seconds - target):
                candidate = current
            if candidate.source_frame_index_zero_based != last_selected_index:
                yield SampledFrame(candidate, target)
                last_selected_index = candidate.source_frame_index_zero_based
            target += interval_seconds
        previous = current
    if last_timed is None:
        raise FrameSamplingError("video has no usable frame timestamps")


def sample_frames(frames: Iterable[DecodedFrame], interval_seconds=1.0):
    return list(iter_sample_frames(frames, interval_seconds))
