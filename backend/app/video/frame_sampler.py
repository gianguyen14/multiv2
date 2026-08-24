import math
from dataclasses import dataclass
from typing import Iterable

from backend.app.video.video_decoder import DecodedFrame


class FrameSamplingError(RuntimeError):
    pass


@dataclass(frozen=True)
class SampledFrame:
    frame: DecodedFrame
    target_timestamp_seconds: float
    sampling_reason: str = "periodic"
    shot_id: int | None = None


def iter_sample_frames(frames: Iterable[DecodedFrame], interval_seconds: float = 1.0):
    if interval_seconds is None or not isinstance(interval_seconds, (int, float)) or math.isnan(interval_seconds) or math.isinf(interval_seconds) or interval_seconds <= 0:
        raise ValueError("sample interval must be a positive and finite number")
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
                yield SampledFrame(candidate, target, sampling_reason="periodic", shot_id=None)
                last_selected_index = candidate.source_frame_index_zero_based
            target += interval_seconds
        previous = current
    if last_timed is None:
        raise FrameSamplingError("video has no usable frame timestamps")


def sample_frames(frames: Iterable[DecodedFrame], interval_seconds: float = 1.0):
    return list(iter_sample_frames(frames, interval_seconds))


def _shot_representatives(
    frame_list: list[DecodedFrame],
    shot_boundaries: list[tuple[float, float]] | None,
):
    """Yield valid shot ordinals and their nearest in-shot midpoint frames."""
    if not shot_boundaries:
        return
    for shot_idx, boundary in enumerate(shot_boundaries):
        if not isinstance(boundary, (tuple, list)) or len(boundary) != 2:
            continue
        s_start, s_end = boundary
        if (
            not isinstance(s_start, (int, float))
            or isinstance(s_start, bool)
            or not isinstance(s_end, (int, float))
            or isinstance(s_end, bool)
            or not math.isfinite(s_start)
            or not math.isfinite(s_end)
            or s_start < 0
            or s_end < s_start
        ):
            continue
        in_window = [
            frame
            for frame in frame_list
            if s_start <= frame.timestamp_seconds <= s_end
        ]
        if not in_window:
            # A detector interval outside the decoded timeline is not a valid shot.
            continue
        midpoint = (s_start + s_end) / 2.0
        best_frame = min(
            in_window,
            key=lambda frame: (
                abs(frame.timestamp_seconds - midpoint),
                frame.source_frame_index_zero_based,
            ),
        )
        yield shot_idx, midpoint, best_frame


def sample_sparse_shot_frames_with_protection(
    frames: Iterable[DecodedFrame],
    interval_seconds: float = 5.0,
    shot_boundaries: list[tuple[float, float]] | None = None,
) -> tuple[list[SampledFrame], set[int]]:
    """Combine sparse periodic sampling with one midpoint representative per shot.

    Args:
        frames: Stream or list of DecodedFrame objects.
        interval_seconds: Periodic temporal sampling interval in seconds (default 5.0).
        shot_boundaries: Optional list of (start_seconds, end_seconds) for detected shots.

    Returns:
        Tuple of (sorted_sampled_frames, protected_source_frame_indices).
    """
    if interval_seconds is None or not isinstance(interval_seconds, (int, float)) or math.isnan(interval_seconds) or math.isinf(interval_seconds) or interval_seconds <= 0:
        raise ValueError("sample interval must be a positive and finite number")

    frame_list = [f for f in frames if f.timestamp_seconds is not None]
    if not frame_list:
        raise FrameSamplingError("video has no usable frame timestamps")

    # 1. Periodic sampling
    periodic_map: dict[int, SampledFrame] = {}
    for item in iter_sample_frames(frame_list, interval_seconds):
        periodic_map[item.frame.source_frame_index_zero_based] = SampledFrame(
            item.frame, item.target_timestamp_seconds, sampling_reason="periodic", shot_id=None
        )

    # 2. Shot midpoint sampling (if shot boundaries are provided)
    shot_map: dict[int, SampledFrame] = {}
    for shot_idx, midpoint, best_frame in _shot_representatives(
        frame_list, shot_boundaries
    ):
        idx = best_frame.source_frame_index_zero_based
        shot_map[idx] = SampledFrame(
            best_frame, midpoint, sampling_reason="shot", shot_id=shot_idx
        )

    # 3. Merge & Deduplicate exact source frame indices with provenance tracking
    merged_map: dict[int, SampledFrame] = {}
    all_indices = set(periodic_map.keys()) | set(shot_map.keys())
    for idx in all_indices:
        in_p = idx in periodic_map
        in_s = idx in shot_map
        if in_p and not in_s:
            merged_map[idx] = periodic_map[idx]
        elif in_s and not in_p:
            merged_map[idx] = shot_map[idx]
        else:  # in_p and in_s
            p_item = periodic_map[idx]
            s_item = shot_map[idx]
            merged_map[idx] = SampledFrame(
                p_item.frame,
                p_item.target_timestamp_seconds,
                sampling_reason="periodic+shot",
                shot_id=s_item.shot_id,
            )

    # 4. Sort chronologically by source_frame_index_zero_based
    sorted_sampled = sorted(merged_map.values(), key=lambda item: item.frame.source_frame_index_zero_based)
    return sorted_sampled, set(shot_map.keys())



def sample_sparse_shot_frames(
    frames: Iterable[DecodedFrame],
    interval_seconds: float = 5.0,
    shot_boundaries: list[tuple[float, float]] | None = None,
) -> list[SampledFrame]:
    """Combine sparse periodic sampling with one midpoint representative per shot."""
    sampled, _ = sample_sparse_shot_frames_with_protection(frames, interval_seconds, shot_boundaries)
    return sampled


def iter_sample_sparse_shot_frames(
    frames: Iterable[DecodedFrame],
    interval_seconds: float = 5.0,
    shot_boundaries: list[tuple[float, float]] | None = None,
):
    """Yield sparse periodic and shot midpoint frames in chronological order."""
    yield from sample_sparse_shot_frames(frames, interval_seconds, shot_boundaries)



def extract_shot_representative_indices(
    frames: Iterable[DecodedFrame],
    shot_boundaries: list[tuple[float, float]] | None = None,
) -> set[int]:
    """Identify the exact source frame indices selected as shot midpoint representatives."""
    if not shot_boundaries:
        return set()
    frame_list = [f for f in frames if f.timestamp_seconds is not None]
    if not frame_list:
        return set()
    return {
        frame.source_frame_index_zero_based
        for _, _, frame in _shot_representatives(frame_list, shot_boundaries)
    }
