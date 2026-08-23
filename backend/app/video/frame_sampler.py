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
    if shot_boundaries:
        for shot_idx, (s_start, s_end) in enumerate(shot_boundaries):
            if not isinstance(s_start, (int, float)) or not isinstance(s_end, (int, float)):
                continue
            if math.isnan(s_start) or math.isnan(s_end) or math.isinf(s_start) or math.isinf(s_end):
                continue
            if s_end < s_start or s_start < 0:
                continue
            mid_sec = (s_start + s_end) / 2.0
            # Prefer candidate frames within the shot's temporal window [s_start, s_end]
            in_window = [f for f in frame_list if s_start <= f.timestamp_seconds <= s_end]
            candidates = in_window if in_window else frame_list
            best_frame = min(candidates, key=lambda f: (abs(f.timestamp_seconds - mid_sec), f.source_frame_index_zero_based))
            idx = best_frame.source_frame_index_zero_based
            shot_map[idx] = SampledFrame(best_frame, mid_sec, sampling_reason="shot", shot_id=shot_idx)

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
    protected = set()
    for s_start, s_end in shot_boundaries:
        if not isinstance(s_start, (int, float)) or not isinstance(s_end, (int, float)):
            continue
        if math.isnan(s_start) or math.isnan(s_end) or math.isinf(s_start) or math.isinf(s_end):
            continue
        if s_end < s_start or s_start < 0:
            continue
        mid_sec = (s_start + s_end) / 2.0
        in_window = [f for f in frame_list if s_start <= f.timestamp_seconds <= s_end]
        candidates = in_window if in_window else frame_list
        best_frame = min(candidates, key=lambda f: (abs(f.timestamp_seconds - mid_sec), f.source_frame_index_zero_based))
        protected.add(best_frame.source_frame_index_zero_based)
    return protected
