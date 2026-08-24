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

    # Retain only selected frames while streaming the decoder.  A decoded RGB
    # frame is several megabytes for normal AIC video, so materializing a whole
    # video here makes sparse sampling use more memory than legacy sampling.
    periodic_map: dict[int, SampledFrame] = {}
    valid_shots = []
    for shot_idx, boundary in enumerate(shot_boundaries or []):
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
        ):
            continue
        if s_end < s_start or s_start < 0:
            continue
        valid_shots.append(((s_start + s_end) / 2.0, shot_idx, s_start, s_end))
    valid_shots.sort(key=lambda item: (item[0], item[1]))

    periodic_target = 0.0
    last_periodic_index = None
    previous = None
    last_timed = None
    next_shot = 0
    shot_choices: list[tuple[int, SampledFrame]] = []

    for current in frames:
        if current.timestamp_seconds is None:
            continue
        last_timed = current
        if previous is None:
            previous = current

        while periodic_target <= current.timestamp_seconds:
            candidate = previous
            if abs(current.timestamp_seconds - periodic_target) < abs(previous.timestamp_seconds - periodic_target):
                candidate = current
            idx = candidate.source_frame_index_zero_based
            if idx != last_periodic_index:
                periodic_map[idx] = SampledFrame(
                    candidate,
                    periodic_target,
                    sampling_reason="periodic",
                    shot_id=None,
                )
                last_periodic_index = idx
            periodic_target += interval_seconds

        while next_shot < len(valid_shots) and valid_shots[next_shot][0] <= current.timestamp_seconds:
            mid_sec, shot_idx, s_start, s_end = valid_shots[next_shot]
            adjacent = [previous] if previous is current else [previous, current]
            in_window = [
                frame for frame in adjacent
                if s_start <= frame.timestamp_seconds <= s_end
            ]
            if in_window:
                best_frame = min(
                    in_window,
                    key=lambda frame: (
                        abs(frame.timestamp_seconds - mid_sec),
                        frame.source_frame_index_zero_based,
                    ),
                )
                shot_choices.append((shot_idx, SampledFrame(
                    best_frame,
                    mid_sec,
                    sampling_reason="shot",
                    shot_id=shot_idx,
                )))
            next_shot += 1
        previous = current

    if last_timed is None:
        raise FrameSamplingError("video has no usable frame timestamps")

    # A trailing shot is valid only when the final decoded frame lies inside
    # the detector interval. Out-of-timeline intervals are ignored.
    while next_shot < len(valid_shots):
        mid_sec, shot_idx, s_start, s_end = valid_shots[next_shot]
        if s_start <= last_timed.timestamp_seconds <= s_end:
            shot_choices.append((shot_idx, SampledFrame(
                last_timed,
                mid_sec,
                sampling_reason="shot",
                shot_id=shot_idx,
            )))
        next_shot += 1

    # Preserve the historical collision rule: when multiple input shots map
    # to one source frame, the later input shot supplies the shot_id.
    shot_map: dict[int, SampledFrame] = {}
    for _, item in sorted(shot_choices, key=lambda choice: choice[0]):
        shot_map[item.frame.source_frame_index_zero_based] = item

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
    _, protected = sample_sparse_shot_frames_with_protection(
        frames,
        interval_seconds=1e300,
        shot_boundaries=shot_boundaries,
    )
    return protected
