"""Optional C++17 acceleration facade.

The public functions in this module always have a pure-Python fallback. Runtime
selection is controlled by ``UVR_NATIVE_CORE``:

- ``auto`` (default): use C++ when the extension is importable, otherwise Python.
- ``cpp``: require the C++ extension and propagate native errors.
- ``python``: force the reference Python implementation.

This keeps the project portable while making the accelerated path easy to test
and benchmark against the reference semantics.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - availability depends on build environment
    from . import _core as _cpp_core
    _native_import_error: Exception | None = None
except Exception as exc:  # pragma: no cover - availability depends on build environment
    _cpp_core = None
    _native_import_error = exc

_ALLOWED_MODES = {"auto", "cpp", "python"}


def _mode() -> str:
    mode = os.getenv("UVR_NATIVE_CORE", "auto").strip().lower()
    if mode not in _ALLOWED_MODES:
        raise ValueError(
            f"UVR_NATIVE_CORE must be one of {sorted(_ALLOWED_MODES)}, got {mode!r}"
        )
    return mode


def native_available() -> bool:
    return _cpp_core is not None


def native_status() -> dict[str, object]:
    mode = _mode()
    return {
        "mode": mode,
        "available": native_available(),
        "backend": "cpp" if mode != "python" and native_available() else "python",
        "version": _cpp_core.version() if native_available() else None,
        "import_error": repr(_native_import_error) if _native_import_error else None,
    }


def _use_cpp() -> bool:
    mode = _mode()
    if mode == "python":
        return False
    if _cpp_core is None:
        if mode == "cpp":
            raise RuntimeError(
                "UVR_NATIVE_CORE=cpp requested but backend.app.native._core is unavailable"
            ) from _native_import_error
        return False
    return True


def _native_or_fallback(name: str, fallback, *args):
    if _use_cpp():
        try:
            return getattr(_cpp_core, name)(*args)
        except Exception:
            if _mode() == "cpp":
                raise
            logger.debug("native kernel %s failed; using Python fallback", name, exc_info=True)
    return fallback(*args)


def _smooth_scores_python(
    raw_scores: np.ndarray,
    weight_visual: float,
    weight_temporal: float,
    pool_window: int,
) -> np.ndarray:
    values = np.asarray(raw_scores).reshape(-1)
    if pool_window < 0:
        raise ValueError("pool_window must be >= 0")
    output = np.zeros(len(values), dtype=np.float32)
    for idx in range(len(values)):
        start = max(0, idx - pool_window)
        end = min(len(values), idx + pool_window + 1)
        local_mean = float(np.mean(values[start:end]))
        output[idx] = float(
            weight_visual * float(values[idx]) + weight_temporal * local_mean
        )
    return output


def smooth_scores(
    raw_scores: np.ndarray,
    weight_visual: float,
    weight_temporal: float,
    pool_window: int,
) -> np.ndarray:
    """Apply the TemporalRefiner local pooling rule to a 1-D score array."""

    values = np.ascontiguousarray(np.asarray(raw_scores).reshape(-1))
    if values.dtype not in (np.float32, np.float64):
        values = values.astype(np.float32, copy=False)

    if _use_cpp():
        try:
            blob = _cpp_core.smooth_scores(
                values, float(weight_visual), float(weight_temporal), int(pool_window)
            )
            return np.frombuffer(blob, dtype=np.float32)
        except Exception:
            if _mode() == "cpp":
                raise
            logger.debug("native smooth_scores failed; using Python fallback", exc_info=True)

    return _smooth_scores_python(values, weight_visual, weight_temporal, pool_window)


def _temporal_nms_python(
    video_ids: Sequence[str],
    frame_ids: Sequence[int],
    min_gap: int,
    top_k: int,
) -> list[int]:
    if len(video_ids) != len(frame_ids):
        raise ValueError("video_ids and frame_ids must have the same length")
    if min_gap < 0 or top_k < 0:
        raise ValueError("min_gap and top_k must be >= 0")

    selected: list[int] = []
    selected_by_video: dict[str, list[int]] = {}
    for idx, (video_id, frame_id) in enumerate(zip(video_ids, frame_ids)):
        chosen = selected_by_video.setdefault(str(video_id), [])
        if any(abs(int(frame_id) - prior) < min_gap for prior in chosen):
            continue
        chosen.append(int(frame_id))
        selected.append(idx)
        if len(selected) >= top_k:
            break
    return selected


def temporal_nms_indices(
    video_ids: Sequence[str],
    frame_ids: Sequence[int],
    min_gap: int,
    top_k: int,
) -> list[int]:
    """Return indices retained by stable per-video temporal NMS."""

    result = _native_or_fallback(
        "temporal_nms_indices",
        _temporal_nms_python,
        list(video_ids),
        [int(v) for v in frame_ids],
        int(min_gap),
        int(top_k),
    )
    return [int(v) for v in result]


def _merge_temporal_regions_python(
    frame_ids: Sequence[int],
    scores: Sequence[float],
    delta_frames: int,
    total_frames: int,
    max_regions: int,
) -> list[tuple[int, int, list[int], float]]:
    if len(frame_ids) != len(scores):
        raise ValueError("frame_ids and scores must have the same length")
    if delta_frames < 0 or total_frames <= 0 or max_regions < 0:
        raise ValueError("invalid temporal-region limits")

    raw = []
    for frame_id, score in zip(frame_ids, scores):
        fid = int(frame_id)
        raw.append(
            (
                max(0, fid - int(delta_frames)),
                min(int(total_frames) - 1, fid + int(delta_frames)),
                fid,
                float(score),
            )
        )
    raw.sort(key=lambda item: item[0])

    merged: list[list[object]] = []
    for start, end, fid, score in raw:
        if not merged or start > int(merged[-1][1]):
            merged.append([start, end, [fid], score])
        else:
            merged[-1][1] = max(int(merged[-1][1]), end)
            merged[-1][2].append(fid)  # type: ignore[union-attr]
            merged[-1][3] = max(float(merged[-1][3]), score)

    if len(merged) > max_regions:
        merged.sort(key=lambda item: -float(item[3]))
        merged = merged[:max_regions]
        merged.sort(key=lambda item: int(item[0]))

    return [
        (int(start), int(end), [int(v) for v in frames], float(max_score))
        for start, end, frames, max_score in merged
    ]


def merge_temporal_regions(
    frame_ids: Sequence[int],
    scores: Sequence[float],
    delta_frames: int,
    total_frames: int,
    max_regions: int,
) -> list[tuple[int, int, list[int], float]]:
    """Build and merge bounded candidate windows for one video."""

    result = _native_or_fallback(
        "merge_temporal_regions",
        _merge_temporal_regions_python,
        [int(v) for v in frame_ids],
        [float(v) for v in scores],
        int(delta_frames),
        int(total_frames),
        int(max_regions),
    )
    return [
        (int(start), int(end), [int(v) for v in frames], float(max_score))
        for start, end, frames, max_score in result
    ]


def _align_trake_python(
    frame_ids_by_event: Sequence[Sequence[int]],
    scores_by_event: Sequence[Sequence[float]],
    transition_penalty: float,
    max_gap: int | None,
):
    if not frame_ids_by_event or len(frame_ids_by_event) != len(scores_by_event):
        return None

    events: list[list[tuple[int, float]]] = []
    for frames, scores in zip(frame_ids_by_event, scores_by_event):
        if len(frames) != len(scores):
            raise ValueError("each event must have matching frame and score lengths")
        event = sorted(
            [(int(frame), float(score)) for frame, score in zip(frames, scores)],
            key=lambda item: item[0],
        )
        if not event:
            return None
        events.append(event)

    states: list[tuple[float, list[int]]] = [
        (score, [frame]) for frame, score in events[0]
    ]
    previous = events[0]

    for current in events[1:]:
        next_states: list[tuple[float, list[int]]] = []
        for current_frame, current_score in current:
            choices: list[tuple[float, list[int]]] = []
            for (prior_frame, _), (score, path) in zip(previous, states):
                if not path or score == float("-inf"):
                    continue
                gap = current_frame - prior_frame
                if gap > 0 and (max_gap is None or gap <= max_gap):
                    choices.append(
                        (
                            score
                            + current_score
                            - float(transition_penalty) * gap,
                            path + [current_frame],
                        )
                    )
            next_states.append(
                max(
                    choices,
                    default=(float("-inf"), []),
                    key=lambda item: (item[0], [-v for v in item[1]]),
                )
            )
        previous, states = current, next_states

    valid = [
        (score, path)
        for score, path in states
        if path
        and len(path) == len(frame_ids_by_event)
        and score > float("-inf")
    ]
    if not valid:
        return None
    return max(valid, key=lambda item: (item[0], [-v for v in item[1]]))


def align_trake_events(
    frame_ids_by_event: Sequence[Sequence[int]],
    scores_by_event: Sequence[Sequence[float]],
    transition_penalty: float = 0.0,
    max_gap: int | None = None,
) -> tuple[float, list[int]] | None:
    """Align one video's event candidates with the original TRAKE DP semantics."""

    frames = [[int(v) for v in event] for event in frame_ids_by_event]
    scores = [[float(v) for v in event] for event in scores_by_event]
    result = _native_or_fallback(
        "align_trake_events",
        _align_trake_python,
        frames,
        scores,
        float(transition_penalty),
        max_gap,
    )
    if result is None:
        return None
    score, path = result
    return float(score), [int(v) for v in path]


__all__ = [
    "align_trake_events",
    "merge_temporal_regions",
    "native_available",
    "native_status",
    "smooth_scores",
    "temporal_nms_indices",
]
