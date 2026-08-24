"""Local Dense Candidate Refinement Module.

Refines global sparse candidates by evaluating fine-grained local frames within
temporal candidate regions around the highest-scoring coarse results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np

from backend.app.config.local_refine_config import LocalRefineConfig


@dataclass(frozen=True)
class CoarseCandidate:
    """Represents a coarse global retrieval candidate."""

    video_id: str
    timestamp_seconds: float
    coarse_score: float = 0.0
    coarse_rank: int = 0
    frame_id: Optional[str] = None
    source_frame_index_zero_based: Optional[int] = None
    payload: Optional[Dict[str, Any]] = None


@dataclass
class MergedRegion:
    """Represents a merged temporal refinement window for a video."""

    video_id: str
    start_seconds: float
    end_seconds: float
    best_coarse_rank: int
    origin_candidates: List[CoarseCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class RefinedCandidate:
    """Represents the outcome of local temporal refinement for a region."""

    video_id: str
    refined_timestamp_seconds: float
    refined_source_frame_index: Optional[int]
    refined_frame_id: Optional[str]
    local_similarity: float
    origin_candidate: Optional[CoarseCandidate] = None
    region: Optional[MergedRegion] = None


def generate_candidate_regions(
    candidates: Sequence[CoarseCandidate],
    window_seconds: float = 10.0,
    duration_by_video: Optional[Dict[str, float]] = None,
    max_regions: int = 5,
) -> List[MergedRegion]:
    """Build, clamp, and merge overlapping temporal windows per video, then rank and truncate.

    Args:
        candidates: Sequence of coarse candidates.
        window_seconds: Half-window size W in seconds (window is [T - W, T + W]).
        duration_by_video: Optional map of video_id -> duration_seconds for end-clamping.
        max_regions: Maximum number of merged regions to return.

    Returns:
        List of at most max_regions merged regions sorted by priority.
    """
    if not candidates:
        return []

    if window_seconds <= 0 or math.isnan(window_seconds) or math.isinf(window_seconds):
        raise ValueError("window_seconds must be a positive finite number")
    if (
        not isinstance(max_regions, int)
        or isinstance(max_regions, bool)
        or max_regions < 1
    ):
        raise ValueError("max_regions must be an integer >= 1")

    # 1. Group candidates by video_id
    by_video: Dict[str, List[CoarseCandidate]] = {}
    for c in candidates:
        by_video.setdefault(c.video_id, []).append(c)

    all_merged_regions: List[MergedRegion] = []

    # 2. For each video, generate raw windows and merge overlaps/touches
    for vid, cand_list in by_video.items():
        dur = duration_by_video.get(vid) if duration_by_video else None

        # Build raw windows
        raw_windows: List[MergedRegion] = []
        for c in cand_list:
            start = max(0.0, c.timestamp_seconds - window_seconds)
            end = c.timestamp_seconds + window_seconds
            if dur is not None:
                end = min(dur, end)
            if end < start:
                end = start
            raw_windows.append(
                MergedRegion(
                    video_id=vid,
                    start_seconds=start,
                    end_seconds=end,
                    best_coarse_rank=c.coarse_rank,
                    origin_candidates=[c],
                )
            )

        # Sort windows chronologically by start, then end
        raw_windows.sort(key=lambda r: (r.start_seconds, r.end_seconds))

        # Merge overlapping or touching windows
        merged_for_video: List[MergedRegion] = []
        for win in raw_windows:
            if not merged_for_video:
                merged_for_video.append(win)
            else:
                last = merged_for_video[-1]
                # Overlapping or touching: win.start_seconds <= last.end_seconds
                if win.start_seconds <= last.end_seconds:
                    last.end_seconds = max(last.end_seconds, win.end_seconds)
                    last.best_coarse_rank = min(last.best_coarse_rank, win.best_coarse_rank)
                    last.origin_candidates.extend(win.origin_candidates)
                else:
                    merged_for_video.append(win)

        all_merged_regions.extend(merged_for_video)

    # 3. Deterministically rank merged regions across all videos
    # Priority: lowest/best coarse rank, then video_id, then region start_seconds
    all_merged_regions.sort(key=lambda r: (r.best_coarse_rank, r.video_id, r.start_seconds))

    # 4. Truncate to max_regions AFTER window merging
    return all_merged_regions[:max_regions]


def generate_local_timestamps(
    start_seconds: float,
    end_seconds: float,
    interval_seconds: float = 0.5,
) -> List[float]:
    """Generate deterministic target timestamps inside [start_seconds, end_seconds] every interval_seconds.

    Args:
        start_seconds: Start of region (inclusive).
        end_seconds: End of region (inclusive).
        interval_seconds: Sampling interval in seconds.

    Returns:
        List of non-duplicated ascending floating timestamps.
    """
    if interval_seconds <= 0 or math.isnan(interval_seconds) or math.isinf(interval_seconds):
        raise ValueError("interval_seconds must be a positive finite number")
    if start_seconds < 0:
        start_seconds = 0.0
    if end_seconds < start_seconds:
        return [round(start_seconds, 6)]

    timestamps: List[float] = []
    # Calculate step count deterministically
    span = end_seconds - start_seconds
    step_count = int(math.floor(span / interval_seconds + 1e-9))

    for k in range(step_count + 1):
        ts = round(start_seconds + k * interval_seconds, 6)
        if ts > end_seconds + 1e-9:
            break
        if not timestamps or abs(ts - timestamps[-1]) > 1e-6:
            timestamps.append(ts)

    return timestamps


def refine_coarse_candidates(
    candidates: Sequence[CoarseCandidate],
    query_embedding: np.ndarray,
    frame_provider: Callable[[str, List[float]], Sequence[Any]],
    embedding_provider: Callable[[Sequence[Any]], np.ndarray],
    config: Optional[LocalRefineConfig] = None,
    duration_by_video: Optional[Dict[str, float]] = None,
) -> List[RefinedCandidate]:
    """Execute local candidate window refinement.

    Args:
        candidates: Input coarse candidates from global search.
        query_embedding: Float32 array representing the query visual/text embedding.
        frame_provider: Callable (video_id, target_timestamps) -> frames.
        embedding_provider: Callable (frames) -> 2D numpy array of shape (N, D).
        config: Optional LocalRefineConfig instance.
        duration_by_video: Optional dictionary of video_id -> duration_seconds.

    Returns:
        List of RefinedCandidate objects (one per retained region).
    """
    if config is None:
        config = LocalRefineConfig.from_env()

    if not config.enabled or not candidates:
        return []

    # Validate query embedding
    q_vec = np.asarray(query_embedding, dtype=np.float32)
    if q_vec.ndim == 2 and q_vec.shape[0] == 1:
        q_vec = q_vec.squeeze(0)
    if q_vec.ndim != 1:
        raise ValueError(f"query_embedding must be a 1D vector or (1, D) array, got shape {query_embedding.shape}")
    if not np.isfinite(q_vec).all():
        raise ValueError("query_embedding contains non-finite values (NaN or Inf)")

    q_norm = float(np.linalg.norm(q_vec))
    if q_norm == 0.0 or math.isnan(q_norm):
        raise ValueError("query_embedding norm is zero or invalid")
    q_unit = q_vec / q_norm

    # 1. Generate merged candidate regions (max_regions limit applied after merging)
    regions = generate_candidate_regions(
        candidates=candidates,
        window_seconds=config.window_seconds,
        duration_by_video=duration_by_video,
        max_regions=config.max_regions,
    )

    refined_results: List[RefinedCandidate] = []

    # 2. Refine each region
    for region in regions:
        target_timestamps = generate_local_timestamps(
            start_seconds=region.start_seconds,
            end_seconds=region.end_seconds,
            interval_seconds=config.interval_seconds,
        )

        frames = frame_provider(region.video_id, target_timestamps)
        if not frames:
            # Skip region gracefully if frame provider returns no frames
            continue

        local_embs = embedding_provider(frames)
        if not isinstance(local_embs, np.ndarray):
            local_embs = np.asarray(local_embs, dtype=np.float32)

        if local_embs.ndim != 2:
            raise ValueError(f"local embeddings must be a 2D array, got shape {local_embs.shape}")
        if local_embs.shape[0] != len(frames):
            raise ValueError(
                f"Mismatched local frames count ({len(frames)}) and embeddings count ({local_embs.shape[0]})"
            )
        if local_embs.shape[1] != q_unit.shape[0]:
            raise ValueError(
                f"Dimension mismatch between local embeddings ({local_embs.shape[1]}) and query embedding ({q_unit.shape[0]})"
            )
        if not np.isfinite(local_embs).all():
            raise ValueError("local embeddings contain non-finite values (NaN or Inf)")

        # Normalize local embeddings
        norms = np.linalg.norm(local_embs, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise ValueError("local embeddings contain a zero-norm vector")
        norm_embs = local_embs / norms

        # Compute cosine similarity
        similarities = np.dot(norm_embs, q_unit)

        # Select best frame per region:
        # Primary: highest similarity (descending)
        # Secondary: earliest timestamp (ascending)
        # Tertiary: source_frame_index_zero_based (ascending)
        best_idx = 0
        best_key = None
        for i, f in enumerate(frames):
            sim = float(similarities[i])
            ts = float(getattr(f, "timestamp_seconds", target_timestamps[min(i, len(target_timestamps) - 1)]))
            src_idx = getattr(f, "source_frame_index_zero_based", i)
            # Tuple key for min(): (-similarity, timestamp, source_frame_index)
            key = (-sim, ts, src_idx)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = i

        best_frame = frames[best_idx]
        best_sim = float(similarities[best_idx])
        best_ts = float(getattr(best_frame, "timestamp_seconds", target_timestamps[min(best_idx, len(target_timestamps) - 1)]))
        best_src_idx = getattr(best_frame, "source_frame_index_zero_based", None)
        best_fid = getattr(best_frame, "frame_uid", None) or getattr(best_frame, "frame_id", None)
        if best_fid is None and best_src_idx is not None:
            best_fid = f"{region.video_id}:{best_src_idx:09d}"

        origin_c = (
            min(
                region.origin_candidates,
                key=lambda candidate: (
                    candidate.coarse_rank,
                    candidate.video_id,
                    candidate.timestamp_seconds,
                    candidate.source_frame_index_zero_based
                    if candidate.source_frame_index_zero_based is not None
                    else -1,
                ),
            )
            if region.origin_candidates
            else None
        )

        refined_results.append(
            RefinedCandidate(
                video_id=region.video_id,
                refined_timestamp_seconds=best_ts,
                refined_source_frame_index=best_src_idx,
                refined_frame_id=best_fid,
                local_similarity=best_sim,
                origin_candidate=origin_c,
                region=region,
            )
        )

    return refined_results
