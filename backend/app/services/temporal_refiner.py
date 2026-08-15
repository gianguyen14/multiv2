"""Temporal Deep Refinement for TRAKE.

Implements coarse-to-fine temporal refinement for TRAKE multi-event queries.
Uses the existing sparse FAISS index for coarse localization, then performs
bounded sequential PyAV dense decoding around candidate temporal regions,
computes dense frame embeddings and temporal context scores, and passes the
refined candidate pool to the existing TRAKEAligner for monotonic dynamic programming.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from backend.app.core.config import (
    TRAKE_TEMPORAL_REFINE_CACHE_ENABLED,
    TRAKE_TEMPORAL_REFINE_ENABLED,
    TRAKE_TEMPORAL_REFINE_MAX_FRAMES_PER_REGION,
    TRAKE_TEMPORAL_REFINE_MAX_REGIONS_PER_VIDEO,
    TRAKE_TEMPORAL_REFINE_MAX_TOTAL_REGIONS,
    TRAKE_TEMPORAL_REFINE_SAMPLE_FPS,
    TRAKE_TEMPORAL_REFINE_WINDOW_SECONDS,
)
from backend.app.retrieval.trake import EventCandidate
from backend.app.video.atomic_io import write_json_atomic, write_numpy_atomic

logger = logging.getLogger(__name__)


@dataclass
class TemporalRefinerConfig:
    enabled: bool = TRAKE_TEMPORAL_REFINE_ENABLED
    window_seconds: float = TRAKE_TEMPORAL_REFINE_WINDOW_SECONDS
    sample_fps: float = TRAKE_TEMPORAL_REFINE_SAMPLE_FPS
    max_regions_per_video: int = TRAKE_TEMPORAL_REFINE_MAX_REGIONS_PER_VIDEO
    max_total_regions: int = TRAKE_TEMPORAL_REFINE_MAX_TOTAL_REGIONS
    max_frames_per_region: int = TRAKE_TEMPORAL_REFINE_MAX_FRAMES_PER_REGION
    cache_enabled: bool = TRAKE_TEMPORAL_REFINE_CACHE_ENABLED
    local_pooling_window: int = 1
    # Engineering defaults (not benchmark tuned against GT)
    weight_visual: float = 0.8
    weight_temporal: float = 0.2

    @classmethod
    def from_env(cls) -> TemporalRefinerConfig:
        def _bool(val: Optional[str], default: bool) -> bool:
            return val.lower() in ("1", "true", "yes") if val is not None else default

        return cls(
            enabled=_bool(os.getenv("TRAKE_TEMPORAL_REFINE_ENABLED"), TRAKE_TEMPORAL_REFINE_ENABLED),
            window_seconds=float(os.getenv("TRAKE_TEMPORAL_REFINE_WINDOW_SECONDS", str(TRAKE_TEMPORAL_REFINE_WINDOW_SECONDS))),
            sample_fps=float(os.getenv("TRAKE_TEMPORAL_REFINE_SAMPLE_FPS", str(TRAKE_TEMPORAL_REFINE_SAMPLE_FPS))),
            max_regions_per_video=int(os.getenv("TRAKE_TEMPORAL_REFINE_MAX_REGIONS_PER_VIDEO", str(TRAKE_TEMPORAL_REFINE_MAX_REGIONS_PER_VIDEO))),
            max_total_regions=int(os.getenv("TRAKE_TEMPORAL_REFINE_MAX_TOTAL_REGIONS", str(TRAKE_TEMPORAL_REFINE_MAX_TOTAL_REGIONS))),
            max_frames_per_region=int(os.getenv("TRAKE_TEMPORAL_REFINE_MAX_FRAMES_PER_REGION", str(TRAKE_TEMPORAL_REFINE_MAX_FRAMES_PER_REGION))),
            cache_enabled=_bool(os.getenv("TRAKE_TEMPORAL_REFINE_CACHE_ENABLED"), TRAKE_TEMPORAL_REFINE_CACHE_ENABLED),
        )


@dataclass(frozen=True)
class TemporalRegion:
    video_id: str
    start_frame: int
    end_frame: int
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    source_candidate_frames: tuple[int, ...] = ()
    max_candidate_score: float = 0.0

    @property
    def region_key(self) -> str:
        return f"r_{self.start_frame:09d}_{self.end_frame:09d}"


class TemporalRefineCache:
    """Lazy local cache for densely decoded and embedded temporal regions."""

    def __init__(self, cache_root: Path):
        self.cache_root = Path(cache_root)

    def _region_dir(self, video_id: str, region_key: str) -> Path:
        return self.cache_root / video_id / region_key

    def get(
        self, video_id: str, region_key: str, expected_fingerprint: str
    ) -> Optional[Tuple[np.ndarray, List[Dict[str, Any]]]]:
        region_dir = self._region_dir(video_id, region_key)
        if not region_dir.is_dir():
            return None
        meta_path = region_dir / "metadata.json"
        emb_path = region_dir / "embeddings.npy"
        records_path = region_dir / "frame_records.json"
        if not (meta_path.is_file() and emb_path.is_file() and records_path.is_file()):
            return None
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("fingerprint") != expected_fingerprint:
                return None
            records = json.loads(records_path.read_text())
            embeddings = np.load(emb_path, allow_pickle=False)
            if embeddings.dtype != np.float32 or embeddings.ndim != 2:
                return None
            if embeddings.shape[0] != len(records) or not np.isfinite(embeddings).all():
                return None
            return embeddings, records
        except Exception as exc:
            logger.debug("Refinement cache entry invalid or corrupt: %s", exc)
            return None

    def put(
        self,
        video_id: str,
        region_key: str,
        metadata: Dict[str, Any],
        embeddings: np.ndarray,
        records: List[Dict[str, Any]],
    ) -> None:
        region_dir = self._region_dir(video_id, region_key)
        staging_dir = region_dir.parent / f".staging_{region_key}_{uuid.uuid4().hex}"
        try:
            staging_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(staging_dir / "metadata.json", metadata)
            write_numpy_atomic(staging_dir / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
            write_json_atomic(staging_dir / "frame_records.json", records)
            region_dir.parent.mkdir(parents=True, exist_ok=True)
            if region_dir.exists():
                shutil.rmtree(region_dir, ignore_errors=True)
            os.replace(staging_dir, region_dir)
        except Exception as exc:
            logger.warning("Failed to write refinement cache for %s/%s: %s", video_id, region_key, exc)
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)


class TemporalRefiner:
    """Service that performs bounded dense temporal refinement for TRAKE candidates."""

    def __init__(
        self,
        processed_root: Optional[Path | str] = None,
        config: Optional[TemporalRefinerConfig] = None,
        video_paths: Optional[Dict[str, Path | str]] = None,
        encoder: Optional[Any] = None,
    ):
        self.processed_root = Path(processed_root) if processed_root else None
        self.config = config or TemporalRefinerConfig.from_env()
        self.video_paths = {k: Path(v) for k, v in (video_paths or {}).items()}
        self.encoder = encoder
        self.cache = (
            TemporalRefineCache(self.processed_root / "temporal_refine_cache")
            if (self.processed_root and self.config.cache_enabled)
            else None
        )

    def resolve_video_path(self, video_id: str) -> Optional[Path]:
        """Locates the source video file for video_id."""
        if video_id in self.video_paths and self.video_paths[video_id].is_file():
            return self.video_paths[video_id]

        if self.processed_root:
            meta_path = self.processed_root / video_id / "metadata.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                    src = meta.get("source_path")
                    if src and Path(src).is_file():
                        return Path(src)
                    filename = meta.get("filename", f"{video_id}.mp4")
                    parent = self.processed_root.parent
                    candidates = [
                        parent / "test-videos" / filename,
                        parent / "test-videos-m27" / filename,
                        parent / "raw" / "videos" / filename,
                        parent / "validation-3videos" / filename,
                    ]
                    for cand in candidates:
                        if cand.is_file():
                            return cand
                except Exception:
                    pass

        raw_root = os.getenv("VIDEO_RAW_ROOT")
        if raw_root:
            for ext in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
                cand = Path(raw_root) / f"{video_id}{ext}"
                if cand.is_file():
                    return cand

        for base in ("data/test-videos", "data/test-videos-m27", "data/raw/videos", "data/validation-3videos"):
            for ext in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
                cand = Path(base) / f"{video_id}{ext}"
                if cand.is_file():
                    return cand.resolve()

        return None

    def get_video_fps_and_count(self, video_id: str) -> Tuple[float, int]:
        """Returns (fps, total_frame_count) from metadata or defaults."""
        if self.processed_root:
            meta_path = self.processed_root / video_id / "metadata.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                    rate_str = meta.get("real_frame_rate") or meta.get("avg_frame_rate")
                    fps = float(Fraction(rate_str)) if rate_str and rate_str != "0/0" else 30.0
                    count = meta.get("decoded_frame_count") or meta.get("reported_frame_count") or 1000000
                    return max(1.0, fps), int(count)
                except Exception:
                    pass
        return 30.0, 1000000

    def compute_cache_fingerprint(
        self,
        video_id: str,
        video_path: Optional[Path],
        region: TemporalRegion,
        sample_step: int,
        encoder_identity: Dict[str, Any],
    ) -> str:
        """Computes deterministic cache key fingerprint for invalidation check."""
        h = hashlib.sha256()
        h.update(video_id.encode("utf-8"))
        if video_path and video_path.is_file():
            try:
                stat = video_path.stat()
                h.update(f"{stat.st_size}_{stat.st_mtime_ns}".encode("utf-8"))
            except Exception:
                pass
        h.update(f"{region.start_frame}_{region.end_frame}_{sample_step}_{self.config.sample_fps}".encode("utf-8"))
        h.update(json.dumps(encoder_identity, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    def build_candidate_regions(
        self,
        candidates_by_event: List[List[EventCandidate]],
    ) -> Dict[str, List[TemporalRegion]]:
        """Groups coarse candidates into limited, merged temporal regions per video."""
        candidates_by_video: Dict[str, List[EventCandidate]] = {}
        for event_candidates in candidates_by_event:
            for cand in event_candidates:
                candidates_by_video.setdefault(cand.video_id, []).append(cand)

        video_regions: Dict[str, List[TemporalRegion]] = {}
        total_regions_count = 0

        # Sort videos by maximum coarse candidate score to prioritize top candidate videos
        sorted_videos = sorted(
            candidates_by_video.keys(),
            key=lambda v: max((c.score for c in candidates_by_video[v]), default=0.0),
            reverse=True,
        )

        for video_id in sorted_videos:
            cands = candidates_by_video[video_id]
            fps, total_frames = self.get_video_fps_and_count(video_id)
            delta_frames = max(1, int(round(self.config.window_seconds * fps)))

            # Build raw windows for each candidate frame
            raw_intervals: List[Tuple[int, int, int, float]] = []
            for cand in cands:
                start_f = max(0, cand.frame_id - delta_frames)
                end_f = min(total_frames - 1, cand.frame_id + delta_frames)
                raw_intervals.append((start_f, end_f, cand.frame_id, cand.score))

            # Sort by start frame
            raw_intervals.sort(key=lambda item: item[0])

            # Merge overlapping intervals
            merged: List[Dict[str, Any]] = []
            for start_f, end_f, fid, score in raw_intervals:
                if not merged or start_f > merged[-1]["end_frame"]:
                    merged.append({
                        "start_frame": start_f,
                        "end_frame": end_f,
                        "candidates": [fid],
                        "max_score": score,
                    })
                else:
                    # Overlap: extend existing window
                    merged[-1]["end_frame"] = max(merged[-1]["end_frame"], end_f)
                    merged[-1]["candidates"].append(fid)
                    merged[-1]["max_score"] = max(merged[-1]["max_score"], score)

            # Limit per video
            if len(merged) > self.config.max_regions_per_video:
                merged.sort(key=lambda item: -item["max_score"])
                merged = merged[: self.config.max_regions_per_video]
                merged.sort(key=lambda item: item["start_frame"])

            # Convert to TemporalRegion instances
            reg_list: List[TemporalRegion] = []
            for item in merged:
                if total_regions_count >= self.config.max_total_regions:
                    break
                reg_list.append(
                    TemporalRegion(
                        video_id=video_id,
                        start_frame=item["start_frame"],
                        end_frame=item["end_frame"],
                        start_seconds=item["start_frame"] / fps,
                        end_seconds=item["end_frame"] / fps,
                        source_candidate_frames=tuple(sorted(set(item["candidates"]))),
                        max_candidate_score=item["max_score"],
                    )
                )
                total_regions_count += 1

            if reg_list:
                video_regions[video_id] = reg_list

            if total_regions_count >= self.config.max_total_regions:
                break

        return video_regions

    def _decode_video_regions(
        self,
        video_id: str,
        regions: List[TemporalRegion],
        encoder: Any,
        timings: Dict[str, float],
        counters: Dict[str, int],
    ) -> Dict[str, Tuple[np.ndarray, List[Dict[str, Any]]]]:
        """Densely decodes and embeds regions for a single video.

        Ensures authoritative frame IDs are strictly the sequential PyAV display ordinals.
        """
        import av

        results: Dict[str, Tuple[np.ndarray, List[Dict[str, Any]]]] = {}
        regions_to_decode: List[Tuple[TemporalRegion, int, str]] = []

        fps, _ = self.get_video_fps_and_count(video_id)
        video_path = self.resolve_video_path(video_id)

        encoder_identity = (
            encoder.identity()
            if hasattr(encoder, "identity")
            else {"model_name": getattr(encoder, "model_name", "siglip2"), "embedding_dim": getattr(encoder, "embedding_dim", 768)}
        )

        for region in regions:
            # Determine step
            step = max(1, int(round(fps / self.config.sample_fps)))
            num_frames = (region.end_frame - region.start_frame) // step + 1
            if num_frames > self.config.max_frames_per_region:
                step = max(1, math.ceil((region.end_frame - region.start_frame) / self.config.max_frames_per_region))

            fingerprint = self.compute_cache_fingerprint(
                video_id, video_path, region, step, encoder_identity
            )

            # Check cache
            if self.cache:
                cached = self.cache.get(video_id, region.region_key, fingerprint)
                if cached is not None:
                    counters["cache_hits"] += 1
                    results[region.region_key] = cached
                    continue

            counters["cache_misses"] += 1
            regions_to_decode.append((region, step, fingerprint))

        if not regions_to_decode:
            return results

        if video_path is None or not video_path.is_file():
            raise FileNotFoundError(f"Source video for {video_id} not found")

        # Map target ordinals to region indices
        target_ordinals: Set[int] = set()
        region_targets: Dict[str, Set[int]] = {}
        for region, step, _ in regions_to_decode:
            targets = set(range(region.start_frame, region.end_frame + 1, step))
            target_ordinals.update(targets)
            region_targets[region.region_key] = targets

        max_target = max(target_ordinals) if target_ordinals else 0
        decoded_frames: Dict[int, Dict[str, Any]] = {}

        t_decode_start = time.perf_counter()
        try:
            with av.open(str(video_path)) as container:
                stream = next((s for s in container.streams if s.type == "video"), None)
                if stream is None:
                    raise RuntimeError(f"No video stream found in {video_path}")

                # Strict PyAV sequential decode invariant
                for ordinal, frame in enumerate(container.decode(stream)):
                    if ordinal in target_ordinals:
                        # Extract PIL image only for target frames
                        time_base = frame.time_base or stream.time_base
                        ts = float(Fraction(frame.pts) * Fraction(time_base)) if frame.pts is not None and time_base else None
                        decoded_frames[ordinal] = {
                            "frame_id": ordinal,
                            "pts": frame.pts,
                            "timestamp_seconds": ts,
                            "image": frame.to_image().convert("RGB"),
                        }
                    if ordinal >= max_target:
                        break
        except Exception as exc:
            raise RuntimeError(f"Failed to decode frames for {video_id}: {exc}") from exc

        timings["dense_decode_ms"] += (time.perf_counter() - t_decode_start) * 1000.0
        counters["dense_frames_decoded"] += len(decoded_frames)

        # Process embeddings region by region
        t_embed_start = time.perf_counter()
        for region, step, fingerprint in regions_to_decode:
            reg_key = region.region_key
            reg_targets = sorted(region_targets[reg_key])
            reg_records: List[Dict[str, Any]] = []
            reg_images: List[Image.Image] = []

            for ord_id in reg_targets:
                if ord_id in decoded_frames:
                    item = decoded_frames[ord_id]
                    reg_images.append(item["image"])
                    reg_records.append({
                        "frame_id": item["frame_id"],
                        "source_frame_index_zero_based": item["frame_id"],
                        "pts": item["pts"],
                        "timestamp_seconds": item["timestamp_seconds"],
                        "video_id": video_id,
                    })

            if not reg_images:
                continue

            # Batch encode
            embeddings = encoder.encode_image(reg_images, batch_size=min(16, len(reg_images)))
            counters["dense_frames_embedded"] += len(reg_images)

            metadata = {
                "schema_version": 1,
                "video_id": video_id,
                "region_key": reg_key,
                "start_frame": region.start_frame,
                "end_frame": region.end_frame,
                "sample_step": step,
                "sample_fps": self.config.sample_fps,
                "frame_count": len(reg_records),
                "encoder_identity": encoder_identity,
                "fingerprint": fingerprint,
            }

            if self.cache:
                self.cache.put(video_id, reg_key, metadata, embeddings, reg_records)

            results[reg_key] = (embeddings, reg_records)

        timings["dense_embedding_ms"] += (time.perf_counter() - t_embed_start) * 1000.0

        # Close all PIL images immediately to free RAM
        for item in decoded_frames.values():
            if "image" in item and hasattr(item["image"], "close"):
                item["image"].close()

        return results

    def _score_event_candidates(
        self,
        event_query: str,
        text_embedding: np.ndarray,
        region_results: Dict[str, Tuple[np.ndarray, List[Dict[str, Any]]]],
        video_id: str,
    ) -> List[EventCandidate]:
        """Computes dense visual similarities with local temporal context for an event."""
        candidates: List[EventCandidate] = []
        w_vis = self.config.weight_visual
        w_temp = self.config.weight_temporal
        pool_win = self.config.local_pooling_window

        for region_key, (embeddings, records) in region_results.items():
            if len(records) == 0:
                continue

            # Cosine similarity: (1, dim) @ (N, dim).T -> (N,)
            raw_sims = (text_embedding @ embeddings.T).flatten()

            N = len(raw_sims)
            smoothed_scores = np.zeros(N, dtype=np.float32)

            for j in range(N):
                # Local pooling window [j - pool_win, j + pool_win]
                start_idx = max(0, j - pool_win)
                end_idx = min(N, j + pool_win + 1)
                local_mean = float(np.mean(raw_sims[start_idx:end_idx]))
                smoothed_scores[j] = float(w_vis * raw_sims[j] + w_temp * local_mean)

            for j, rec in enumerate(records):
                candidates.append(
                    EventCandidate(
                        video_id=video_id,
                        frame_id=rec["frame_id"],
                        score=float(smoothed_scores[j]),
                    )
                )

        return candidates

    def refine_trake_candidates(
        self,
        events: List[str],
        coarse_candidates_by_event: List[List[EventCandidate]],
        encoder: Optional[Any] = None,
    ) -> Tuple[List[List[EventCandidate]], Dict[str, Any]]:
        """Refines coarse TRAKE event candidates using dense temporal decoding.

        Returns:
            (refined_candidates_by_event, metrics_dict)
        """
        t_total_start = time.perf_counter()
        metrics: Dict[str, Any] = {
            "refinement_used": False,
            "regions_considered": 0,
            "regions_refined": 0,
            "dense_frames_decoded": 0,
            "dense_frames_embedded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "coarse_retrieval_ms": 0.0,
            "region_build_ms": 0.0,
            "dense_decode_ms": 0.0,
            "dense_embedding_ms": 0.0,
            "temporal_scoring_ms": 0.0,
            "trake_alignment_ms": 0.0,
            "total_ms": 0.0,
        }

        if not self.config.enabled:
            metrics["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
            return coarse_candidates_by_event, metrics

        if not coarse_candidates_by_event or any(not c for c in coarse_candidates_by_event):
            metrics["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
            return coarse_candidates_by_event, metrics

        enc = encoder or self.encoder
        if enc is None:
            logger.debug("No encoder provided for temporal refiner; skipping refinement")
            metrics["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
            return coarse_candidates_by_event, metrics

        try:
            # 1. Build merged candidate regions
            t_reg_start = time.perf_counter()
            video_regions = self.build_candidate_regions(coarse_candidates_by_event)
            metrics["region_build_ms"] = (time.perf_counter() - t_reg_start) * 1000.0

            total_regions = sum(len(regs) for regs in video_regions.values())
            metrics["regions_considered"] = total_regions

            if not video_regions:
                metrics["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
                return coarse_candidates_by_event, metrics

            # 2. Encode event text queries
            event_embeddings = enc.encode_text(events)

            # 3. Dense decode & embed per video
            timings = {"dense_decode_ms": 0.0, "dense_embedding_ms": 0.0}
            counters = {"dense_frames_decoded": 0, "dense_frames_embedded": 0, "cache_hits": 0, "cache_misses": 0}

            video_dense_results: Dict[str, Dict[str, Tuple[np.ndarray, List[Dict[str, Any]]]]] = {}

            for video_id, regions in video_regions.items():
                try:
                    res = self._decode_video_regions(video_id, regions, enc, timings, counters)
                    if res:
                        video_dense_results[video_id] = res
                        metrics["regions_refined"] += len(res)
                except Exception as exc:
                    logger.warning("Failed dense decode for %s (skipping video refinement): %s", video_id, exc)

            metrics["dense_decode_ms"] = timings["dense_decode_ms"]
            metrics["dense_embedding_ms"] = timings["dense_embedding_ms"]
            metrics["dense_frames_decoded"] = counters["dense_frames_decoded"]
            metrics["dense_frames_embedded"] = counters["dense_frames_embedded"]
            metrics["cache_hits"] = counters["cache_hits"]
            metrics["cache_misses"] = counters["cache_misses"]

            # 4. Dense scoring & candidate pool expansion
            t_score_start = time.perf_counter()
            refined_candidates_by_event: List[List[EventCandidate]] = []

            for e_idx, event_query in enumerate(events):
                event_emb = event_embeddings[e_idx : e_idx + 1]
                coarse_list = coarse_candidates_by_event[e_idx] if e_idx < len(coarse_candidates_by_event) else []

                dense_candidates: List[EventCandidate] = []
                for video_id, region_dict in video_dense_results.items():
                    dense_cands = self._score_event_candidates(event_query, event_emb, region_dict, video_id)
                    dense_candidates.extend(dense_cands)

                # Merge dense candidates with coarse candidates, deduplicating by (video_id, frame_id)
                cand_map: Dict[Tuple[str, int], float] = {}
                for cand in coarse_list:
                    cand_map[(cand.video_id, cand.frame_id)] = cand.score
                for cand in dense_candidates:
                    key = (cand.video_id, cand.frame_id)
                    cand_map[key] = max(cand_map.get(key, float("-inf")), cand.score)

                merged_list = [
                    EventCandidate(video_id=vid, frame_id=fid, score=score)
                    for (vid, fid), score in cand_map.items()
                ]
                # Sort by score desc, frame_id asc
                merged_list.sort(key=lambda item: (-item.score, item.frame_id))
                refined_candidates_by_event.append(merged_list)

            metrics["temporal_scoring_ms"] = (time.perf_counter() - t_score_start) * 1000.0
            metrics["refinement_used"] = metrics["regions_refined"] > 0
            metrics["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0

            return refined_candidates_by_event, metrics

        except Exception as exc:
            logger.warning("Temporal refinement failed (falling back to coarse candidates): %s", exc)
            metrics["error"] = str(exc)
            metrics["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
            return coarse_candidates_by_event, metrics
