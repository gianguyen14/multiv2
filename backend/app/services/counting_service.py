"""Bounded candidate-frame object counting with an optional JSON cache."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from backend.app.services.query_refiner import QueryPlan
from backend.app.video.atomic_io import write_json_atomic
from backend.app.vision.object_detector import (
    Detection,
    build_object_detector,
    counting_configuration,
)

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1
SUPPORTED_FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

_COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)
TARGET_LABELS = {label: {label} for label in _COCO_LABELS}


def resolve_candidate_image(processed_root: str | Path, candidate: dict[str, Any]) -> Path:
    """Resolve an indexed frame only from its owning video directory."""
    root = Path(processed_root).expanduser().resolve()
    video_id = str(candidate.get("video_id") or "")
    if not VIDEO_ID_PATTERN.fullmatch(video_id) or video_id in {".", ".."}:
        raise ValueError("invalid candidate video_id")
    source_index = candidate.get("source_frame_index_zero_based")
    if source_index is not None and (
        isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0
    ):
        raise ValueError("candidate has no valid source frame index")
    expected_uid = f"{video_id}:{source_index:09d}" if source_index is not None else None
    if candidate.get("frame_uid") and candidate["frame_uid"] != expected_uid:
        raise ValueError("candidate frame UID does not match its video and source index")
    video_root = (root / video_id).resolve()
    if not video_root.is_relative_to(root):
        raise ValueError("candidate video directory escapes processed root")

    raw = candidate.get("image_path")
    if raw:
        supplied = Path(str(raw))
        if ".." in supplied.parts:
            raise ValueError("candidate frame path contains traversal")
        if supplied.is_absolute():
            path = supplied.resolve()
        elif supplied.parts and supplied.parts[0] == video_id:
            path = (root / supplied).resolve()
        else:
            path = (video_root / supplied).resolve()
        if not path.is_relative_to(video_root):
            raise ValueError("candidate frame path escapes video directory")
        if path.suffix.lower() not in SUPPORTED_FRAME_EXTENSIONS:
            raise ValueError("candidate frame extension is unsupported")
        if source_index is not None and path.stem != f"{source_index:09d}":
            raise ValueError("candidate frame path does not match its source index")
        if not path.is_file():
            raise FileNotFoundError("candidate frame image is missing")
        return path

    if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
        raise ValueError("candidate has no valid source frame index")
    stem = f"{source_index:09d}"
    for extension in (".jpg", ".webp", ".png", ".jpeg"):
        path = video_root / "frames" / f"{stem}{extension}"
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("candidate frame image is missing")


class CountingService:
    def __init__(
        self,
        processed_root: str | Path,
        *,
        detector=None,
        detector_factory=build_object_detector,
        top_n: int | None = None,
        max_candidates: int | None = None,
        include_boxes: bool | None = None,
        cache_root: str | Path | None = None,
    ):
        self.processed_root = Path(processed_root).expanduser().resolve()
        configured_cache = cache_root or os.getenv("COUNTING_CACHE_ROOT", "").strip()
        self.cache_root = (
            Path(configured_cache).expanduser().resolve()
            if configured_cache else self.processed_root / "detections"
        )
        self.configuration = counting_configuration()
        self.detector = detector
        self.detector_factory = detector_factory
        self.top_n = min(30, max(1, int(top_n if top_n is not None else self.configuration["top_n"])))
        self.max_candidates = min(
            30,
            max(1, int(max_candidates if max_candidates is not None else self.configuration["max_candidates"])),
        )
        self.include_boxes = (
            os.getenv("COUNTING_INCLUDE_BOXES", "false").strip().lower() in {"1", "true", "yes"}
            if include_boxes is None else bool(include_boxes)
        )

    def status(self) -> dict[str, Any]:
        config = dict(self.configuration)
        model_path = config.get("model_path")
        config["model_path"] = Path(model_path).name if model_path else None
        config.pop("confidence", None)
        config["confidence_threshold"] = self.configuration["confidence"]
        return config

    @staticmethod
    def _fingerprint(detector: Any) -> str:
        data = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "backend": str(getattr(detector, "backend", "unknown")),
            "model_identity": str(getattr(detector, "model_identity", "unknown")),
            "confidence": float(getattr(detector, "confidence", 0.25)),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]

    def _cache_file(self, detector: Any, video_id: str, frame_index: int) -> Path:
        fingerprint = self._fingerprint(detector)
        return self.cache_root / fingerprint / video_id / f"{frame_index:09d}.json"

    @staticmethod
    def _decode_cache(path: Path) -> list[Detection] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != CACHE_SCHEMA_VERSION or not isinstance(payload.get("detections"), list):
                return None
            values = []
            for item in payload["detections"]:
                bbox = item["bbox_xyxy"]
                if (not isinstance(bbox, list) or len(bbox) != 4 or not isinstance(item["label"], str)
                        or not 0.0 <= float(item["confidence"]) <= 1.0):
                    return None
                values.append(Detection(
                    str(item["label"]), int(item["class_id"]), float(item["confidence"]),
                    tuple(float(v) for v in bbox),
                ))
            return values
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def _detections(self, detector: Any, candidate: dict[str, Any], image_path: Path):
        video_id = str(candidate["video_id"])
        frame_index = int(candidate["source_frame_index_zero_based"])
        cache_path = self._cache_file(detector, video_id, frame_index)
        if cache_path.is_file():
            cached = self._decode_cache(cache_path)
            if cached is not None:
                return cached, True
        detections = detector.detect(image_path)
        threshold = float(getattr(detector, "confidence", self.configuration["confidence"]))
        filtered = [d for d in detections if float(d.confidence) >= threshold]
        try:
            write_json_atomic(cache_path, {
                "schema_version": CACHE_SCHEMA_VERSION,
                "detections": [d.to_dict(include_bbox=True) for d in filtered],
            })
        except OSError as exc:
            logger.debug("could not write detection cache: %s", type(exc).__name__)
        return filtered, False

    def count(self, plan: QueryPlan, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows = [dict(row) for row in candidates]
        metrics: dict[str, Any] = {
            "intent": plan.intent,
            "detector_invoked": False,
            "detector_backend": None,
            "detector_candidates": 0,
            "detector_cache_hits": 0,
            "detector_ms": 0.0,
        }

        if plan.intent == "temporal_count" or plan.requires_tracking:
            capability = {"supported": False, "status": "tracking_required"}
            for row in rows:
                row.update({"detector_status": "tracking_required", "tracking": capability})
            metrics["tracking"] = capability
            return rows, metrics

        targets = list(dict.fromkeys(plan.detector_targets))
        if not plan.requires_detector or not targets or any(target not in TARGET_LABELS for target in targets):
            status = "unsupported_target" if plan.intent == "count" else "not_requested"
            for row in rows:
                row["detector_status"] = status
            metrics["detector_status"] = status
            return rows, metrics

        try:
            detector = self.detector or self.detector_factory(self.configuration)
            self.detector = detector
        except Exception as exc:  # noqa: BLE001 - optional detector errors must not fail search
            logger.info("object detector unavailable (%s)", type(exc).__name__)
            for row in rows[: self.top_n]:
                row["detector_status"] = "unavailable"
            metrics["detector_status"] = "unavailable"
            metrics["detector_error"] = type(exc).__name__
            return rows, metrics

        metrics["detector_backend"] = str(getattr(detector, "backend", "unknown"))
        max_rows = min(self.top_n, self.max_candidates, len(rows))
        metrics["detector_candidates"] = max_rows
        metrics["detector_invoked"] = bool(max_rows)
        target = targets[0]
        labels = TARGET_LABELS[target]
        start = time.perf_counter()
        first_answer_set = False
        for index, row in enumerate(rows):
            if index >= max_rows:
                row["detector_status"] = "not_selected"
                continue
            try:
                image_path = resolve_candidate_image(self.processed_root, row)
                detections, cache_hit = self._detections(detector, row, image_path)
                if cache_hit:
                    metrics["detector_cache_hits"] += 1
                selected = [d for d in detections if d.label.casefold() in labels]
                row.update({
                    "intent": "count",
                    "detector_target": target,
                    "detector_count": len(selected),
                    "detector_detections": [d.to_dict(self.include_boxes) for d in selected],
                    "detector_model": str(getattr(detector, "model_identity", "unknown")),
                    "detector_status": "ok",
                })
                if not first_answer_set:
                    row["answer"] = str(len(selected))
                    first_answer_set = True
            except FileNotFoundError:
                row["detector_status"] = "missing_frame"
            except (ValueError, OSError) as exc:
                row["detector_status"] = "invalid_frame_path"
                row["detector_error"] = type(exc).__name__
            except Exception as exc:  # noqa: BLE001 - isolate failures to this candidate
                logger.info("count candidate failed (%s)", type(exc).__name__)
                row["detector_status"] = "error"
                row["detector_error"] = type(exc).__name__
        metrics["detector_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
        metrics["detector_status"] = "ok" if first_answer_set else "unavailable"
        return rows, metrics
