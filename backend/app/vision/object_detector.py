"""Provider-neutral object detection contract and lazy detector factory."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Detection:
    label: str
    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def to_dict(self, include_bbox: bool = False) -> dict:
        value = asdict(self)
        if not include_bbox:
            value.pop("bbox_xyxy")
        return value


class ObjectDetector(Protocol):
    backend: str
    model_identity: str
    device: str
    confidence: float

    def detect(self, image_or_path) -> list[Detection]: ...


class DetectorUnavailable(RuntimeError):
    """The optional detector is configured but cannot run for this request."""


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def counting_configuration() -> dict:
    backend = os.getenv("OBJECT_DETECTOR_BACKEND", "yolo").strip().lower() or "yolo"
    raw_path = os.getenv("OBJECT_DETECTOR_MODEL", "").strip()
    model_path = Path(raw_path).expanduser() if raw_path else None
    enabled = _env_bool("COUNTING_ENABLED", False)
    present = bool(model_path and model_path.is_file())
    try:
        confidence = float(os.getenv("OBJECT_DETECTOR_CONFIDENCE", "0.25"))
    except ValueError:
        confidence = 0.25
    confidence = min(1.0, max(0.0, confidence))
    try:
        top_n = int(os.getenv("COUNTING_TOP_N", "10"))
    except ValueError:
        top_n = 10
    top_n = min(30, max(1, top_n))
    return {
        "enabled": enabled,
        "configured": backend == "yolo" and model_path is not None,
        "backend": backend,
        "model_path": str(model_path) if model_path else None,
        "model_present": present,
        "device": os.getenv("OBJECT_DETECTOR_DEVICE", "auto").strip() or "auto",
        "confidence": confidence,
        "top_n": top_n,
        "max_candidates": min(30, max(1, int(os.getenv("COUNTING_MAX_CANDIDATES", "30"))))
        if os.getenv("COUNTING_MAX_CANDIDATES", "30").lstrip("+-").isdigit() else 30,
    }


def build_object_detector(configuration: dict | None = None) -> ObjectDetector:
    config = configuration or counting_configuration()
    if not config["enabled"]:
        raise DetectorUnavailable("object counting is disabled")
    if config["backend"] != "yolo":
        raise DetectorUnavailable(f"unsupported object detector backend: {config['backend']}")
    model_path = config.get("model_path")
    if not model_path:
        raise DetectorUnavailable("OBJECT_DETECTOR_MODEL is not configured")
    if not Path(model_path).is_file():
        raise DetectorUnavailable("configured object detector model file is missing")
    # Do not import Ultralytics until a count request actually needs a detector.
    from backend.app.vision.yolo_detector import YoloDetector

    return YoloDetector(
        model_path=model_path,
        device=config["device"],
        confidence=config["confidence"],
    )
