"""Lazy Ultralytics YOLO adapter. Weights must be provisioned explicitly."""

from __future__ import annotations

import hashlib
from pathlib import Path

from backend.app.vision.object_detector import Detection, DetectorUnavailable


class YoloDetector:
    backend = "yolo"

    def __init__(self, *, model_path, device: str = "auto", confidence: float = 0.25):
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise DetectorUnavailable("configured object detector model file is missing")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailable(
                "Ultralytics is not installed; install the optional YOLO dependencies"
            ) from exc
        try:
            import torch
        except ImportError as exc:
            raise DetectorUnavailable("PyTorch is required by the YOLO detector") from exc

        self.model_path = path
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.confidence = min(1.0, max(0.0, float(confidence)))
        self.model_identity = self._sha256(path)
        try:
            self._model = YOLO(str(path))
        except Exception as exc:
            raise DetectorUnavailable(f"failed to load configured YOLO model ({type(exc).__name__})") from exc

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def detect(self, image_or_path) -> list[Detection]:
        try:
            results = self._model.predict(
                source=str(image_or_path),
                device=self.device,
                conf=self.confidence,
                verbose=False,
            )
            names = getattr(self._model, "names", {})
            detections: list[Detection] = []
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                class_ids = boxes.cls.detach().cpu().tolist()
                confidences = boxes.conf.detach().cpu().tolist()
                coordinates = boxes.xyxy.detach().cpu().tolist()
                for class_id, confidence, bbox in zip(class_ids, confidences, coordinates):
                    class_id = int(class_id)
                    label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                    detections.append(Detection(
                        label=str(label).strip().lower(),
                        class_id=class_id,
                        confidence=float(confidence),
                        bbox_xyxy=tuple(float(value) for value in bbox[:4]),
                    ))
            return detections
        except Exception as exc:
            raise RuntimeError(f"YOLO inference failed ({type(exc).__name__})") from exc
