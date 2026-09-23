
import pytest

from backend.app.vision.object_detector import (
    Detection,
    DetectorUnavailable,
    build_object_detector,
)


def test_detection_output_can_omit_or_include_bbox():
    detection = Detection("bus", 5, 0.91, (1.0, 2.0, 3.0, 4.0))
    assert detection.to_dict() == {"label": "bus", "class_id": 5, "confidence": 0.91}
    assert detection.to_dict(include_bbox=True)["bbox_xyxy"] == (1.0, 2.0, 3.0, 4.0)


def test_disabled_detector_fails_before_optional_package_import():
    with pytest.raises(DetectorUnavailable, match="disabled"):
        build_object_detector({"enabled": False, "backend": "yolo", "model_path": "/missing.pt"})


def test_missing_model_is_explicit_and_never_downloaded(tmp_path):
    with pytest.raises(DetectorUnavailable, match="missing"):
        build_object_detector({
            "enabled": True,
            "backend": "yolo",
            "model_path": str(tmp_path / "not-provisioned.pt"),
        })


def test_unknown_backend_is_explicit():
    with pytest.raises(DetectorUnavailable, match="unsupported"):
        build_object_detector({"enabled": True, "backend": "unknown", "model_path": "model.pt"})
