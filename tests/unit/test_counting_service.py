from pathlib import Path

import pytest

from backend.app.services.counting_service import (
    CountingService,
    resolve_candidate_image,
)
from backend.app.services.query_refiner import QueryPlan
from backend.app.vision.object_detector import Detection, DetectorUnavailable


class FakeDetector:
    backend = "yolo"
    model_identity = "fake-model-sha256"
    device = "cpu"
    confidence = 0.25

    def __init__(self, detections=None, error=None):
        self.detections = detections or []
        self.error = error
        self.calls = []

    def detect(self, image_or_path):
        self.calls.append(Path(image_or_path))
        if self.error:
            raise self.error
        return self.detections


def _plan(target="motorcycle"):
    return QueryPlan(
        task_type="qa", original_query="Có bao nhiêu xe máy ở ngã tư?",
        intent="count", requires_detector=True, detector_targets=[target],
    )


def _candidates(root, count=3):
    rows = []
    for index in range(count):
        frame = root / "VID" / "frames" / f"{index:09d}.jpg"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"fixture frame")
        rows.append({"video_id": "VID", "source_frame_index_zero_based": index, "frame_id": index})
    return rows


def test_detector_filters_target_and_confidence_and_limits_top_n(tmp_path):
    rows = _candidates(tmp_path)
    detector = FakeDetector([
        Detection("motorcycle", 3, 0.91, (1, 2, 3, 4)),
        Detection("car", 2, 0.99, (1, 2, 3, 4)),
        Detection("motorcycle", 3, 0.12, (1, 2, 3, 4)),
    ])
    service = CountingService(tmp_path, detector=detector, top_n=2, max_candidates=30)
    output, metrics = service.count(_plan(), rows)

    assert metrics["detector_invoked"] is True
    assert metrics["detector_candidates"] == 2
    assert len(detector.calls) == 2
    assert output[0]["detector_count"] == 1
    assert output[0]["answer"] == "1"
    assert len(output[0]["detector_detections"]) == 1
    assert "bbox_xyxy" not in output[0]["detector_detections"][0]
    assert output[2]["detector_status"] == "not_selected"


def test_cache_hit_skips_inference_and_corrupt_cache_recomputes(tmp_path):
    rows = _candidates(tmp_path, count=1)
    detector = FakeDetector([Detection("motorcycle", 3, 0.9, (0, 0, 1, 1))])
    service = CountingService(tmp_path, detector=detector)
    service.count(_plan(), rows)
    assert len(detector.calls) == 1

    output, metrics = CountingService(tmp_path, detector=detector).count(_plan(), rows)
    assert len(detector.calls) == 1
    assert metrics["detector_cache_hits"] == 1
    assert output[0]["detector_count"] == 1

    cache_path = next((tmp_path / "detections").rglob("*.json"))
    cache_path.write_text("not-json", encoding="utf-8")
    CountingService(tmp_path, detector=detector).count(_plan(), rows)
    assert len(detector.calls) == 2


def test_cache_fingerprint_changes_with_confidence(tmp_path):
    rows = _candidates(tmp_path, count=1)
    detector = FakeDetector([Detection("motorcycle", 3, 0.9, (0, 0, 1, 1))])
    CountingService(tmp_path, detector=detector).count(_plan(), rows)
    detector.confidence = 0.5
    CountingService(tmp_path, detector=detector).count(_plan(), rows)
    assert len(detector.calls) == 2


def test_candidate_path_traversal_is_rejected(tmp_path):
    outside = tmp_path.parent / "secret.jpg"
    outside.write_bytes(b"no")
    row = {"video_id": "VID", "image_path": "../secret.jpg"}
    with pytest.raises(ValueError, match="traversal"):
        resolve_candidate_image(tmp_path, row)

    row = {"video_id": "VID", "image_path": str(outside)}
    with pytest.raises(ValueError, match="escapes"):
        resolve_candidate_image(tmp_path, row)


@pytest.mark.parametrize("bad_row", [
    {"video_id": "../VID", "source_frame_index_zero_based": 0},
    {"video_id": "VID", "image_path": "frames/file.gif"},
])
def test_invalid_candidate_metadata_is_rejected(tmp_path, bad_row):
    with pytest.raises(ValueError):
        resolve_candidate_image(tmp_path, bad_row)


def test_detector_failure_is_per_candidate_and_does_not_raise(tmp_path):
    rows = _candidates(tmp_path, count=1)
    detector = FakeDetector(error=RuntimeError("model error"))
    output, metrics = CountingService(tmp_path, detector=detector).count(_plan(), rows)
    assert output[0]["detector_status"] == "error"
    assert metrics["detector_status"] == "unavailable"
    assert not (tmp_path / "CURRENT").exists()


def test_unavailable_detector_is_reported_without_fake_count(tmp_path):
    rows = _candidates(tmp_path, count=1)
    def unavailable(_):
        raise DetectorUnavailable("missing model")
    output, metrics = CountingService(tmp_path, detector_factory=unavailable).count(_plan(), rows)
    assert output[0]["detector_status"] == "unavailable"
    assert "detector_count" not in output[0]
    assert metrics["detector_invoked"] is False


def test_unknown_target_is_explicit_and_never_counts_other_labels(tmp_path):
    rows = _candidates(tmp_path, count=1)
    detector = FakeDetector([Detection("cat", 15, 0.9, (0, 0, 1, 1))])
    plan = _plan("unicorn")
    output, _ = CountingService(tmp_path, detector=detector).count(plan, rows)
    assert output[0]["detector_status"] == "unsupported_target"
    assert "detector_count" not in output[0]
    assert detector.calls == []


def test_temporal_count_reports_tracking_required(tmp_path):
    rows = _candidates(tmp_path, count=1)
    detector = FakeDetector()
    plan = QueryPlan(task_type="qa", original_query="vehicles passed", intent="temporal_count",
                     requires_tracking=True, detector_targets=["motorcycle"])
    output, metrics = CountingService(tmp_path, detector=detector).count(plan, rows)
    assert output[0]["tracking"] == {"supported": False, "status": "tracking_required"}
    assert metrics["tracking"]["supported"] is False
    assert detector.calls == []
