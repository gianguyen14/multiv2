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


# ── Items 2 & 3: multi-target and vehicle group ───────────────────────────────

def test_multi_target_unsupported_returns_explicit_status(tmp_path):
    """Incompatible multi-target (cat + dog share no semantic group) -> multi_target_not_supported."""
    from backend.app.services.counting_service import CountingService
    from backend.app.services.query_refiner import QueryPlan
    from backend.app.vision.object_detector import Detection

    plan = QueryPlan(
        task_type="qa", original_query="Có bao nhiêu mèo và chó?",
        intent="count", requires_detector=True, detector_targets=["cat", "dog"],
    )
    root = tmp_path
    rows = [{"video_id": "VID", "source_frame_index_zero_based": 0, "frame_id": 0}]
    frame = root / "VID" / "frames" / "000000000.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"fixture")

    class _Det:
        backend = "yolo"; model_identity = "sha256:test"; device = "cpu"; confidence = 0.25
        def detect(self, p): return [Detection("cat", 15, 0.9, (0, 0, 1, 1))]

    service = CountingService(tmp_path, detector=_Det())
    output, metrics = service.count(plan, rows)
    assert metrics["detector_status"] == "multi_target_not_supported"
    assert output[0]["detector_status"] == "multi_target_not_supported"


def test_multi_target_per_target_counts(tmp_path):
    """Two compatible targets (car + motorcycle) -> per_target_counts populated."""
    from backend.app.services.counting_service import CountingService
    from backend.app.services.query_refiner import QueryPlan
    from backend.app.vision.object_detector import Detection

    root = tmp_path
    frame = root / "VID" / "frames" / "000000000.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"fixture")
    rows = [{"video_id": "VID", "source_frame_index_zero_based": 0, "frame_id": 0}]

    class _Det:
        backend = "yolo"; model_identity = "sha256:test"; device = "cpu"; confidence = 0.25
        calls = 0
        def detect(self, p):
            self.calls += 1
            return [
                Detection("car", 2, 0.92, (0, 0, 1, 1)),
                Detection("motorcycle", 3, 0.88, (0, 0, 1, 1)),
                Detection("car", 2, 0.80, (1, 1, 2, 2)),
            ]

    det = _Det()
    plan = QueryPlan(
        task_type="qa", original_query="Có bao nhiêu xe hơi và xe máy?",
        intent="count", requires_detector=True, detector_targets=["car", "motorcycle"],
    )
    service = CountingService(tmp_path, detector=det, top_n=1)
    output, metrics = service.count(plan, rows)

    assert metrics["detector_status"] == "ok"
    per = output[0]["per_target_counts"]
    assert per["car"] == 2
    assert per["motorcycle"] == 1
    assert output[0]["detector_count"] == 3


def test_single_detector_pass_for_multi_target(tmp_path):
    """detect() must be called exactly once per frame even for 2 targets."""
    from backend.app.services.counting_service import CountingService
    from backend.app.services.query_refiner import QueryPlan
    from backend.app.vision.object_detector import Detection

    root = tmp_path
    frame0 = root / "VID" / "frames" / "000000000.jpg"
    frame1 = root / "VID" / "frames" / "000000001.jpg"
    frame0.parent.mkdir(parents=True, exist_ok=True)
    frame0.write_bytes(b"f"); frame1.write_bytes(b"f")
    rows = [
        {"video_id": "VID", "source_frame_index_zero_based": 0, "frame_id": 0},
        {"video_id": "VID", "source_frame_index_zero_based": 1, "frame_id": 1},
    ]

    class _Det:
        backend = "yolo"; model_identity = "sha256:test"; device = "cpu"; confidence = 0.25
        calls = []
        def detect(self, p):
            self.calls.append(p)
            return [Detection("car", 2, 0.9, (0, 0, 1, 1))]

    det = _Det()
    plan = QueryPlan(
        task_type="kis", original_query="xe hơi và xe máy",
        intent="count", requires_detector=True, detector_targets=["car", "motorcycle"],
    )
    service = CountingService(tmp_path, detector=det, top_n=2)
    service.count(plan, rows)
    # Two frames -> two calls; NOT four (2 targets × 2 frames)
    assert len(det.calls) == 2


def test_vehicle_group_expansion(tmp_path):
    """target=vehicle expands to bicycle+car+motorcycle+bus+truck."""
    from backend.app.services.counting_service import CountingService
    from backend.app.services.query_refiner import QueryPlan
    from backend.app.vision.object_detector import Detection

    root = tmp_path
    frame = root / "VID" / "frames" / "000000000.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"fixture")
    rows = [{"video_id": "VID", "source_frame_index_zero_based": 0, "frame_id": 0}]

    class _Det:
        backend = "yolo"; model_identity = "sha256:test"; device = "cpu"; confidence = 0.25
        def detect(self, p):
            return [
                Detection("car", 2, 0.9, (0, 0, 1, 1)),
                Detection("motorcycle", 3, 0.9, (0, 0, 1, 1)),
                Detection("bus", 5, 0.9, (0, 0, 1, 1)),
                Detection("person", 0, 0.95, (0, 0, 1, 1)),   # NOT in vehicle group
            ]

    plan = QueryPlan(
        task_type="qa", original_query="Có bao nhiêu xe ở ngã tư?",
        intent="count", requires_detector=True, detector_targets=["vehicle"],
    )
    service = CountingService(tmp_path, detector=_Det(), top_n=1)
    output, metrics = service.count(plan, rows)

    assert metrics["detector_status"] == "ok"
    # car + motorcycle + bus = 3 (person excluded)
    assert output[0]["detector_count"] == 3
    per = output[0]["per_target_counts"]
    assert per.get("car") == 1
    assert per.get("motorcycle") == 1
    assert per.get("bus") == 1
    assert "person" not in per


def test_vehicle_group_excludes_unrelated(tmp_path):
    """target=vehicle never counts unrelated COCO classes."""
    from backend.app.services.counting_service import CountingService
    from backend.app.services.query_refiner import QueryPlan
    from backend.app.vision.object_detector import Detection

    root = tmp_path
    frame = root / "VID" / "frames" / "000000000.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"fixture")
    rows = [{"video_id": "VID", "source_frame_index_zero_based": 0, "frame_id": 0}]

    class _Det:
        backend = "yolo"; model_identity = "sha256:test"; device = "cpu"; confidence = 0.25
        def detect(self, p):
            return [
                Detection("person", 0, 0.99, (0, 0, 1, 1)),
                Detection("dog", 16, 0.99, (0, 0, 1, 1)),
                Detection("car", 2, 0.9, (0, 0, 1, 1)),
            ]

    plan = QueryPlan(
        task_type="qa", original_query="xe ở đây",
        intent="count", requires_detector=True, detector_targets=["vehicle"],
    )
    service = CountingService(tmp_path, detector=_Det(), top_n=1)
    output, _ = service.count(plan, rows)
    assert output[0]["detector_count"] == 1  # only car
    assert output[0]["per_target_counts"].get("car") == 1


def test_xe_vehicle_target_routes_and_expands():
    """DeterministicQueryParser maps 'xe' query -> vehicle target; CountingService expands."""
    from backend.app.services.query_refiner import DeterministicQueryParser
    from backend.app.services.counting_service import CountingService, SEMANTIC_TARGET_GROUPS
    parser = DeterministicQueryParser()
    plan = parser.parse("Có bao nhiêu xe ở ngã tư?", task_type="kis")
    assert plan.intent == "count"
    # Parser must produce 'vehicle' as a target (mapped from 'xe')
    assert "vehicle" in plan.detector_targets

    # CountingService must resolve 'vehicle' without error
    expanded, err = CountingService._resolve_targets(["vehicle"])
    assert err is None
    assert set(expanded) == set(SEMANTIC_TARGET_GROUPS["vehicle"])
