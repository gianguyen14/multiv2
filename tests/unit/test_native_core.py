import numpy as np
import pytest

from backend.app.native import (
    align_trake_events,
    merge_temporal_regions,
    native_available,
    native_status,
    smooth_scores,
    temporal_nms_indices,
)
from backend.app.retrieval.trake import EventCandidate, TRAKEAligner


def _run_in_mode(monkeypatch, mode, func, *args, **kwargs):
    monkeypatch.setenv("UVR_NATIVE_CORE", mode)
    return func(*args, **kwargs)


def test_native_status_is_explicit(monkeypatch):
    monkeypatch.setenv("UVR_NATIVE_CORE", "python")
    status = native_status()
    assert status["mode"] == "python"
    assert status["backend"] == "python"
    assert isinstance(status["available"], bool)


def test_temporal_nms_cpp_matches_python(monkeypatch):
    if not native_available():
        pytest.skip("C++ extension not built in this environment")

    video_ids = ["a", "a", "b", "a", "b", "a", "c"]
    frame_ids = [100, 120, 5, 161, 50, 220, 1]

    py = _run_in_mode(
        monkeypatch, "python", temporal_nms_indices, video_ids, frame_ids, 60, 5
    )
    cpp = _run_in_mode(
        monkeypatch, "cpp", temporal_nms_indices, video_ids, frame_ids, 60, 5
    )
    # a: 100 kept, 120 suppressed, 161 kept, 220 suppressed because gap=59.
    # b: 5 kept, 50 suppressed. c: 1 kept.
    assert cpp == py == [0, 2, 3, 6]


def test_temporal_region_merge_cpp_matches_python(monkeypatch):
    if not native_available():
        pytest.skip("C++ extension not built in this environment")

    frame_ids = [100, 130, 500, 520, 900]
    scores = [0.8, 0.9, 0.7, 0.6, 0.95]

    py = _run_in_mode(
        monkeypatch,
        "python",
        merge_temporal_regions,
        frame_ids,
        scores,
        50,
        1000,
        2,
    )
    cpp = _run_in_mode(
        monkeypatch,
        "cpp",
        merge_temporal_regions,
        frame_ids,
        scores,
        50,
        1000,
        2,
    )
    assert cpp == py
    assert cpp == [
        (50, 180, [100, 130], 0.9),
        (850, 950, [900], 0.95),
    ]


def test_temporal_smoothing_cpp_matches_python(monkeypatch):
    if not native_available():
        pytest.skip("C++ extension not built in this environment")

    raw = np.asarray([0.1, 0.5, 0.2, 0.9, -0.1, 0.3], dtype=np.float32)
    py = _run_in_mode(monkeypatch, "python", smooth_scores, raw, 0.8, 0.2, 1)
    cpp = _run_in_mode(monkeypatch, "cpp", smooth_scores, raw, 0.8, 0.2, 1)
    np.testing.assert_allclose(cpp, py, rtol=1e-6, atol=1e-6)


def test_trake_cpp_matches_python_and_tie_break(monkeypatch):
    if not native_available():
        pytest.skip("C++ extension not built in this environment")

    frames = [[2, 1, 10], [4, 3, 20], [7, 6, 30]]
    scores = [[1.0, 1.0, 0.1], [1.0, 1.0, 0.1], [1.0, 1.0, 0.1]]

    py = _run_in_mode(monkeypatch, "python", align_trake_events, frames, scores, 0.0, None)
    cpp = _run_in_mode(monkeypatch, "cpp", align_trake_events, frames, scores, 0.0, None)
    assert cpp == py == (3.0, [1, 3, 6])


def test_trake_cpp_respects_gap_and_transition_penalty(monkeypatch):
    if not native_available():
        pytest.skip("C++ extension not built in this environment")

    frames = [[10, 20], [30, 100]]
    scores = [[1.0, 1.2], [0.8, 5.0]]

    py = _run_in_mode(monkeypatch, "python", align_trake_events, frames, scores, 0.01, 50)
    cpp = _run_in_mode(monkeypatch, "cpp", align_trake_events, frames, scores, 0.01, 50)
    assert cpp == py
    assert cpp is not None
    assert cpp[1] == [20, 30]
    assert cpp[0] == pytest.approx(1.9)


def test_public_trake_aligner_runs_in_required_cpp_mode(monkeypatch):
    if not native_available():
        pytest.skip("C++ extension not built in this environment")

    monkeypatch.setenv("UVR_NATIVE_CORE", "cpp")
    events = [
        [EventCandidate("v", 2, 1.0), EventCandidate("v", 1, 1.0)],
        [EventCandidate("v", 4, 1.0), EventCandidate("v", 3, 1.0)],
    ]
    result = TRAKEAligner().align(events)
    assert result is not None
    assert result.video_id == "v"
    assert result.frame_ids == [1, 3]
    assert result.score == 2.0


def test_invalid_native_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("UVR_NATIVE_CORE", "invalid")
    with pytest.raises(ValueError, match="UVR_NATIVE_CORE"):
        native_status()
