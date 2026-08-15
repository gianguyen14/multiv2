from eval.m22_end_to_end_benchmark import run_benchmark


def test_benchmark_marks_unconfigured_stages_unavailable(monkeypatch):
    monkeypatch.delenv("VIDEO_PROCESSED_ROOT", raising=False)
    report = run_benchmark()
    assert report["end_to_end_kis"] == "unavailable"
    assert report["reason"] == "VIDEO_PROCESSED_ROOT is not configured"
