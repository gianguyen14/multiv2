from backend.app.utils.latency import LatencyRecorder


def test_latency_summary_reports_required_percentiles():
    recorder = LatencyRecorder()
    recorder.values = [1, 2, 3, 4, 5]
    assert recorder.summary() == {"mean_ms": 3, "p50_ms": 3, "p95_ms": 5, "p99_ms": 5}
