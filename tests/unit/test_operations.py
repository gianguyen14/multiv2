import json
from pathlib import Path

from backend.app.runtime.operations import memory_guard, resource_limits, resource_preflight, write_run_manifest


def test_memory_guard_refuses_low_available_memory(monkeypatch):
    monkeypatch.setattr("backend.app.runtime.operations.memory_snapshot", lambda: {
        "process_rss_bytes": 1, "total_ram_bytes": 100, "available_ram_bytes": 10,
        "swap_total_bytes": 100, "swap_free_bytes": 50})
    monkeypatch.setenv("MAX_MEMORY_FRACTION", "0.75")
    monkeypatch.setenv("MIN_AVAILABLE_MEMORY_BYTES", "1")
    try:
        memory_guard("ASR stage")
    except RuntimeError as exc:
        assert "existing artifacts were preserved" in str(exc)
    else:
        raise AssertionError("low memory must stop the stage")


def test_resource_limits_are_conservative(monkeypatch):
    for name in ("MAX_WORKERS", "DECODE_WORKERS", "OCR_WORKERS", "ASR_CONCURRENCY"):
        monkeypatch.delenv(name, raising=False)
    limits = resource_limits()
    assert limits["max_workers"] == limits["decode_workers"] == 1
    assert limits["ocr_workers"] == limits["asr_concurrency"] == 1


def test_resource_preflight_and_run_manifest(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "processed"
    report = resource_preflight(source, output)
    assert report["source_bytes"] == 5
    manifest = write_run_manifest(output / "run_manifest.json", "preprocess", source,
        output, {"frame_id_policy": "zero_based"}, {"device": "cpu"})
    saved = json.loads((output / "run_manifest.json").read_text())
    assert saved["run_id"] == manifest["run_id"]
    assert saved["compute"]["device"] == "cpu"
    assert "API_KEY" not in json.dumps(saved)
