"""Focused hermetic tests for the Qwen3-VL packed-DB runtime adapter.

These tests build a tiny synthetic FAISS generation + OCR/ASR spool with the
same on-disk schema as the verified AIC runtime DB and inject a stub encoder.
They never load the real 2B model, never touch corpus data, and never ingest.
"""

import hashlib
import json
import os
from pathlib import Path

import faiss
import numpy as np
import pytest

from backend.app.services.qwen_runtime_search import QwenRuntimeSearch

VIDEO = "VID"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_uid(index: int) -> str:
    return f"{VIDEO}:{index:09d}"


def _write_generation(root: Path, *, dim: int, n: int, backend: str = "qwen3_vl_embedding_2b"):
    """Write a schema-v1 generation identical in shape to the packed DB."""
    index = faiss.IndexFlatIP(dim)
    # One-hot rows keep every test fully deterministic.
    rows = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        rows[i, i % dim] = 1.0
    index.add(rows)
    generation = root / "generations" / "gen-test-000000000000000000000000"
    generation.mkdir(parents=True)
    mapping = {str(i): _canonical_uid(i) for i in range(n)}
    payloads = {}
    for i in range(n):
        uid = _canonical_uid(i)
        payloads[uid] = {
            "candidate_id": uid,
            "frame_uid": uid,
            "video_id": VIDEO,
            "source_frame_index_zero_based": i,
            "timestamp_seconds": float(i * 10),
        }
    faiss.write_index(index, str(generation / "frames.faiss"))
    (generation / "mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    (generation / "payloads.json").write_text(
        json.dumps({"schema_version": 1, "payloads": payloads}), encoding="utf-8"
    )
    metadata = {
        "schema_version": 1,
        "generation_id": generation.name,
        "index_type": "flat",
        "embedding_dim": dim,
        "vector_count": n,
        "video_ids": [VIDEO],
        "encoder_identity": {
            "provider": "sentence-transformers",
            "backend": backend,
            "model": "Qwen/Qwen3-VL-Embedding-2B",
            "embedding_dim": dim,
            "normalization": "l2",
            "instruction": "Retrieve the video frame that best matches the described visual scene.",
        },
    }
    metadata["artifact_sha256"] = {
        name: _sha256(generation / name)
        for name in ("frames.faiss", "mapping.json", "payloads.json")
    }
    (generation / "generation.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "CURRENT").write_text(
        json.dumps({"schema_version": 1, "generation_id": generation.name}), encoding="utf-8"
    )
    return index.ntotal, metadata


def _write_spool(root: Path, kind: str, rows) -> None:
    target = root / kind
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{VIDEO}.json").write_text(json.dumps(rows), encoding="utf-8")


class StubEncoder:
    """Maps queries to one-hot unit vectors: query 't<i>' -> e_{i % dim}."""

    def __init__(self, dim: int):
        self.dim = dim

    def encode_query(self, query: str, dimension: int) -> np.ndarray:
        import re

        match = re.search(r"t(\d+)", query)
        hot = int(match.group(1)) % dimension if match else 0
        vector = np.zeros((dimension,), dtype=np.float32)
        vector[hot] = 1.0
        return vector


@pytest.fixture()
def runtime(tmp_path):
    root = tmp_path / "runtime"
    (root / "index").mkdir(parents=True)
    n, metadata = _write_generation(root / "index", dim=4, n=6)
    ocr = [
        {
            "video_id": VIDEO,
            "timestamp_seconds": 10.0,  # nearest indexed frame index 1
            "raw_text": "bien so 50H 12345",
            "normalized_text": "biển số 50h 12345",
        }
    ]
    asr_rows = [
        {
            "video_id": VIDEO,
            "start_seconds": 25.0,
            "end_seconds": 35.0,
            "raw_transcript": "anh ta dang noi chuyen",
            "normalized_transcript": "anh ta đang nói chuyện",
        }
    ]
    _write_spool(root, "ocr", ocr)
    _write_spool(root, "asr", asr_rows)
    return root, n, metadata


def _search(runtime, model_dir, query, top_k=5, **kwargs):
    root, _, _ = runtime
    provider = QwenRuntimeSearch(
        processed_root=root,
        model_dir=model_dir,
        encoder_factory=lambda: StubEncoder(4),
    )
    for key, value in kwargs.items():
        setattr(provider, key, value)
    return provider


def _touch_model(model_dir: Path) -> Path:
    scripts = model_dir / "scripts"
    scripts.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"x")
    (model_dir / "config.json").write_text("{}")
    (scripts / "qwen3_vl_embedding.py").write_text("# stub")
    return model_dir


def test_configured_and_readiness(runtime, tmp_path):
    root, n, metadata = runtime
    model_dir = _touch_model(tmp_path / "model")
    provider = _search(runtime, model_dir, "x")
    assert provider.configured is True
    ready = provider.readiness()
    assert ready["ready"] is True
    assert ready["generation_id"] == metadata["generation_id"]
    assert n == 6


def test_readiness_reports_missing_model_and_missing_root(tmp_path):
    root = tmp_path / "runtime"
    (root / "index").mkdir(parents=True)
    _write_generation(root / "index", dim=4, n=2)
    empty_model = tmp_path / "no-model"
    provider = QwenRuntimeSearch(processed_root=root, model_dir=empty_model)
    assert provider.readiness()["ready"] is False
    assert "not available" in provider.readiness()["reason"]
    unconfigured = QwenRuntimeSearch(processed_root=None)
    assert unconfigured.configured is False
    assert unconfigured.readiness()["ready"] is False


def test_refuses_siglip_index(runtime, tmp_path):
    root = tmp_path / "runtime-siglip"
    (root / "index").mkdir(parents=True)
    _write_generation(root / "index", dim=4, n=2, backend="siglip2")
    provider = QwenRuntimeSearch(
        processed_root=root,
        encoder_factory=lambda: StubEncoder(4),
    )
    assert provider.readiness()["ready"] is False
    with pytest.raises(RuntimeError, match="non-Qwen index"):
        provider.handle({"query_type": "kis", "query": "t1", "top_k": 5})


def test_kis_search_returns_verified_visual_top(runtime, tmp_path):
    model_dir = _touch_model(tmp_path / "model")
    provider = _search(runtime, model_dir, "x")
    results = provider.handle({"query_type": "kis", "query": "t2", "top_k": 3})
    assert results
    top = results[0]
    assert top["frame_uid"] == _canonical_uid(2)
    assert top["frame_id"] == 2
    assert top["video_id"] == VIDEO
    assert set(top) >= {
        "video_id", "frame_id", "source_frame_index_zero_based", "frame_uid",
        "timestamp_seconds", "score", "visual_score", "ocr_score", "asr_score",
        "ocr_evidence", "asr_evidence", "rank",
    }
    assert top["rank"] == 1
    assert top["visual_score"] > 0.0
    assert len(results) == 3


def test_ocr_evidence_expands_candidates(runtime, tmp_path):
    model_dir = _touch_model(tmp_path / "model")
    provider = _search(runtime, model_dir, "x")
    # Query '50h' is absent visually (no t<digit>); lexical OCR should surface
    # the frame nearest timestamp 10.0 -> indexed frame index 1.
    results = provider.handle({"query_type": "kis", "query": "bien so 50h 12345", "top_k": 6})
    ocr_hit = next((row for row in results if row["ocr_score"] > 0.0), None)
    assert ocr_hit is not None
    assert ocr_hit["frame_uid"] == _canonical_uid(1)
    assert "50h" in (ocr_hit["ocr_evidence"] or "").lower()


def test_qa_uses_same_pipeline_and_empty_query_rejected(runtime, tmp_path):
    model_dir = _touch_model(tmp_path / "model")
    provider = _search(runtime, model_dir, "x")
    results = provider.handle({"query_type": "qa", "query": "t0", "top_k": 2})
    assert results[0]["frame_uid"] == _canonical_uid(0)
    with pytest.raises(ValueError, match="query is required"):
        provider.handle({"query_type": "qa", "query": "   ", "top_k": 2})


def test_trake_success_returns_ordered_frames(runtime, tmp_path):
    model_dir = _touch_model(tmp_path / "model")
    provider = _search(runtime, model_dir, "x")
    results = provider.handle(
        {"query_type": "trake", "events": ["t0", "t2"], "top_k": 5}
    )
    assert len(results) == 1
    row = results[0]
    assert row["video_id"] == VIDEO
    assert row["frame_ids"] == [0, 2]
    assert row["frame_id"] == 0
    assert row["events"] == [{"frame_id": 0}, {"frame_id": 2}]


def test_trake_impossible_sequence_errors(runtime, tmp_path):
    model_dir = _touch_model(tmp_path / "model")
    provider = _search(runtime, model_dir, "x")
    # Six identical events exhaust the six candidate frames; the seventh event
    # cannot find a strictly larger frame in the same video.
    with pytest.raises(ValueError, match="no single video covers every event"):
        provider.handle({"query_type": "trake", "events": ["t0"] * 7, "top_k": 5})


def test_trake_empty_events_rejected(runtime, tmp_path):
    model_dir = _touch_model(tmp_path / "model")
    provider = _search(runtime, model_dir, "x")
    with pytest.raises(ValueError, match="non-empty ordered events"):
        provider.handle({"query_type": "trake", "events": [], "top_k": 5})


def test_qa_attaches_evidence_answer(runtime, tmp_path):
    model_dir = _touch_model(tmp_path / "model")
    provider = _search(runtime, model_dir, "x")
    results = provider.handle(
        {"query_type": "qa", "query": "bien so 50h 12345", "top_k": 6}
    )
    answered = [row for row in results if row.get("answer")]
    assert answered
    assert answered[0]["answer"] == "bien so 50H 12345"
    assert len(answered[0]["answer"]) <= 100


def test_image_search_explicit_error(runtime, tmp_path):
    model_dir = _touch_model(tmp_path / "model")
    provider = _search(runtime, model_dir, "x")
    with pytest.raises(RuntimeError, match="image search is not supported"):
        provider.search_image(object(), top_k=5)
    with pytest.raises(RuntimeError, match="image search is not supported"):
        provider.handle({"query_type": "image", "query": "x", "top_k": 5})


def test_capabilities_reported(runtime, tmp_path):
    model_dir = _touch_model(tmp_path / "model")
    provider = _search(runtime, model_dir, "x")
    caps = provider.status()["capabilities"]
    assert caps == {
        "kis": True, "qa": True, "trake": True, "image": False,
        "thumbnails": False, "raw_video_preview": False,
    }


def test_search_is_readonly(runtime, tmp_path):
    model_dir = _touch_model(tmp_path / "model")
    root, _, _ = runtime
    before = sorted(
        (str(p.relative_to(root)), p.stat().st_mtime_ns)
        for p in root.rglob("*") if p.is_file()
    )
    provider = _search(runtime, model_dir, "x")
    provider.handle({"query_type": "kis", "query": "t3", "top_k": 4})
    after = sorted(
        (str(p.relative_to(root)), p.stat().st_mtime_ns)
        for p in root.rglob("*") if p.is_file()
    )
    assert after == before


def test_create_app_defaults_to_qwen_backend(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    monkeypatch.delenv("SEARCH_BACKEND", raising=False)
    monkeypatch.delenv("SEARCH_ENCODER", raising=False)
    client = TestClient(create_app(media_root=tmp_path))
    health = client.get("/health").json()
    assert health["search"]["backend"] == "qwen3_vl"
    assert health["search"]["capabilities"]["kis"] is True
    assert health["search"]["capabilities"]["image"] is False
    assert client.get("/health/ready").status_code == 503


def test_create_app_selects_provider_by_env(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    # Canonical SEARCH_BACKEND selector.
    monkeypatch.setenv("SEARCH_BACKEND", "qwen3_vl")
    client = TestClient(create_app(media_root=tmp_path))
    health = client.get("/health").json()
    assert health["search"]["backend"] == "qwen3_vl"
    # processed root is configured; readiness still 503 because tmp_path holds
    # no valid index generation.
    assert health["search_configured"] is True
    assert client.get("/health/ready").status_code == 503


def test_create_app_accepts_legacy_encoder_alias(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    monkeypatch.delenv("SEARCH_BACKEND", raising=False)
    monkeypatch.setenv("SEARCH_ENCODER", "qwen3_vl")
    client = TestClient(create_app(media_root=tmp_path))
    assert client.get("/health").json()["search"]["backend"] == "qwen3_vl"


def test_create_app_legacy_siglip_backend(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    monkeypatch.setenv("SEARCH_BACKEND", "siglip2")
    client = TestClient(create_app(media_root=tmp_path))
    search = client.get("/health").json()["search"]
    # ConfiguredSearch reports device fields, not a qwen backend.
    assert "backend" not in search
    assert search["configured"] is True


def test_create_app_rejects_unknown_backend(monkeypatch):
    from backend.app.main import create_app

    monkeypatch.setenv("SEARCH_BACKEND", "banana")
    with pytest.raises(RuntimeError, match="Unknown SEARCH_BACKEND"):
        create_app()


def test_siglip_backend_refuses_qwen_index(tmp_path):
    """SigLIP2 must never silently query a Qwen-built generation."""
    from backend.app.services.configured_search import ConfiguredSearch

    root = tmp_path / "runtime"
    (root / "index").mkdir(parents=True)
    _write_generation(root / "index", dim=4, n=2, backend="qwen3_vl_embedding_2b")
    provider = ConfiguredSearch(processed_root=root)
    ready = provider.readiness()
    assert ready["ready"] is False
    assert "SEARCH_BACKEND" in ready["reason"]
    with pytest.raises(RuntimeError, match="Qwen3-VL embeddings"):
        provider._initialize()
