import json
from pathlib import Path
from types import SimpleNamespace

import projectctl
from backend.app.config.video_ingest_config import VideoIngestConfig


def test_all_ingest_defaults_are_qwen(monkeypatch):
    monkeypatch.delenv("INGEST_BACKEND", raising=False)
    assert VideoIngestConfig.from_env().ingest_backend == "qwen3_vl"
    parsed = projectctl.parser().parse_args(["ingest", "videos"])
    assert parsed.ingest_backend == "qwen3_vl"
    standalone = Path("backend/app/video/ingest.py").read_text()
    assert 'default=os.getenv("INGEST_BACKEND", "qwen3_vl")' in standalone


def test_siglip_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("INGEST_BACKEND", raising=False)
    parsed = projectctl.parser().parse_args(["ingest", "videos", "--ingest-backend", "siglip2"])
    assert parsed.ingest_backend == "siglip2"


def test_command_index_passes_canonical_identity(monkeypatch, tmp_path):
    captured = {}
    root = tmp_path / "processed"
    root.mkdir()
    monkeypatch.setenv("VIDEO_PROCESSED_ROOT", str(root))
    identity = {"backend": "qwen3_vl", "embedding_dim": 1024, "normalization": "l2"}

    class Manifest:
        encoder_identity = identity
        embedding_dim = 1024

    monkeypatch.setattr(projectctl, "__name__", projectctl.__name__)
    import backend.app.video.frame_store as frame_store
    monkeypatch.setattr(frame_store.FrameStore, "manifests", lambda self: iter([Manifest()]))
    import backend.app.video.frame_index as frame_index
    monkeypatch.setattr(frame_index, "build_frame_index", lambda *args, **kwargs: captured.update(kwargs) or SimpleNamespace(generation_id="gen-test", index=SimpleNamespace(index=SimpleNamespace(ntotal=1))))
    projectctl.command_index(SimpleNamespace(index_type="flat", json=True, verbose=False, output=None))
    assert captured["encoder_identity"] == identity
