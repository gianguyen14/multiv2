import sys
from types import ModuleType

from backend.app import model_cache


def module(monkeypatch, name, **values):
    value = ModuleType(name)
    for key, item in values.items():
        setattr(value, key, item)
    monkeypatch.setitem(sys.modules, name, value)


def test_visual_cached_and_missing(monkeypatch):
    module(monkeypatch, "huggingface_hub", snapshot_download=lambda **kwargs: "/cache/visual")
    assert model_cache.visual_status()["cache_path"] == "/cache/visual"
    module(monkeypatch, "huggingface_hub", snapshot_download=lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    assert not model_cache.visual_status()["cached"]


def test_asr_cached_and_missing(monkeypatch):
    package = ModuleType("faster_whisper")
    utils = ModuleType("faster_whisper.utils")
    utils.download_model = lambda *args, **kwargs: "/cache/asr"
    monkeypatch.setitem(sys.modules, "faster_whisper", package)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", utils)
    assert model_cache.asr_status("small")["cache_path"] == "/cache/asr"
    utils.download_model = lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError())
    assert not model_cache.asr_status("small")["cached"]


def test_model_cache_dir_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path))
    assert model_cache.model_cache_dir("huggingface") == tmp_path / "huggingface"
    assert model_cache.model_cache_dir("huggingface", tmp_path / "explicit") == tmp_path / "explicit"


def test_prepare_uses_provider_resolvers(monkeypatch):
    calls = []
    module(monkeypatch, "huggingface_hub", snapshot_download=lambda **kwargs: calls.append(kwargs) or "/visual")
    package = ModuleType("faster_whisper")
    utils = ModuleType("faster_whisper.utils")
    utils.download_model = lambda *args, **kwargs: calls.append((args, kwargs)) or "/asr"
    monkeypatch.setitem(sys.modules, "faster_whisper", package)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", utils)
    model_cache.prepare_visual()
    model_cache.prepare_asr("small")
    assert calls[0]["repo_id"] == "google/siglip2-base-patch16-224"
    assert "local_files_only" not in calls[0]
    assert calls[1][1]["local_files_only"] is False


def test_resolve_siglip2_and_whisper_snapshot_revisions(monkeypatch):
    from backend.app.embeddings.siglip2 import resolve_siglip2_revision
    from backend.app.video.text_backends import resolve_whisper_revision

    module(monkeypatch, "huggingface_hub", snapshot_download=lambda **kwargs: "/cache/snapshots/commit_sha_123")
    assert resolve_siglip2_revision("test-model") == "commit_sha_123"

    package = ModuleType("faster_whisper")
    utils = ModuleType("faster_whisper.utils")
    utils.download_model = lambda *args, **kwargs: "/cache/snapshots/whisper_commit_456"
    monkeypatch.setitem(sys.modules, "faster_whisper", package)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", utils)
    assert resolve_whisper_revision("small") == "whisper_commit_456"


def test_siglip2_encoder_identity_uses_resolved_revision(monkeypatch):
    from backend.app.embeddings.siglip2 import SigLIP2Encoder
    monkeypatch.setenv("SIGLIP_ENABLED", "true")
    monkeypatch.setattr("backend.app.embeddings.siglip2.resolve_siglip2_revision", lambda *args: "immutable_commit_789")
    encoder = SigLIP2Encoder(device="cpu", revision="immutable_commit_789")
    ident = encoder.identity()
    assert ident["revision"] == "immutable_commit_789"
    assert ident["model_name"] == "google/siglip2-base-patch16-224"
    assert ident["embedding_dim"] == 768
    assert not encoder._initialized  # Must not have loaded heavy model weights

