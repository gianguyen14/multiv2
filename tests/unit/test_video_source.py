"""Focused tests for lazy raw-video source resolution."""

import functools
import http.server
import threading

import pytest

from backend.app.services.video_source import resolve_video_path, valid_video_id


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("VIDEO_CACHE_DIR", "VIDEO_SOURCE_DIR", "VIDEO_SOURCE_URL_TEMPLATE"):
        monkeypatch.delenv(name, raising=False)


def test_valid_video_id():
    assert valid_video_id("L21_V001")
    assert valid_video_id("abc-123")
    assert not valid_video_id("../etc/passwd")
    assert not valid_video_id("a/b")
    assert not valid_video_id("")
    assert not valid_video_id("a" * 80)


def test_resolve_from_cache(monkeypatch, tmp_path):
    cache = tmp_path / "videos"
    cache.mkdir()
    video = cache / "L21_V001.mp4"
    video.write_bytes(b"cached-video")
    monkeypatch.setenv("VIDEO_CACHE_DIR", str(cache))
    assert resolve_video_path("L21_V001") == video


def test_resolve_from_source_dir_serves_in_place(monkeypatch, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    video = src / "L22_V030.mp4"
    video.write_bytes(b"source-video")
    monkeypatch.setenv("VIDEO_SOURCE_DIR", str(src))
    assert resolve_video_path("L22_V030") == video
    assert not (tmp_path / "videos").exists()


def test_resolve_prefers_cache_over_source(monkeypatch, tmp_path):
    cache = tmp_path / "videos"
    cache.mkdir()
    cached = cache / "L21_V001.mp4"
    cached.write_bytes(b"cached")
    src = tmp_path / "src"
    src.mkdir()
    (src / "L21_V001.mp4").write_bytes(b"source")
    monkeypatch.setenv("VIDEO_CACHE_DIR", str(cache))
    monkeypatch.setenv("VIDEO_SOURCE_DIR", str(src))
    assert resolve_video_path("L21_V001") == cached


def _serve(directory):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_lazy_download_from_url_template(monkeypatch, tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    (served / "L21_V001.mp4").write_bytes(b"remote-video-bytes")
    server, thread = _serve(served)
    try:
        monkeypatch.setenv("VIDEO_SOURCE_URL_TEMPLATE", f"http://127.0.0.1:{server.server_address[1]}/{{video_id}}.mp4")
        cache = tmp_path / "videos"
        monkeypatch.setenv("VIDEO_CACHE_DIR", str(cache))
        path = resolve_video_path("L21_V001")
        assert path.read_bytes() == b"remote-video-bytes"
        assert resolve_video_path("L21_V001") == path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_lazy_download_is_atomic_on_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_SOURCE_URL_TEMPLATE", "http://127.0.0.1:1/{video_id}.mp4")
    cache = tmp_path / "videos"
    monkeypatch.setenv("VIDEO_CACHE_DIR", str(cache))
    with pytest.raises(OSError):
        resolve_video_path("L21_V001")
    assert not cache.exists() or not list(cache.glob("*.part"))


def test_rejects_non_http_template(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_SOURCE_URL_TEMPLATE", "file:///etc/passwd/{video_id}")
    monkeypatch.setenv("VIDEO_CACHE_DIR", str(tmp_path / "videos"))
    with pytest.raises(RuntimeError, match="http"):
        resolve_video_path("L21_V001")


def test_rejects_malformed_http_template(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_SOURCE_URL_TEMPLATE", "http:///missing-host/{video_id}.mp4")
    monkeypatch.setenv("VIDEO_CACHE_DIR", str(tmp_path / "videos"))
    with pytest.raises(RuntimeError, match="http"):
        resolve_video_path("L21_V001")


def test_not_configured_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_CACHE_DIR", str(tmp_path / "videos"))
    with pytest.raises(FileNotFoundError, match="not configured"):
        resolve_video_path("L21_V001")


def test_invalid_id_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_SOURCE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="invalid video_id"):
        resolve_video_path("../etc/passwd")
