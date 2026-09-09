"""Tests for the optional seekable raw-video preview route."""

import functools
import http.server
import threading

from fastapi.testclient import TestClient

from backend.app.main import create_app


def _client_with_source(tmp_path, monkeypatch, content=b"0123456789abcdef"):
    src = tmp_path / "src"
    src.mkdir()
    (src / "L21_V001.mp4").write_bytes(content)
    monkeypatch.setenv("VIDEO_SOURCE_DIR", str(src))
    monkeypatch.setenv("VIDEO_CACHE_DIR", str(tmp_path / "videos"))
    return TestClient(create_app(media_root=tmp_path))


def test_video_route_serves_mp4_and_head(tmp_path, monkeypatch):
    client = _client_with_source(tmp_path, monkeypatch)
    response = client.get("/api/video/L21_V001")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.headers["accept-ranges"] == "bytes"
    assert response.content == b"0123456789abcdef"
    head = client.head("/api/video/L21_V001")
    assert head.status_code == 200
    assert head.headers["content-length"] == "16"


def test_video_route_supports_range_seek(tmp_path, monkeypatch):
    client = _client_with_source(tmp_path, monkeypatch)
    response = client.get("/api/video/L21_V001", headers={"Range": "bytes=4-7"})
    assert response.status_code == 206
    assert response.content == b"4567"
    assert response.headers["content-range"] == "bytes 4-7/16"
    suffix = client.get("/api/video/L21_V001", headers={"Range": "bytes=-4"})
    assert suffix.status_code == 206
    assert suffix.content == b"cdef"
    invalid = client.get("/api/video/L21_V001", headers={"Range": "bytes=100-"})
    assert invalid.status_code == 416
    assert invalid.headers["content-range"] == "bytes */16"
    multi = client.get("/api/video/L21_V001", headers={"Range": "bytes=0-1,4-5"})
    assert multi.status_code == 416
    assert multi.headers["content-range"] == "bytes */16"


def test_video_route_rejects_invalid_or_missing_video(tmp_path, monkeypatch):
    client = _client_with_source(tmp_path, monkeypatch)
    assert client.get("/api/video/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/api/video/L99_V999").status_code == 404
    assert client.get("/api/video/").status_code == 404


def test_video_route_lazy_downloads_from_url(tmp_path, monkeypatch):
    served = tmp_path / "served"
    served.mkdir()
    (served / "L21_V001.mp4").write_bytes(b"remote-bytes")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("VIDEO_SOURCE_URL_TEMPLATE", f"http://127.0.0.1:{server.server_address[1]}/{{video_id}}.mp4")
        cache = tmp_path / "videos"
        monkeypatch.setenv("VIDEO_CACHE_DIR", str(cache))
        client = TestClient(create_app(media_root=tmp_path))
        response = client.get("/api/video/L21_V001")
        assert response.status_code == 200
        assert response.content == b"remote-bytes"
        assert (cache / "L21_V001.mp4").read_bytes() == b"remote-bytes"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
