import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_operator_ui_health_search_and_media(tmp_path):
    frame = tmp_path / "video" / "frames" / "frame_000000001.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"image")
    app = create_app(lambda request: [{"video_id": "video", "frame_id": 1,
        "score": 0.9, "image_url": "/api/frames/video/frame_000000001.jpg"}], tmp_path)
    client = TestClient(app)
    assert client.get("/").status_code == 200
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["search_configured"] is True
    assert health["processed_root"] == str(tmp_path)
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").json()["status"] == "ready"
    response = client.post("/api/search", json={"query": "red car", "query_type": "kis"})
    assert response.json()["results"][0]["frame_id"] == 1
    assert client.get("/api/frames/video/frame_000000001.jpg").content == b"image"


def test_operator_page_contains_competition_controls():
    html = TestClient(create_app()).get("/").text
    for value in ("Textual KIS", "Video Q&amp;A", "TRAKE", "visual_score", "ocr_score", "asr_score", "evidence_sources", "Copy submission"):
        assert value in html
    assert "fetch('/api/search'" in html
    assert "onclick=" not in html and "innerHTML" not in html
    assert "addEventListener('click'" in html


def test_unconfigured_search_and_path_traversal_are_rejected(tmp_path):
    client = TestClient(create_app(media_root=tmp_path))
    assert client.get("/health/ready").status_code == 503
    assert client.post("/api/search", json={"query": "test"}).status_code == 503
    assert client.get("/api/frames/../secret").status_code == 404


def test_frame_server_rejects_symlink_escape(tmp_path):
    media = tmp_path / "media"
    outside = tmp_path / "outside"
    frame = outside / "frames" / "000000001.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"outside")
    media.mkdir()
    (media / "video").symlink_to(outside, target_is_directory=True)

    response = TestClient(create_app(media_root=media)).get(
        "/api/frames/video/000000001.jpg")

    assert response.status_code == 404


def test_cors_rejects_wildcard_with_credentials(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="explicit origins"):
        create_app()


def test_security_headers_present_on_responses(tmp_path):
    client = TestClient(create_app(media_root=tmp_path))
    res = client.get("/health")
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert res.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert res.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_search_endpoint_rejects_server_local_image_query(tmp_path):
    client = TestClient(create_app(search_handler=lambda r: [], media_root=tmp_path))
    # query_type="image" is rejected on /api/search
    res = client.post("/api/search", json={"query_type": "image", "image_path": "/etc/passwd"})
    assert res.status_code == 422  # Pydantic validation rejects "image" in Literal["kis", "qa", "trake"]

