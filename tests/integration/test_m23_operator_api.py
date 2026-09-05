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
    # The refactored frontend splits runtime JS/CSS out of index.html into
    # served assets (/scripts/*.js, /styles/main.css). Assertions that target
    # competition-control strings and safe-DOM properties therefore now read
    # the served assets rather than the inline page; security intent is
    # preserved (no onclick=, no innerHTML, no eval, listeners via addEventListener).
    client = TestClient(create_app())
    html = client.get("/").text
    for value in ("Textual KIS", "Video Q&amp;A", "TRAKE", "Image Search"):
        assert value in html
    api_js = client.get("/scripts/api.js").text
    app_js = client.get("/scripts/app.js").text
    assert "fetch('/api/search'" in api_js
    assert "fetch('/api/search/image?top_k=100'" in api_js
    for value in ("visual_score", "ocr_score", "asr_score", "evidence_sources", "Copy submission"):
        assert value in app_js
    assert "addEventListener('click'" in app_js
    # Check served runtime bodies for unsafe constructs. index.html is the
    # page at "/"; the rest are asset routes.
    for path, asset in (("/", "index.html"), ("/styles/main.css", "styles/main.css"),
                        ("/scripts/api.js", "scripts/api.js"),
                        ("/scripts/app.js", "scripts/app.js"),
                        ("/scripts/shortcuts.js", "scripts/shortcuts.js")):
        body = client.get(path).text
        assert f"<html" in body or "function" in body or "{" in body, asset
        assert "onclick=" not in body, asset
        assert "innerHTML" not in body, asset
        assert "eval(" not in body, asset


def test_frontend_static_assets_are_served_and_traversal_rejected(tmp_path):
    client = TestClient(create_app(media_root=tmp_path))
    expected = {
        "/": "text/html",
        "/styles/main.css": "text/css",
        "/scripts/api.js": "text/javascript",
        "/scripts/app.js": "text/javascript",
        "/scripts/shortcuts.js": "text/javascript",
    }
    for path, ctype in expected.items():
        res = client.get(path)
        assert res.status_code == 200, path
        assert res.headers["content-type"].startswith(ctype), path
    # New asset routes must not allow path traversal outside frontend/src.
    for path in ("/scripts/../../main.py", "/styles/%2e%2e%2fmain.py",
                 "/scripts/..%2f..%2fmain.py", "/scripts/nope.js", "/styles/nope.css"):
        assert client.get(path).status_code == 404, path


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

