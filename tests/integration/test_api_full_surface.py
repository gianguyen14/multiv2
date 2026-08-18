from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import create_app


class FakeConfiguredSearch:
    def __init__(self, processed_root: Path, *, configured: bool = True):
        self.processed_root = Path(processed_root)
        self.configured = configured
        self.requests: list[dict] = []
        self.image_calls: list[dict] = []
        self.last_query_plan = None
        self.last_query_metrics = {"fake": True}

    def handle(self, request: dict):
        self.requests.append(dict(request))
        query_type = request.get("query_type", "kis")
        if query_type == "trake":
            return [
                {
                    "video_id": "V_TRAKE",
                    "frame_id": 10,
                    "frame_ids": [10, 20, 30],
                    "events": [{"frame_id": 10}, {"frame_id": 20}, {"frame_id": 30}],
                    "score": 0.9,
                    "image_url": "/api/frames/V_TRAKE/000000010.jpg",
                }
            ]
        if query_type == "qa":
            return [
                {
                    "video_id": "V_QA",
                    "frame_id": 22,
                    "timestamp_seconds": 1.5,
                    "score": 0.8,
                    "answer": "red",
                    "confidence": 0.9,
                    "evidence_sources": ["visual"],
                    "image_url": "/api/frames/V_QA/000000022.jpg",
                }
            ]
        return [
            {
                "video_id": "V_KIS",
                "frame_id": 12,
                "timestamp_seconds": 0.4,
                "score": 0.7,
                "visual_score": 0.6,
                "ocr_score": 0.2,
                "asr_score": 0.1,
                "image_url": "/api/frames/V_KIS/000000012.jpg",
            }
        ]

    def search_image(self, image, top_k: int = 100, deduplicate: bool = True):
        self.image_calls.append(
            {
                "size": image.size,
                "mode": image.mode,
                "top_k": top_k,
                "deduplicate": deduplicate,
            }
        )
        return [
            {
                "video_id": "V_IMAGE",
                "frame_id": 7,
                "score": 0.95,
                "image_url": "/api/frames/V_IMAGE/000000007.jpg",
            }
        ]

    def status(self):
        return {"configured": self.configured, "backend": "fake"}

    def readiness(self):
        return {"ready": self.configured, "reason": None if self.configured else "not configured"}


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _client(tmp_path: Path, *, configured: bool = True):
    search = FakeConfiguredSearch(tmp_path, configured=configured)
    app = create_app(configured_search=search, media_root=tmp_path)
    return TestClient(app), search


def test_root_health_ready_and_security_headers(tmp_path):
    client, _ = _client(tmp_path)

    root = client.get("/")
    assert root.status_code == 200
    assert "AIC 2026 Retrieval Console" in root.text
    assert root.headers["x-content-type-options"] == "nosniff"
    assert root.headers["x-frame-options"] == "SAMEORIGIN"
    assert root.headers["referrer-policy"] == "strict-origin-when-cross-origin"

    live = client.get("/health/live")
    assert live.status_code == 200
    payload = live.json()
    assert payload["status"] == "ok"
    assert payload["search_configured"] is True

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["search_readiness"]["ready"] is True


def test_ready_returns_503_when_search_is_not_configured(tmp_path):
    client, _ = _client(tmp_path, configured=False)
    response = client.get("/health/ready")
    assert response.status_code == 503


def test_kis_qa_and_trake_request_contracts(tmp_path):
    client, search = _client(tmp_path)

    kis = client.post(
        "/api/search",
        json={"query": "red truck", "query_type": "kis", "top_k": 5},
    )
    assert kis.status_code == 200
    assert kis.json()["results"][0]["video_id"] == "V_KIS"
    assert search.requests[-1]["query_type"] == "kis"
    assert search.requests[-1]["top_k"] == 5

    qa = client.post(
        "/api/search",
        json={"query": "what color is the car?", "query_type": "qa"},
    )
    assert qa.status_code == 200
    assert qa.json()["results"][0]["answer"] == "red"
    assert search.requests[-1]["query_type"] == "qa"

    trake = client.post(
        "/api/search",
        json={
            "query": "",
            "query_type": "trake",
            "events": ["person enters", "person sits", "person leaves"],
            "temporal_refine": False,
        },
    )
    assert trake.status_code == 200
    row = trake.json()["results"][0]
    assert row["frame_ids"] == [10, 20, 30]
    assert search.requests[-1]["events"] == ["person enters", "person sits", "person leaves"]
    assert search.requests[-1]["temporal_refine"] is False


def test_search_validation_covers_empty_query_invalid_mode_and_top_k(tmp_path):
    client, _ = _client(tmp_path)

    assert client.post("/api/search", json={"query": "", "query_type": "kis"}).status_code == 400
    assert client.post("/api/search", json={"query": "x", "query_type": "image"}).status_code == 422
    assert client.post("/api/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/api/search", json={"query": "x", "top_k": 1001}).status_code == 422
    assert client.post(
        "/api/search",
        json={"query_type": "trake", "events": ["", "valid"]},
    ).status_code == 422


def test_image_search_supports_multipart_and_raw_image_bodies(tmp_path):
    client, search = _client(tmp_path)
    body = _png_bytes()

    multipart = client.post(
        "/api/search/image?top_k=7",
        files={"file": ("query.png", body, "image/png")},
    )
    assert multipart.status_code == 200
    assert multipart.json()["results"][0]["video_id"] == "V_IMAGE"
    assert search.image_calls[-1] == {
        "size": (4, 3),
        "mode": "RGB",
        "top_k": 7,
        "deduplicate": True,
    }

    raw = client.post(
        "/api/search/image?top_k=3&raw=true",
        content=body,
        headers={"content-type": "image/png"},
    )
    assert raw.status_code == 200
    assert search.image_calls[-1]["top_k"] == 3
    assert search.image_calls[-1]["deduplicate"] is False


def test_image_search_rejects_unsupported_or_corrupt_input(tmp_path):
    client, _ = _client(tmp_path)

    unsupported = client.post(
        "/api/search/image",
        content=b"hello",
        headers={"content-type": "text/plain"},
    )
    assert unsupported.status_code == 415

    corrupt = client.post(
        "/api/search/image",
        content=b"not-a-real-png",
        headers={"content-type": "image/png"},
    )
    assert corrupt.status_code == 400


def test_frame_serving_and_path_safety(tmp_path):
    client, _ = _client(tmp_path)
    frame_dir = tmp_path / "V001" / "frames"
    frame_dir.mkdir(parents=True)
    frame = frame_dir / "000000123.jpg"
    frame.write_bytes(b"fake-jpeg")

    response = client.get("/api/frames/V001/000000123.jpg")
    assert response.status_code == 200
    assert response.content == b"fake-jpeg"
    assert response.headers["cache-control"] == "public, max-age=86400"

    assert client.get("/api/frames/V001/missing.jpg").status_code == 404
    assert client.get("/api/frames/V001/000000123.exe").status_code == 404
    assert client.get("/api/frames/V001/%2E%2E%2Fsecret.jpg").status_code in {404, 422}


def test_cors_preflight_for_local_frontend_origin(tmp_path):
    client, _ = _client(tmp_path)
    response = client.options(
        "/api/search",
        headers={
            "origin": "http://localhost:3000",
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
