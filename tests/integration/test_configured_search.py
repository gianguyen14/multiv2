import numpy as np

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.main import create_app
from backend.app.services.configured_search import ConfiguredSearch
from backend.app.video.ingest import ingest_path
from fastapi.testclient import TestClient
from tests.m15_support import MeanRGBEncoder


class QueryEncoder:
    def encode_text(self, texts):
        return np.array([[1, 0, 0, 0]], dtype=np.float32)


def test_configured_app_lazily_opens_current_generation(tmp_path):
    root = tmp_path / "processed"
    ingest_path("tests/fixtures/test_5s.mp4", MeanRGBEncoder(), VideoIngestConfig(processed_root=root))
    search = ConfiguredSearch(root, encoder_factory=QueryEncoder)
    assert search._bundle is None and search._encoder is None
    client = TestClient(create_app(configured_search=search))
    health = client.get("/health").json()
    assert health["search_configured"] is True and health["processed_root"] == str(root)
    assert client.get("/health/ready").status_code == 200
    result = client.post("/api/search", json={"query": "red car", "top_k": 2}).json()["results"]
    assert len(result) == 2
    assert result[0]["video_id"] == "test_5s"
    assert isinstance(result[0]["frame_id"], int)
    assert search._bundle is not None and search._encoder is not None
