import json

import numpy as np

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.indexes.faiss_siglip_index import FaissSigLIPIndex
from backend.app.retrieval.candidate_resolver import PersistentCandidateResolver
from backend.app.video.frame_index import load_current_frame_index
from backend.app.video.ingest import ingest_path


class FakeEncoder:
    embedding_dim = 4

    def encode_image(self, images, batch_size=8, normalize=True):
        values = []
        for index, _ in enumerate(images):
            vector = np.zeros(4, dtype=np.float32)
            vector[index % 4] = 1
            values.append(vector)
        return np.asarray(values)


def test_m15_ingestion_resume_index_and_resolution(tmp_path):
    config = VideoIngestConfig(processed_root=tmp_path, sample_interval_seconds=1.0)
    first = ingest_path("tests/fixtures/test_5s.mp4", FakeEncoder(), config)
    assert first["videos_succeeded"] == 1
    assert first["indexed_frames"] == 5
    second = ingest_path("tests/fixtures/test_5s.mp4", FakeEncoder(), config)
    assert second["results"][0]["status"] == "resumed"
    assert second["indexed_frames"] == 5
    bundle = load_current_frame_index(tmp_path / "index")
    index = bundle.index
    resolver = bundle.resolver
    hit = index.search(np.array([1, 0, 0, 0], dtype=np.float32), 1)[0]
    payload = resolver.resolve(hit["frame_id"])
    assert payload["video_id"] == "test_5s"
    assert isinstance(payload["source_frame_index_zero_based"], int)
    assert payload["submission_frame_id"] == payload["source_frame_index_zero_based"]
    assert payload["timestamp_seconds"] is not None
