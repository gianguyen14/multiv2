import json
import os
from fractions import Fraction
from pathlib import Path
import av
import numpy as np
import pytest
from PIL import Image

from backend.app.retrieval.trake import EventCandidate
from backend.app.services.configured_search import ConfiguredSearch
from backend.app.services.temporal_refiner import (
    TemporalRefiner,
    TemporalRefinerConfig,
)


def make_test_video(path: Path, num_frames: int = 30, fps: int = 10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = Fraction(1, fps)
        for idx in range(num_frames):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            if idx == 10:
                pixels[:] = (255, 0, 0)  # Pure red action
            elif idx == 20:
                pixels[:] = (0, 0, 255)  # Pure blue action
            else:
                pixels[:] = (50, 50, 50)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")

            frame.pts = idx
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class MockEncoder:
    embedding_dim = 4

    def identity(self):
        return {
            "provider": "mock",
            "model_name": "mock-siglip",
            "revision": "v1",
            "embedding_dim": 4,
            "normalization": "l2",
            "contract_version": "m15.1-v1",
        }

    def encode_text(self, texts, batch_size=8, normalize=True):
        vectors = []
        for t in texts:
            if "red" in t.lower():
                v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif "blue" in t.lower():
                v = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            else:
                v = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            if normalize:
                v = v / (np.linalg.norm(v) + 1e-9)
            vectors.append(v)
        return np.stack(vectors)

    def encode_image(self, images, batch_size=8, normalize=True):
        vectors = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            arr = np.asarray(img, dtype=np.float32)
            rgb = arr.mean(axis=(0, 1))
            v = np.array([rgb[0], rgb[1], rgb[2], 1.0], dtype=np.float32)
            if normalize:
                v = v / (np.linalg.norm(v) + 1e-9)
            vectors.append(v)
        return np.stack(vectors)


def test_configured_search_trake_end_to_end(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    vid_dir = processed / "L22_TEST"
    vid_dir.mkdir(parents=True)
    video_path = tmp_path / "L22_TEST.mp4"
    make_test_video(video_path, num_frames=30, fps=10)

    (vid_dir / "metadata.json").write_text(json.dumps({
        "video_id": "L22_TEST",
        "filename": "L22_TEST.mp4",
        "source_path": str(video_path),
        "real_frame_rate": "10/1",
        "avg_frame_rate": "10/1",
        "decoded_frame_count": 30,
    }))

    mock_enc = MockEncoder()

    class StubConfiguredSearch(ConfiguredSearch):
        def __init__(self):
            super().__init__(processed_root=processed, encoder_factory=lambda: mock_enc)
            self._bundle = None
            self._encoder = mock_enc

        def _initialize(self):
            pass

        def search(self, query, top_k=100):
            # Returns coarse candidate around frame 8 for red, frame 18 for blue
            if "red" in query.lower():
                return [{
                    "video_id": "L22_TEST",
                    "frame_id": 8,
                    "source_frame_index_zero_based": 8,
                    "score": 0.4,
                }]
            elif "blue" in query.lower():
                return [{
                    "video_id": "L22_TEST",
                    "frame_id": 18,
                    "source_frame_index_zero_based": 18,
                    "score": 0.4,
                }]
            return []

    search = StubConfiguredSearch()

    # 1. Search with temporal refinement enabled
    monkeypatch.setenv("TRAKE_TEMPORAL_REFINE_ENABLED", "true")
    monkeypatch.setenv("TRAKE_TEMPORAL_REFINE_SAMPLE_FPS", "10")
    monkeypatch.setenv("TRAKE_TEMPORAL_REFINE_WINDOW_SECONDS", "1.0")

    res = search.search_trake(["red action", "blue action"], top_k=10, temporal_refine=True)
    assert len(res) == 1
    assert res[0]["video_id"] == "L22_TEST"
    assert res[0]["refinement_used"] is True
    # Dense refinement should find the exact action frames 10 and 20!
    assert res[0]["frame_ids"] == [10, 20]
    assert res[0]["frame_id"] == 10

    # 2. Search with temporal refinement disabled
    res_no_refine = search.search_trake(["red action", "blue action"], top_k=10, temporal_refine=False)
    assert len(res_no_refine) == 1
    assert res_no_refine[0]["refinement_used"] is False
    # Without refinement, only the coarse frames 8 and 18 are returned
    assert res_no_refine[0]["frame_ids"] == [8, 18]


def test_offline_temporal_refinement_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    video_path = tmp_path / "offline_vid.mp4"
    make_test_video(video_path, num_frames=15, fps=5)

    processed_root = tmp_path / "processed"
    vid_dir = processed_root / "offline_vid"
    vid_dir.mkdir(parents=True)
    (vid_dir / "metadata.json").write_text(json.dumps({
        "video_id": "offline_vid",
        "filename": "offline_vid.mp4",
        "source_path": str(video_path),
        "real_frame_rate": "5/1",
        "decoded_frame_count": 15,
    }))

    config = TemporalRefinerConfig(enabled=True, sample_fps=5.0, window_seconds=1.0)
    refiner = TemporalRefiner(processed_root=processed_root, config=config, encoder=MockEncoder())

    coarse = [[EventCandidate("offline_vid", 5, 0.5)]]
    refined, metrics = refiner.refine_trake_candidates(["red"], coarse)

    assert metrics["refinement_used"] is True
    assert len(refined[0]) > 0


def test_kis_qa_image_search_contracts_unaffected(tmp_path):
    class StubSearch(ConfiguredSearch):
        def __init__(self):
            super().__init__(processed_root=tmp_path)
            self._bundle = None
            self._encoder = MockEncoder()

        def _initialize(self):
            pass

        def search(self, query, top_k=100):
            return [{"video_id": "v1", "source_frame_index_zero_based": 100, "score": 0.95, "frame_id": 100}]

        def search_image(self, image, top_k=100, deduplicate=True):
            return [{"video_id": "v1", "source_frame_index_zero_based": 100, "score": 0.95, "frame_id": 100}]

    search = StubSearch()

    # KIS
    kis_res = search.handle({"query_type": "kis", "query": "test query", "top_k": 10})
    assert len(kis_res) == 1
    assert kis_res[0]["video_id"] == "v1"

    # Image
    img_res = search.handle({"query_type": "image", "image": Image.new("RGB", (64, 64)), "top_k": 10})
    assert len(img_res) == 1
    assert img_res[0]["video_id"] == "v1"
