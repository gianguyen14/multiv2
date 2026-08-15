import numpy as np

from backend.app.retrieval.kis_pipeline import DenseTemporalRefiner, KISPipeline, frame_interval_hit
from backend.app.retrieval.video_multimodal import VideoSegmentCandidate
from backend.app.video.frame_id_policy import FrameIdPolicy


class FakeEncoder:
    def encode_text(self, texts):
        return np.array([[1.0, 0.0]], dtype=np.float32)

    def encode_image(self, images):
        vectors = []
        for image in images:
            red = np.asarray(image)[..., 0].mean() / 255
            vectors.append([red, 1 - red])
        return np.asarray(vectors, dtype=np.float32)


class FakeRetriever:
    def __init__(self, candidate):
        self.candidate = candidate

    def search(self, query, top_k):
        return [self.candidate]


def test_dense_refinement_selects_exact_semantic_frame(tmp_path, monkeypatch):
    candidate = VideoSegmentCandidate("video", 8, 8, 8, fused_score=0.2)
    frames = []
    for index in range(5, 12):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        if index == 9:
            image[..., 0] = 255
        frames.append(type("Frame", (), {"source_frame_index_zero_based": index, "image": image})())
    monkeypatch.setattr("backend.app.retrieval.kis_pipeline.decode_frame_indices", lambda path, indices: frames)
    refiner = DenseTemporalRefiner(FakeEncoder(), FrameIdPolicy("one_based"), radius_frames=3)
    result = KISPipeline(FakeRetriever(candidate), refiner, {"video": tmp_path / "video.mp4"}).search("red event")[0]
    assert result.source_frame_index_zero_based == 9
    assert result.submission_frame_id == 10
    assert frame_interval_hit(result, {"video_id": "video", "start_frame": 9, "end_frame": 9})


def test_dense_refinement_falls_back_and_enforces_budget(monkeypatch):
    candidate = VideoSegmentCandidate("video", 100, 100, 100, fused_score=0.7)
    captured = []

    def fail(path, indices):
        captured.extend(indices)
        raise RuntimeError("decode failed")

    monkeypatch.setattr("backend.app.retrieval.kis_pipeline.decode_frame_indices", fail)
    refiner = DenseTemporalRefiner(FakeEncoder(), FrameIdPolicy("zero_based"), radius_frames=100, max_frames=7)
    result = refiner.refine("missing.mp4", "query", candidate)
    assert len(captured) == 7
    assert result.source_frame_index_zero_based == result.coarse_frame == 100
    assert result.score == 0.7
