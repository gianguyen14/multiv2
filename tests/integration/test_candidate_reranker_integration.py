"""Integration tests for CandidateReranker with ConfiguredSearch."""

import pytest
from unittest.mock import MagicMock

from backend.app.services.configured_search import ConfiguredSearch
from backend.app.services.query_refiner import QueryPlan, VisualQuery
from backend.app.video.text_evidence import OCRRecord, ASRSegment


class DummyIndex:
    def __init__(self, hit_map):
        self.hit_map = hit_map

    def search(self, vector, top_k):
        return self.hit_map.get("default", [])


class DummyResolver:
    def resolve(self, frame_id):
        return {
            "video_id": "L22_V001",
            "frame_id": frame_id,
            "source_frame_index_zero_based": frame_id,
            "submission_frame_id": frame_id,
            "frame_uid": f"L22_V001:{str(frame_id).zfill(9)}",
            "timestamp_seconds": float(frame_id) / 25.0,
            "image_path": f"frames/{str(frame_id).zfill(9)}.jpg",
        }


class DummyEncoder:
    def encode_text(self, texts):
        return [[0.1] * 768 for _ in texts]

    def encode_image(self, img):
        return [[0.1] * 768]


def test_configured_search_reranker_integration():
    search = ConfiguredSearch()
    search._encoder = DummyEncoder()

    # Visual hits
    hits = [{"frame_id": 100, "score": 0.88}, {"frame_id": 200, "score": 0.85}]

    search._bundle = MagicMock()
    search._bundle.index = DummyIndex({"default": hits})
    search._bundle.resolver = DummyResolver()

    # OCR evidence contains exact match for frame 200
    search._ocr = [
        OCRRecord(
            video_id="L22_V001",
            frame_uid="L22_V001:000000200",
            source_frame_index_zero_based=200,
            timestamp_seconds=8.0,
            raw_text="XE LAM 79H-6072",
            normalized_text="xe lam 79h-6072",
            boxes=[],
            confidence=0.95,
        )
    ]
    search._asr = []

    plan = QueryPlan(
        task_type="kis",
        original_query="xe lam 79H-6072",
        visual_queries=[VisualQuery(language="vi", text="xe lam", channel="visual_vi")],
        exact_strings=["79H-6072"],
        lexical_terms=["79H-6072"],
    )

    # 1. Search with reranker enabled (default)
    results_reranked, timings_reranked = search._search_multi_path(plan, top_k=10, rerank=True)
    assert len(results_reranked) >= 2
    # Frame 200 with verified exact string match must rank #1
    assert results_reranked[0]["source_frame_index_zero_based"] == 200
    assert timings_reranked["rerank_ms"] >= 0.0

    # 2. Search with reranker disabled
    results_norerank, _ = search._search_multi_path(plan, top_k=10, rerank=False)
    # Frame 100 has higher initial visual score (0.88 vs 0.85) and ranks #1 when reranker is disabled
    assert results_norerank[0]["source_frame_index_zero_based"] == 100
