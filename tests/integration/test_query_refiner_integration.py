"""Integration tests for Query Intelligence and Multi-Path Retrieval."""

import os
import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from PIL import Image

from backend.app.services.configured_search import ConfiguredSearch
from backend.app.services.query_refiner import (
    QueryPlan,
    QueryRefiner,
    VisualQuery,
)
from backend.app.retrieval.trake import EventCandidate, TRAKEAligner
from backend.app.video.text_evidence import OCRRecord, ASRSegment


class DummyIndex:
    def __init__(self, hit_map):
        self.hit_map = hit_map

    def search(self, vector, top_k):
        # Return hits based on a deterministic query or default
        return self.hit_map.get("default", [])


class DummyResolver:
    def __init__(self, metadata_map):
        self.metadata_map = metadata_map

    def resolve(self, frame_id):
        return self.metadata_map.get(frame_id, {
            "video_id": "L22_V001",
            "frame_id": frame_id,
            "source_frame_index_zero_based": frame_id,
            "submission_frame_id": frame_id,
            "frame_uid": f"L22_V001:{str(frame_id).zfill(9)}",
            "timestamp_seconds": float(frame_id) / 25.0,
            "image_path": f"frames/{str(frame_id).zfill(9)}.jpg",
        })


class DummyEncoder:
    def encode_text(self, texts):
        return [[0.1] * 768 for _ in texts]

    def encode_image(self, img):
        return [[0.1] * 768]


class DummyBundle:
    def __init__(self, hits, metadata_map=None):
        self.index = DummyIndex({"default": hits})
        self.resolver = DummyResolver(metadata_map or {})
        self.metadata = {"video_ids": ["L22_V001"]}


# =========================================================================
# Test G: Dual Visual Retrieval Deduplication
# =========================================================================

def test_dual_visual_retrieval_dedup():
    search = ConfiguredSearch()
    search._encoder = DummyEncoder()
    # Mock index search to return different candidates for VI vs EN
    hits_vi = [{"frame_id": 100, "score": 0.9}, {"frame_id": 200, "score": 0.8}]
    hits_en = [{"frame_id": 200, "score": 0.85}, {"frame_id": 300, "score": 0.75}]

    call_count = 0
    def mock_search(vec, top_k):
        nonlocal call_count
        call_count += 1
        return hits_vi if call_count % 2 == 1 else hits_en

    search._bundle = MagicMock()
    search._bundle.index.search = mock_search
    search._bundle.resolver = DummyResolver({})
    search._ocr = []
    search._asr = []

    plan = QueryPlan(
        task_type="kis",
        original_query="người phụ nữ mặc áo dài tím",
        visual_queries=[
            VisualQuery(language="vi", text="phụ nữ áo dài tím", channel="visual_vi"),
            VisualQuery(language="en", text="woman in purple ao dai", channel="visual_en"),
        ],
    )

    results, timings = search._search_multi_path(plan, top_k=100)

    # Frame 200 appeared in both VI and EN, must appear exactly once in merged results
    frame_ids = [r["source_frame_index_zero_based"] for r in results]
    assert len(frame_ids) == len(set(frame_ids))
    assert 200 in frame_ids
    assert 100 in frame_ids
    assert 300 in frame_ids

    # Check candidate provenance
    f200_row = next(r for r in results if r["source_frame_index_zero_based"] == 200)
    assert len(f200_row["matched_by"]) == 2
    assert "visual_vi_0" in f200_row["matched_by"]
    assert "visual_en_1" in f200_row["matched_by"]
    assert timings["candidate_union_count"] == 3


# =========================================================================
# Test H: OCR and Visual Multi-Path Fusion
# =========================================================================

def test_ocr_and_visual_fusion():
    search = ConfiguredSearch()
    search._encoder = DummyEncoder()
    search._bundle = MagicMock()
    search._bundle.index.search = lambda vec, top_k: [
        {"frame_id": 100, "score": 0.9},
        {"frame_id": 200, "score": 0.8},
    ]
    search._bundle.resolver = DummyResolver({})

    # OCR contains evidence for frame 200 and frame 400
    search._ocr = [
        OCRRecord(
            video_id="L22_V001",
            frame_uid="L22_V001:000000200",
            source_frame_index_zero_based=200,
            timestamp_seconds=8.0,
            raw_text="BIỂN SỐ 79H-6072",
            normalized_text="bien so 79h 6072",
            boxes=[],
            confidence=0.95,
        ),
        OCRRecord(
            video_id="L22_V001",
            frame_uid="L22_V001:000000400",
            source_frame_index_zero_based=400,
            timestamp_seconds=16.0,
            raw_text="79H-6072",
            normalized_text="79h 6072",
            boxes=[],
            confidence=0.98,
        ),
    ]
    search._asr = []
    search.enable_ocr = True

    plan = QueryPlan(
        task_type="kis",
        original_query="xe lam biển số 79H-6072",
        visual_queries=[VisualQuery(language="vi", text="xe lam", channel="visual_vi")],
        exact_strings=["79H-6072"],
        lexical_terms=["79H-6072"],
    )

    results, timings = search._search_multi_path(plan, top_k=100)
    frame_ids = [r["source_frame_index_zero_based"] for r in results]
    assert len(frame_ids) == len(set(frame_ids))
    assert 200 in frame_ids
    assert 400 in frame_ids

    # Frame 200 matched both visual and OCR
    f200_row = next(r for r in results if r["source_frame_index_zero_based"] == 200)
    assert "ocr" in f200_row["matched_by"]
    assert "visual_vi_0" in f200_row["matched_by"]


# =========================================================================
# Test K: TRAKE DP Preservation
# =========================================================================

def test_trake_dp_preservation():
    search = ConfiguredSearch()
    search._encoder = DummyEncoder()
    search.enable_query_refine = True

    c1 = [EventCandidate("L22_V001", 100, 0.9), EventCandidate("L22_V001", 500, 0.8)]
    c2 = [EventCandidate("L22_V001", 300, 0.9), EventCandidate("L22_V001", 200, 0.85)]
    c3 = [EventCandidate("L22_V001", 450, 0.95), EventCandidate("L22_V001", 150, 0.7)]

    # Dynamic Programming Aligner must select monotonic sequence 100 -> 300 -> 450
    result = TRAKEAligner().align([c1, c2, c3])
    assert result is not None
    assert result.video_id == "L22_V001"
    assert result.frame_ids == [100, 300, 450]
    assert result.frame_ids[0] < result.frame_ids[1] < result.frame_ids[2]


# =========================================================================
# Test M: KIS Fallback
# =========================================================================

def test_kis_fallback_when_query_refine_disabled():
    search = ConfiguredSearch()
    search._initialize = MagicMock()
    search._search_single_query = MagicMock(return_value=[{"frame_id": 123, "score": 1.0}])

    # With query_refine=False, must call _search_single_query
    res = search.search("test query", top_k=10, query_refine=False)
    assert search._search_single_query.called
    assert res[0]["frame_id"] == 123


# =========================================================================
# Test O: Image Search Regression
# =========================================================================

def test_image_search_does_not_invoke_query_refiner(tmp_path):
    search = ConfiguredSearch()
    search._initialize = MagicMock()
    search._encoder = DummyEncoder()
    search._bundle = MagicMock()
    search._bundle.index.search = lambda vec, top_k: [{"frame_id": 50, "score": 0.99}]
    search._bundle.resolver = DummyResolver({})

    img_file = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color="blue").save(img_file)

    with patch.object(search, "_get_query_refiner") as mock_refiner:
        res = search.search_image(img_file, top_k=10)
        assert not mock_refiner.called
        assert len(res) == 1
        assert res[0]["source_frame_index_zero_based"] == 50


# =========================================================================
# Test CLI query-plan and options
# =========================================================================

def test_cli_query_plan():
    from projectctl import parser
    args = parser().parse_args(["query-plan", "--task", "kis", "xe lam trắng biển số 79H-6072"])
    assert args.task == "kis"
    assert "79H-6072" in args.query

    args_no_refine = parser().parse_args(["search", "xe lam", "--no-query-refine"])
    assert args_no_refine.no_query_refine is True


# =========================================================================
# Test API debug_query_plan
# =========================================================================

def test_api_debug_query_plan():
    from fastapi.testclient import TestClient
    from backend.app.main import create_app

    configured_search = MagicMock(spec=ConfiguredSearch)
    configured_search.configured = True
    configured_search.processed_root = Path("data/processed/videos")
    configured_search.status.return_value = {"status": "ok"}
    configured_search.readiness.return_value = {"ready": True}
    configured_search.last_query_plan = QueryPlan(
        task_type="kis",
        original_query="người đi bộ",
        visual_queries=[VisualQuery(language="vi", text="người đi bộ", channel="visual_vi")],
    )
    configured_search.last_query_metrics = {"total_query_ms": 12.5}
    configured_search.handle.return_value = [{"video_id": "L22_V001", "frame_id": 100, "score": 0.9}]

    app = create_app(configured_search=configured_search)
    client = TestClient(app)

    # 1. Default request (debug_query_plan=False)
    resp = client.post("/api/search", json={"query": "người đi bộ", "debug_query_plan": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "query_plan" not in data  # default contract preserved

    # 2. Debug request (debug_query_plan=True)
    resp_debug = client.post("/api/search", json={"query": "người đi bộ", "debug_query_plan": True})
    assert resp_debug.status_code == 200
    data_debug = resp_debug.json()
    assert "results" in data_debug
    assert "query_plan" in data_debug
    assert data_debug["query_plan"]["original_query"] == "người đi bộ"
    assert "query_metrics" in data_debug


# =========================================================================
# Test QueryPlanCache Corruption Resilience
# =========================================================================

def test_query_cache_corruption_resilience(tmp_path):
    from backend.app.services.query_refiner import QueryPlanCache

    cache = QueryPlanCache(cache_dir=tmp_path)
    corrupt_file = tmp_path / "corrupt_key.json"
    corrupt_file.write_text("{invalid json corrupt content!@#$")

    # get on corrupt file must return None and not raise exception
    res = cache.get("corrupt_key")
    assert res is None


# =========================================================================
# Hardening Pass Tests (Gates F, G, H, M, N)
# =========================================================================

def test_synthetic_ocr_and_asr_routing_and_provenance():
    search = ConfiguredSearch()
    search._encoder = DummyEncoder()
    search._bundle = MagicMock()
    search._bundle.index.search = lambda vec, top_k: [{"frame_id": 100, "score": 0.8}]
    search._bundle.resolver = DummyResolver({})

    # Synthetic OCR evidence for frame 250
    search._ocr = [
        OCRRecord(
            video_id="L22_V001",
            frame_uid="L22_V001:000000250",
            source_frame_index_zero_based=250,
            timestamp_seconds=10.0,
            raw_text="XE CẨU TADANO ZE300",
            normalized_text="xe cau tadano ze300",
            boxes=[],
            confidence=0.99,
        )
    ]
    # Synthetic ASR evidence for frame 500
    search._asr = [
        ASRSegment(
            video_id="L22_V001",
            segment_id="L22_V001:asr:00001",
            start_seconds=20.0,
            end_seconds=24.0,
            start_frame=500,
            end_frame=600,
            raw_text="bão số 3 đang tiến vào biển Đông",
            normalized_text="bao so 3 dang tien vao bien dong",
            confidence=0.95,
            language="vi",
        )
    ]
    search.enable_ocr = True
    search.enable_asr = True

    plan = QueryPlan(
        task_type="kis",
        original_query="cần cẩu TADANO ZE300 trong bản tin bão số 3",
        visual_queries=[VisualQuery(language="vi", text="cần cẩu trong bản tin", channel="visual_vi")],
        exact_strings=["TADANO", "ZE300"],
        kept_vi_terms=["bão số 3"],
        lexical_terms=["TADANO", "ZE300", "bão số 3"],
    )

    results, timings = search._search_multi_path(plan, top_k=50)

    # 1. Verify routing diagnostics
    assert "ocr_routing" in timings
    assert timings["ocr_routing"]["invoked"] is True
    assert "TADANO" in timings["ocr_routing"]["matched_terms"] or "ZE300" in timings["ocr_routing"]["matched_terms"]
    assert timings["ocr_routing"]["candidate_count"] >= 1

    assert "asr_routing" in timings
    assert timings["asr_routing"]["invoked"] is True
    assert "bão số 3" in timings["asr_routing"]["matched_terms"]
    assert timings["asr_routing"]["candidate_count"] >= 1

    # 2. Verify candidate inclusion and channel provenance
    frame_ids = [r["source_frame_index_zero_based"] for r in results]
    assert 100 in frame_ids
    assert 250 in frame_ids
    assert 500 in frame_ids

    f250_row = next(r for r in results if r["source_frame_index_zero_based"] == 250)
    assert "ocr" in f250_row["matched_by"]

    f500_row = next(r for r in results if r["source_frame_index_zero_based"] == 500)
    assert "asr" in f500_row["matched_by"]


def test_trake_gap_diagnostics_preserves_monotonicity():
    search = ConfiguredSearch()
    search._encoder = DummyEncoder()
    search._bundle = MagicMock()
    search._bundle.index.search = lambda vec, top_k: [{"frame_id": 100, "score": 0.8}]
    search._bundle.resolver = DummyResolver({})
    search._ocr = []
    search._asr = []

    # Large gap path: 100 -> 250 -> 15000 (gap of 14750 frames)
    c1 = [EventCandidate("L22_V001", 100, 0.9)]
    c2 = [EventCandidate("L22_V001", 250, 0.85)]
    c3 = [EventCandidate("L22_V001", 15000, 0.95)]

    aligner = TRAKEAligner()
    result = aligner.align([c1, c2, c3])

    assert result is not None
    assert result.video_id == "L22_V001"
    assert result.frame_ids == [100, 250, 15000]

    # Check gap calculation
    frame_gaps = [result.frame_ids[i+1] - result.frame_ids[i] for i in range(len(result.frame_ids) - 1)]
    assert frame_gaps == [150, 14750]
    assert max(frame_gaps) == 14750

