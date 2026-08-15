"""Unit tests for CandidateReranker."""

import pytest
from unittest.mock import MagicMock

from backend.app.services.candidate_reranker import CandidateReranker, classify_exact_match
from backend.app.services.query_refiner import QueryPlan, VisualQuery
from backend.app.video.text_evidence import OCRRecord, ASRSegment


# =========================================================================
# Test A: Visual-only query preserves sane RRF ordering
# =========================================================================

def test_visual_only_preserves_rrf_order():
    reranker = CandidateReranker(enabled=True)
    plan = QueryPlan(
        task_type="kis",
        original_query="người đi bộ trên đường",
        visual_queries=[VisualQuery(language="vi", text="người đi bộ trên đường", channel="visual_vi")],
    )

    candidates = [
        {"video_id": "V1", "source_frame_index_zero_based": 100, "score": 0.050, "matched_by": ["visual_vi_0"]},
        {"video_id": "V1", "source_frame_index_zero_based": 200, "score": 0.045, "matched_by": ["visual_vi_0"]},
        {"video_id": "V2", "source_frame_index_zero_based": 50, "score": 0.040, "matched_by": ["visual_vi_0"]},
    ]

    reranked = reranker.rerank(candidates, plan)
    assert [c["source_frame_index_zero_based"] for c in reranked] == [100, 200, 50]
    assert [c["score"] for c in reranked] == [0.050, 0.045, 0.040]


# =========================================================================
# Test B: Exact OCR match receives correct evidence distinction
# =========================================================================

def test_exact_ocr_match_priority():
    reranker = CandidateReranker(enabled=True)
    plan = QueryPlan(
        task_type="kis",
        original_query="xe lam trắng biển số 79H-6072",
        visual_queries=[VisualQuery(language="vi", text="xe lam trắng", channel="visual_vi")],
        exact_strings=["79H-6072"],
        lexical_terms=["79H-6072"],
    )

    ocr_records = [
        OCRRecord(
            video_id="V1",
            source_frame_index_zero_based=200,
            frame_uid="V1:000000200",
            raw_text="XE LAM 79H-6072 NHA TRANG",
            normalized_text="xe lam 79h-6072 nha trang",
            boxes=[],
            confidence=0.95,
            timestamp_seconds=8.0,
        )
    ]

    candidates = [
        # Candidate 1: High visual RRF score, but no exact string match
        {"video_id": "V1", "source_frame_index_zero_based": 100, "score": 0.050, "matched_by": ["visual_vi_0"]},
        # Candidate 2: Slightly lower visual RRF score, but has verified exact OCR match
        {"video_id": "V1", "source_frame_index_zero_based": 200, "score": 0.035, "matched_by": ["visual_vi_0", "ocr"]},
    ]

    reranked = reranker.rerank(candidates, plan, ocr_evidence=ocr_records)
    # Candidate with verified exact OCR match receives priority
    assert reranked[0]["source_frame_index_zero_based"] == 200
    assert reranked[0]["rerank_metadata"]["exact_tier"] == 3  # full_exact
    assert "79H-6072" in reranked[0]["rerank_metadata"]["matched_exact_terms"]


# =========================================================================
# Test C: Partial identifier does not masquerade as full exact match
# =========================================================================

def test_partial_match_distinct_from_full_exact():
    # 50H-052 vs 50H-052.03
    assert classify_exact_match("50H-052.03", "xe 50H-052 đi qua", "xe 50h-052 di qua") == "partial"
    assert classify_exact_match("50H-052.03", "xe 50H 052.03", "xe 50h 052.03") in ("full_exact", "normalized_separator")
    assert classify_exact_match("50H-052.03", "xe 50H05203", "xe 50h05203") == "normalized_separator"

    reranker = CandidateReranker(enabled=True)
    plan = QueryPlan(
        task_type="kis",
        original_query="xe 50H-052.03",
        visual_queries=[VisualQuery(language="vi", text="xe", channel="visual_vi")],
        exact_strings=["50H-052.03"],
    )

    ocr_records = [
        # Full match
        OCRRecord(
            video_id="V1",
            frame_uid="V1:000000100",
            source_frame_index_zero_based=100,
            timestamp_seconds=4.0,
            raw_text="xe 50H-052.03 đi qua",
            normalized_text="xe 50h-052.03 di qua",
            boxes=[],
            confidence=0.9,
        ),
        # Partial match only (50H-052)
        OCRRecord(
            video_id="V1",
            frame_uid="V1:000000200",
            source_frame_index_zero_based=200,
            timestamp_seconds=8.0,
            raw_text="xe 50H-052 đi qua",
            normalized_text="xe 50h-052 di qua",
            boxes=[],
            confidence=0.9,
        ),
    ]

    candidates = [
        {"video_id": "V1", "source_frame_index_zero_based": 200, "score": 0.040, "matched_by": ["ocr"]},
        {"video_id": "V1", "source_frame_index_zero_based": 100, "score": 0.035, "matched_by": ["ocr"]},
    ]

    reranked = reranker.rerank(candidates, plan, ocr_evidence=ocr_records)
    # Full exact match (100) must rank above partial match (200)
    assert reranked[0]["source_frame_index_zero_based"] == 100
    assert reranked[0]["rerank_metadata"]["exact_tier"] == 3
    assert reranked[1]["rerank_metadata"]["exact_tier"] == 1  # partial


# =========================================================================
# Test D: Multi-channel candidate agreement
# =========================================================================

def test_multi_channel_candidate_priority():
    reranker = CandidateReranker(enabled=True)
    plan = QueryPlan(
        task_type="kis",
        original_query="người sửa xe",
        visual_queries=[VisualQuery(language="vi", text="người sửa xe", channel="visual_vi")],
    )

    candidates = [
        # Candidate 1: 1 channel (visual only), score 0.040
        {"video_id": "V1", "source_frame_index_zero_based": 100, "score": 0.040, "matched_by": ["visual_vi_0"]},
        # Candidate 2: 2 independent channels (visual + ASR), score 0.038
        {"video_id": "V1", "source_frame_index_zero_based": 200, "score": 0.038, "matched_by": ["visual_vi_0", "asr"]},
    ]

    reranked = reranker.rerank(candidates, plan)
    # Candidate 2 with 2 independent channels ranks above 1 channel
    assert reranked[0]["source_frame_index_zero_based"] == 200
    assert reranked[0]["rerank_metadata"]["independent_channels"] == 2


# =========================================================================
# Test E: Reranker disabled -> exact baseline path
# =========================================================================

def test_reranker_disabled_returns_unchanged():
    reranker = CandidateReranker(enabled=False)
    plan = QueryPlan(
        task_type="kis",
        original_query="xe 79H-6072",
        exact_strings=["79H-6072"],
    )

    candidates = [
        {"video_id": "V1", "source_frame_index_zero_based": 100, "score": 0.050, "matched_by": ["visual_vi_0"]},
        {"video_id": "V1", "source_frame_index_zero_based": 200, "score": 0.035, "matched_by": ["ocr"]},
    ]

    reranked = reranker.rerank(candidates, plan)
    assert [c["source_frame_index_zero_based"] for c in reranked] == [100, 200]


# =========================================================================
# Test F: Reranker exception fallback
# =========================================================================

def test_reranker_exception_fallback():
    reranker = CandidateReranker(enabled=True)
    plan = QueryPlan(task_type="kis", original_query="test")

    # Pass malformed candidates causing internal exception
    malformed_candidates = [{"video_id": "V1", "source_frame_index_zero_based": 100, "score": 0.05}]

    # Mock _rerank_internal to raise
    reranker._rerank_internal = MagicMock(side_effect=RuntimeError("unexpected failure"))

    reranked = reranker.rerank(malformed_candidates, plan)
    assert reranked == malformed_candidates


# =========================================================================
# Test G: Score scale independence
# =========================================================================

def test_score_scale_independence():
    reranker = CandidateReranker(enabled=True)
    plan = QueryPlan(task_type="kis", original_query="cảnh biển")

    # Scale 1: scores around 0.01 - 0.05
    c_small = [
        {"video_id": "V1", "source_frame_index_zero_based": 10, "score": 0.03, "matched_by": ["visual_vi_0"]},
        {"video_id": "V1", "source_frame_index_zero_based": 20, "score": 0.02, "matched_by": ["visual_vi_0"]},
    ]
    # Scale 2: scores around 100 - 500
    c_large = [
        {"video_id": "V1", "source_frame_index_zero_based": 10, "score": 300.0, "matched_by": ["visual_vi_0"]},
        {"video_id": "V1", "source_frame_index_zero_based": 20, "score": 200.0, "matched_by": ["visual_vi_0"]},
    ]

    r_small = reranker.rerank(c_small, plan)
    r_large = reranker.rerank(c_large, plan)

    assert [c["source_frame_index_zero_based"] for c in r_small] == [10, 20]
    assert [c["source_frame_index_zero_based"] for c in r_large] == [10, 20]


# =========================================================================
# Test H: No GT-specific strings
# =========================================================================

def test_no_gt_specific_strings():
    import inspect
    from backend.app.services import candidate_reranker
    source = inspect.getsource(candidate_reranker)
    assert "L22_V001" not in source
    assert "L22_V002" not in source
    assert "L22_V003" not in source
