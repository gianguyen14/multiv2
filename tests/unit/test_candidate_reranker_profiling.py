import copy
import logging
import pytest
from copy import deepcopy

from backend.app.services.candidate_reranker import CandidateReranker
from backend.app.services.query_refiner import QueryPlan, VisualQuery
from backend.app.video.text_evidence import OCRRecord, ASRSegment


def _base_candidates():
    return [
        {
            "video_id": "V1",
            "source_frame_index_zero_based": 100,
            "score": 0.050,
            "matched_by": ["visual_vi_0"],
        },
        {
            "video_id": "V1",
            "source_frame_index_zero_based": 200,
            "score": 0.045,
            "matched_by": ["visual_vi_0", "ocr"],
        },
        {
            "video_id": "V2",
            "source_frame_index_zero_based": 150,
            "score": 0.040,
            "matched_by": ["visual_vi_0", "asr"],
        },
    ]


def _plan():
    return QueryPlan(
        task_type="kis",
        original_query="test query",
        visual_queries=[VisualQuery(language="vi", text="test query", channel="visual_vi")],
        exact_strings=["ABC-123"],
        lexical_terms=["ABC-123"],
    )


def _ocr_evidence():
    return [
        OCRRecord(
            video_id="V1",
            source_frame_index_zero_based=200,
            frame_uid="V1:000000200",
            raw_text="ABC-123",
            normalized_text="abc-123",
            boxes=[],
            confidence=0.95,
            timestamp_seconds=5.0,
        )
    ]


def _asr_evidence():
    return [
        ASRSegment.create(
            video_id="V2",
            index=0,
            start_seconds=2.0,
            end_seconds=2.5,
            start_frame=150,
            end_frame=150,
            raw_text="some speech",
            language="vi",
            confidence=0.9,
        )
    ]


def test_candidate_reranker_profiling(monkeypatch, caplog):
    reranker = CandidateReranker(enabled=True)
    base_candidates = _base_candidates()
    plan = _plan()
    ocr = _ocr_evidence()
    asr = _asr_evidence()

    # Non‑profile run
    monkeypatch.delenv("RERANKER_PROFILE", raising=False)
    caplog.clear()
    result_no_profile = reranker.rerank(
        deepcopy(base_candidates), deepcopy(plan), deepcopy(ocr), deepcopy(asr)
    )
    # Ensure no profiling record
    profile_records = [rec for rec in caplog.records if rec.getMessage() == "candidate_reranker_profile"]
    assert len(profile_records) == 0, "Profiling should be disabled"

    # Profile run
    monkeypatch.setenv("RERANKER_PROFILE", "1")
    caplog.clear()
    result_profile = reranker.rerank(
        deepcopy(base_candidates), deepcopy(plan), deepcopy(ocr), deepcopy(asr)
    )
    # Ensure profiling record exists and has required attributes
    profile_records = [rec for rec in caplog.records if rec.getMessage() == "candidate_reranker_profile"]
    assert len(profile_records) == 1, "Profiling should emit exactly one log record"
    rec = profile_records[0]
    for attr in ["candidate_count", "exact_term_count", "ocr_evidence_count", "asr_evidence_count", "duration_ms"]:
        assert hasattr(rec, attr), f"LogRecord missing expected attribute {attr}"
    assert rec.candidate_count == len(base_candidates)
    assert rec.exact_term_count == len([ex for ex in plan.exact_strings if ex.strip()])
    assert rec.ocr_evidence_count == len(ocr)
    assert rec.asr_evidence_count == len(asr)

    # Results should be identical regardless of profiling
    assert result_no_profile == result_profile
