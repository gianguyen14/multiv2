from unittest.mock import MagicMock

import pytest

from backend.app.services.configured_search import ConfiguredSearch
from backend.app.services.query_refiner import QueryPlan, VisualQuery
from backend.app.video.text_evidence import ASRSegment, OCRRecord


class _Encoder:
    def __init__(self):
        self.calls = 0

    def encode_text(self, texts):
        self.calls += 1
        return [[0.1, 0.2]]


class _Resolver:
    def resolve(self, frame_id):
        return {
            "video_id": "video",
            "frame_id": frame_id,
            "source_frame_index_zero_based": frame_id,
            "submission_frame_id": frame_id,
            "frame_uid": f"video:{frame_id:09d}",
            "timestamp_seconds": float(frame_id),
            "image_path": f"frames/{frame_id:09d}.jpg",
        }


def _search(hits):
    search = ConfiguredSearch()
    search._encoder = _Encoder()
    search._bundle = MagicMock()
    search._bundle.index.search.return_value = hits
    search._bundle.resolver = _Resolver()
    search._ocr = []
    search._asr = []
    search.enable_ocr = True
    search.enable_asr = True
    return search


def _plan(*, lexical_terms=None):
    return QueryPlan(
        original_query="target",
        visual_queries=[VisualQuery(language="en", text="target", channel="visual_en")],
        lexical_terms=lexical_terms or [],
    )


def _ocr(frame_id):
    return OCRRecord(
        video_id="video",
        frame_uid=f"video:{frame_id:09d}",
        source_frame_index_zero_based=frame_id,
        timestamp_seconds=float(frame_id),
        raw_text="target",
        normalized_text="target",
        boxes=[],
        confidence=1.0,
    )


def _asr(frame_id):
    return ASRSegment.create(
        video_id="video",
        index=frame_id,
        start_seconds=float(frame_id),
        end_seconds=float(frame_id + 1),
        start_frame=frame_id,
        end_frame=frame_id,
        raw_text="target",
    )


def test_weighted_rrf_is_sorted_by_fused_rank(monkeypatch):
    monkeypatch.setenv("VISUAL_WEIGHT", "1")
    monkeypatch.setenv("OCR_WEIGHT", "1")
    monkeypatch.setenv("ASR_WEIGHT", "0")
    search = _search([
        {"frame_id": 10, "score": 0.9},
        {"frame_id": 20, "score": 0.8},
    ])
    search._ocr = [_ocr(20)]

    results, _ = search._search_multi_path(
        _plan(lexical_terms=["target"]), rerank=False
    )

    assert [row["source_frame_index_zero_based"] for row in results[:2]] == [20, 10]
    assert results[0]["score"] > results[1]["score"]


def test_modality_weights_are_isolated_and_zero_disables_work(monkeypatch):
    search = _search([{"frame_id": 10, "score": 0.9}])
    search._ocr = [_ocr(20)]
    search._asr = [_asr(30)]
    plan = _plan(lexical_terms=["target"])

    monkeypatch.setenv("VISUAL_WEIGHT", "0")
    monkeypatch.setenv("OCR_WEIGHT", "2")
    monkeypatch.setenv("ASR_WEIGHT", "0")
    ocr_results, ocr_metrics = search._search_multi_path(plan, rerank=False)
    assert [row["source_frame_index_zero_based"] for row in ocr_results] == [20]
    assert ocr_results[0]["matched_by"] == ["ocr"]
    assert ocr_metrics["ocr_routing"]["invoked"] is True
    assert ocr_metrics["asr_routing"]["invoked"] is False
    assert search._encoder.calls == 0

    monkeypatch.setenv("OCR_WEIGHT", "0")
    monkeypatch.setenv("ASR_WEIGHT", "3")
    asr_results, asr_metrics = search._search_multi_path(plan, rerank=False)
    assert [row["source_frame_index_zero_based"] for row in asr_results] == [30]
    assert asr_results[0]["matched_by"] == ["asr"]
    assert asr_metrics["ocr_routing"]["invoked"] is False
    assert asr_metrics["asr_routing"]["invoked"] is True
    assert search._encoder.calls == 0


def test_default_rrf_weights_preserve_existing_asr_weight(monkeypatch):
    for name in ("VISUAL_WEIGHT", "OCR_WEIGHT", "ASR_WEIGHT"):
        monkeypatch.delenv(name, raising=False)
    search = _search([{"frame_id": 10, "score": 0.9}])
    search._asr = [_asr(30)]

    results, _ = search._search_multi_path(
        _plan(lexical_terms=["target"]), rerank=False
    )
    by_frame = {row["source_frame_index_zero_based"]: row for row in results}

    assert by_frame[10]["score"] == pytest.approx(1.0 / 61.0)
    assert by_frame[30]["score"] == pytest.approx(0.8 / 61.0)
    assert results[0]["source_frame_index_zero_based"] == 10


@pytest.mark.parametrize("env_name", ["VISUAL_WEIGHT", "OCR_WEIGHT", "ASR_WEIGHT"])
@pytest.mark.parametrize("invalid_value", ["-0.1", "nan", "inf", "not-a-number"])
def test_invalid_modality_weight_is_rejected(monkeypatch, env_name, invalid_value):
    monkeypatch.setenv(env_name, invalid_value)
    search = _search([])

    with pytest.raises(ValueError, match=env_name):
        search._search_multi_path(_plan(), rerank=False)


@pytest.mark.parametrize("invalid_weight", [-1.0, float("nan"), float("inf")])
def test_invalid_visual_variant_weight_is_rejected(monkeypatch, invalid_weight):
    monkeypatch.setenv("VISUAL_WEIGHT", "1")
    search = _search([])
    plan = QueryPlan(
        original_query="target",
        visual_queries=[
            VisualQuery(
                language="en",
                text="target",
                channel="visual_en",
                weight=invalid_weight,
            )
        ],
    )

    with pytest.raises(ValueError, match="visual query weight"):
        search._search_multi_path(plan, rerank=False)
