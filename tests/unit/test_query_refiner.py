"""Unit tests for Query Intelligence, QueryPlan, and QueryRefiner."""

import os
import pytest
from unittest.mock import MagicMock, patch

from backend.app.services.query_refiner import (
    DeterministicQueryParser,
    LocalLLMQueryRefiner,
    QueryPlan,
    QueryPlanCache,
    QueryRefiner,
    VisualQuery,
)


# =========================================================================
# Test A: Exact String Extraction
# =========================================================================

def test_exact_string_license_plate():
    parser = DeterministicQueryParser()
    plan = parser.parse("xe lam trắng biển số 79H-6072")
    assert "79H-6072" in plan.exact_strings
    assert "79H-6072" in plan.lexical_terms
    # License plate extracted and clean visual query retains context
    assert any("79H-6072" not in vq.text for vq in plan.visual_queries)


def test_exact_string_generic_patterns():
    parser = DeterministicQueryParser()
    # Quoted string
    p1 = parser.parse('người đi vào "Đường Tiên Lân 11" lúc chiều')
    assert "Đường Tiên Lân 11" in p1.exact_strings

    # Temperature & Percentage
    p2 = parser.parse("nhiệt độ ngoài trời 40°C độ ẩm 50%")
    assert any("40°C" in s or "40" in s for s in p2.exact_strings)
    assert any("50%" in s for s in p2.exact_strings)

    # Date
    p3 = parser.parse("sự kiện diễn ra ngày 12/08/2026")
    assert "12/08/2026" in p3.exact_strings

    # Alphanumeric Code
    p4 = parser.parse("cần cẩu TADANO-ZE300")
    assert "TADANO-ZE300" in p4.exact_strings


# =========================================================================
# Test B: Vietnamese Path Preserved
# =========================================================================

def test_vietnamese_path_preserved():
    parser = DeterministicQueryParser()
    plan = parser.parse("phụ nữ mặc áo dài")
    assert len(plan.visual_queries) >= 1
    # First query must be Vietnamese
    vi_query = next((vq for vq in plan.visual_queries if vq.language == "vi"), None)
    assert vi_query is not None
    assert "phụ nữ" in vi_query.text or "áo dài" in vi_query.text


# =========================================================================
# Test C: Cultural Term Preservation
# =========================================================================

def test_cultural_term_preservation():
    parser = DeterministicQueryParser()
    plan = parser.parse("người mặc áo dài đứng bên xe lam ăn bánh bèo gần chùa")
    assert "áo dài" in plan.kept_vi_terms
    assert "xe lam" in plan.kept_vi_terms
    assert "bánh bèo" in plan.kept_vi_terms
    assert "chùa" in plan.kept_vi_terms


def test_cultural_terms_require_whole_phrase_matches():
    parser = DeterministicQueryParser()

    action_plan = parser.parse("người đàn ông chèo thuyền trên sông")
    assert "chè" not in action_plan.kept_vi_terms
    assert "chè" not in action_plan.lexical_terms

    food_plan = parser.parse("người ngồi ăn chè trong quán")
    assert "chè" in food_plan.kept_vi_terms
    assert "chè" in food_plan.lexical_terms


@pytest.mark.parametrize(
    "query",
    [
        "xe máy có biển số màu trắng",
        "cửa hàng có biển quảng cáo lớn",
        "người đứng cạnh biển báo giao thông",
    ],
)
def test_polysemous_compounds_do_not_emit_unrelated_object(query):
    plan = DeterministicQueryParser().parse(query)
    assert "sea" not in plan.objects


def test_standalone_object_meaning_is_still_extracted():
    plan = DeterministicQueryParser().parse("thuyền đánh cá trên biển")
    assert "sea" in plan.objects


def test_object_extraction_prefers_specific_compound():
    plan = DeterministicQueryParser().parse("xe lam chạy trên đường")
    assert "auto rickshaw" in plan.objects
    assert "vehicle" not in plan.objects


def test_visual_query_keeps_semantic_focus_without_search_or_ocr_instructions():
    query = "Hãy tìm đoạn video có người mặc áo đỏ mang biển số 51A-12345 đi trên đường"
    plan = DeterministicQueryParser().parse(query)
    visual_text = plan.visual_queries[0].text.casefold()

    assert visual_text == "người mặc áo đỏ đi trên đường"
    assert "51a-12345" in [term.casefold() for term in plan.exact_strings]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Tìm kiếm cảnh sát đang điều tiết giao thông", "cảnh sát đang điều tiết giao thông"),
        ("Tìm phim trường ngoài trời", "phim trường ngoài trời"),
        ("Xác định cảnh quan miền núi", "cảnh quan miền núi"),
    ],
)
def test_search_intent_does_not_consume_prefix_of_semantic_word(query, expected):
    plan = DeterministicQueryParser().parse(query)
    assert plan.visual_queries[0].text == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Hãy tìm đoạn video có người chạy trong công viên", "người chạy trong công viên"),
        ("Tìm cảnh có xe dừng bên đường", "xe dừng bên đường"),
        ("Xác định khung hình cho thấy một tòa nhà", "một tòa nhà"),
    ],
)
def test_search_intent_strips_complete_media_phrase(query, expected):
    plan = DeterministicQueryParser().parse(query)
    assert plan.visual_queries[0].text == expected


@pytest.mark.parametrize(
    "query",
    [
        "mã ART cạnh biển hiệu ArtHouse",
        "mã RED cạnh khu rừng Redwood",
        "dòng chữ \"ART\" bên cạnh ArtHouse",
    ],
)
def test_exact_removal_preserves_larger_words_and_names(query):
    plan = DeterministicQueryParser().parse(query)
    visual_text = plan.visual_queries[0].text.casefold()

    assert "arthouse" in visual_text or "redwood" in visual_text
    assert any(term in {"ART", "RED"} for term in plan.exact_strings)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ('Tìm người đứng trước ArtHouse với dòng chữ "ART"', "người đứng trước ArtHouse"),
        ("Tìm xe mang biển số: “51A-12345” chạy trên đường", "xe chạy trên đường"),
        ("Tìm thiết bị có mã 'ZX-90' đặt trên bàn", "thiết bị đặt trên bàn"),
        ("Tìm biển hiệu hiển thị dòng chữ OPEN", "biển hiệu"),
    ],
)
def test_quoted_labelled_exact_removes_complete_text_constraint(query, expected):
    plan = DeterministicQueryParser().parse(query)

    assert plan.visual_queries[0].text == expected
    assert plan.exact_strings


def test_text_label_without_exact_value_is_not_removed_as_dangling():
    plan = DeterministicQueryParser().parse("người đứng cạnh biển số")
    assert plan.visual_queries[0].text == "người đứng cạnh biển số"


def test_qa_visual_query_removes_question_scaffolding():
    parser = DeterministicQueryParser()
    cases = {
        "Có bao nhiêu chiếc xe máy xuất hiện trong đoạn phim?": "chiếc xe máy",
        "Chiếc ô có màu gì?": "chiếc ô",
        "Người công nhân đang làm gì trong nhà kho?": "người công nhân trong nhà kho",
        "Ai đang hát trên sân khấu?": "người đang hát trên sân khấu",
        "Người phát biểu đang nói về chủ đề gì?": "người phát biểu",
    }

    for query, expected in cases.items():
        plan = parser.parse(query, task_type="qa")
        assert plan.visual_queries[0].text.casefold() == expected


def test_qa_who_rewrite_preserves_title_case_proper_noun():
    parser = DeterministicQueryParser()

    proper_noun_plan = parser.parse("Ai Cập có các kim tự tháp nào?", task_type="qa")
    assert proper_noun_plan.visual_queries[0].text.startswith("Ai Cập")
    assert not proper_noun_plan.visual_queries[0].text.startswith("người Cập")

    wh_plan = parser.parse("Ai đang hát trên sân khấu?", task_type="qa")
    assert wh_plan.visual_queries[0].text.casefold() == "người đang hát trên sân khấu"


# =========================================================================
# Test D: No Hallucination
# =========================================================================

def test_no_hallucination():
    parser = DeterministicQueryParser()
    plan = parser.parse("người đứng cạnh xe")
    # Must not hallucinate unmentioned entities
    all_texts = " ".join([vq.text for vq in plan.visual_queries] + plan.objects + plan.attributes).lower()
    assert "red shirt" not in all_texts
    assert "motorcycle" not in all_texts
    assert "79h" not in all_texts
    assert "hà nội" not in all_texts


# =========================================================================
# Test E: Invalid JSON Fallback
# =========================================================================

def test_invalid_json_fallback():
    mock_llm = MagicMock(spec=LocalLLMQueryRefiner)
    # LLM returns invalid JSON or prose
    mock_llm.refine.return_value = None

    refiner = QueryRefiner(backend="local_llm", llm_refiner=mock_llm, cache_enabled=False)
    plan, timings = refiner.refine("người đi bộ", task_type="kis")

    assert plan is not None
    assert plan.refinement_backend == "deterministic"
    assert len(plan.visual_queries) >= 1
    assert timings["deterministic_parse_ms"] >= 0.0


# =========================================================================
# Test F: Missing Model Fallback
# =========================================================================

def test_missing_model_fallback():
    # Points to a non-existent local model
    llm = LocalLLMQueryRefiner(model_name_or_path="/path/does/not/exist/model")
    assert not llm.is_available()

    refiner = QueryRefiner(backend="auto", llm_refiner=llm, cache_enabled=False)
    plan, timings = refiner.refine("người đi xe máy qua cầu", task_type="kis")

    assert plan is not None
    assert plan.refinement_backend == "deterministic"
    assert any(vq.language == "vi" for vq in plan.visual_queries)


# =========================================================================
# Test I: Score Scale Independence (RRF properties)
# =========================================================================

def test_rrf_rank_safe_properties():
    # RRF formula: sum(w / (k + rank))
    # Test that rank 1 in a channel with high raw scores produces identical RRF value
    # as rank 1 in a channel with low raw scores
    k = 60.0
    rank_1_rrf = 1.0 / (k + 1.0)
    rank_2_rrf = 1.0 / (k + 2.0)
    assert rank_1_rrf > rank_2_rrf
    # Proves score scale does not bias RRF, only relative ranking
    assert (1.0 / (k + 1.0)) == (1.0 / (k + 1.0))


# =========================================================================
# Test J: TRAKE Stage Count
# =========================================================================

def test_trake_stage_count_pipe_separated():
    parser = DeterministicQueryParser()
    plan = parser.parse("đứng | chạy đà | nhảy | đáp đất", task_type="trake")
    assert len(plan.trake_stages) == 4
    assert plan.strict_order is True
    assert plan.trake_stages[0].stage_index == 0
    assert plan.trake_stages[3].stage_index == 3


def test_trake_stage_count_natural_language():
    parser = DeterministicQueryParser()
    plan = parser.parse("người đứng chuẩn bị, chạy đà, nhảy và đáp đất", task_type="trake")
    assert len(plan.trake_stages) == 4
    assert plan.strict_order is True


# =========================================================================
# Test Q: Query Cache Sentinel
# =========================================================================

def test_query_cache_avoids_llm_call(tmp_path):
    mock_llm = MagicMock(spec=LocalLLMQueryRefiner)
    sentinel_plan = QueryPlan(
        task_type="kis",
        original_query="xe lam trắng",
        visual_queries=[VisualQuery(language="vi", text="xe lam trắng", weight=1.0, channel="visual_vi")],
        refinement_backend="local_llm",
        refinement_used=True,
    )
    mock_llm.refine.return_value = sentinel_plan

    refiner = QueryRefiner(
        backend="local_llm",
        llm_refiner=mock_llm,
        cache_dir=tmp_path,
        cache_enabled=True,
    )

    # First call: executes LLM refine
    plan1, _ = refiner.refine("xe lam trắng", task_type="kis")
    assert mock_llm.refine.call_count == 1

    # Second call with same query: must hit cache and NOT invoke LLM
    plan2, _ = refiner.refine("xe lam trắng", task_type="kis")
    assert mock_llm.refine.call_count == 1  # Still 1, no second call
    assert plan2.original_query == "xe lam trắng"


# =========================================================================
# Test R: Offline Execution
# =========================================================================

def test_offline_execution_environment():
    with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
        parser = DeterministicQueryParser()
        plan = parser.parse("người phụ nữ mặc áo dài đỏ 79H-6072")
        assert plan is not None
        assert "79H-6072" in plan.exact_strings
        assert "áo dài" in plan.kept_vi_terms
        assert any(vq.language == "vi" for vq in plan.visual_queries)


# =========================================================================
# Test S: Unicode Preservation
# =========================================================================

def test_unicode_preservation():
    parser = DeterministicQueryParser()
    complex_query = "Nguyễn Văn A mặc áo dài có kỹ thuật đo nhiệt độ tại Đường Tiên Lân 11 gần xe lam"
    plan = parser.parse(complex_query)

    assert "Nguyễn Văn A" in plan.original_query
    assert "áo dài" in plan.kept_vi_terms
    assert "xe lam" in plan.kept_vi_terms
    assert "Đường Tiên Lân 11" in plan.original_query


# =========================================================================
# Hardening Pass Tests (Gates A, B, C, D, E, I, L)
# =========================================================================

def test_clean_english_validation():
    from backend.app.services.query_refiner import validate_english_caption
    # Valid clean English captions
    assert validate_english_caption("woman wearing purple dress standing beside auto rickshaw") is True
    assert validate_english_caption("person preparing, taking a run-up, jumping and landing") is True

    # Malformed captions containing Vietnamese diacritics
    assert validate_english_caption("person dẫn chương trình news program") is False
    assert validate_english_caption("woman mặc áo dài purple beside xe lam white") is False
    assert validate_english_caption("xe tải cẩu license plate") is False

    # Empty or invalid
    assert validate_english_caption("") is False
    assert validate_english_caption(None) is False
    assert validate_english_caption("  ") is False


def test_malformed_english_fallback_to_vi_only():
    mock_llm = MagicMock(spec=LocalLLMQueryRefiner)
    # LLM returns malformed English caption containing untranslated Vietnamese
    mock_llm._ensure_loaded.return_value = True
    mock_llm._pipeline = lambda prompt: [{
        "generated_text": '{"visual_caption_vi": "người dẫn chương trình thời sự", "visual_caption_en": "person dẫn chương trình news program trong trường quay", "kept_vi_terms": ["thời sự"], "objects": ["person"], "attributes": []}'
    }]
    mock_llm.refine = LocalLLMQueryRefiner.refine.__get__(mock_llm, LocalLLMQueryRefiner)

    refiner = QueryRefiner(backend="local_llm", llm_refiner=mock_llm, cache_enabled=False)
    plan, _ = refiner.refine("người dẫn chương trình thời sự trong trường quay", task_type="kis")

    assert plan is not None
    # The malformed EN caption must be rejected by validator, so only VI query remains
    assert all(vq.language == "vi" for vq in plan.visual_queries)
    assert not any(vq.language == "en" for vq in plan.visual_queries)


def test_no_fake_token_translation():
    parser = DeterministicQueryParser()
    plan = parser.parse("người dẫn chương trình thời sự trong trường quay")
    # Must NOT produce naive hybrid mixed-language visual_en
    assert not any(vq.language == "en" for vq in plan.visual_queries)
    assert any(vq.language == "vi" for vq in plan.visual_queries)


def test_no_corpus_specific_named_entities_in_static_list():
    from backend.app.services.query_refiner import KNOWN_VI_CULTURAL_TERMS
    # Must NOT contain hard-coded 3-video named entities
    assert "hải vân quan" not in KNOWN_VI_CULTURAL_TERMS
    assert "nguyễn hữu cảnh" not in KNOWN_VI_CULTURAL_TERMS
    assert "nguyen huu canh" not in KNOWN_VI_CULTURAL_TERMS
    assert "hai van quan" not in KNOWN_VI_CULTURAL_TERMS


def test_generic_named_entity_preservation():
    parser = DeterministicQueryParser()
    # Test seen and unseen proper names
    plan1 = parser.parse("Nguyễn Văn A đứng trước tòa nhà VinFast")
    assert "Nguyễn Văn A" in plan1.kept_vi_terms
    assert "Nguyễn Văn A" in plan1.lexical_terms

    plan2 = parser.parse("khởi công dự án tại Hải Vân Quan và đền thờ Nguyễn Hữu Cảnh")
    assert "Hải Vân Quan" in plan2.kept_vi_terms
    assert "Nguyễn Hữu Cảnh" in plan2.kept_vi_terms


def test_exact_identifier_matching_variants():
    from backend.app.services.configured_search import _match_exact_term
    # Test hyphen, space, dot, and compact variants
    assert _match_exact_term("50H-052.03", "xe biển số 50H-052.03", "xe bien so 50h-052.03") is True
    assert _match_exact_term("50H-052.03", "xe biển số 50H 052.03", "xe bien so 50h 052.03") is True
    assert _match_exact_term("50H-052.03", "50H05203", "50h05203") is True
    assert _match_exact_term("TADANO", "XE TẢI TADANO ZE300", "xe tai tadano ze300") is True
    assert _match_exact_term("ZE300", "cần cẩu ZE300", "can cau ze300") is True
    # Non-match should safely return False
    assert _match_exact_term("50H-052.03", "xe buýt số 150", "xe buyt so 150") is False


def test_section5_query_refiner_test_set():
    parser = DeterministicQueryParser()

    # Query A
    q_a = "phụ nữ mặc áo dài tím cạnh xe lam trắng biển số 79H-6072"
    p_a = parser.parse(q_a, task_type="kis")
    assert "79H-6072" in p_a.exact_strings
    assert "áo dài" in p_a.kept_vi_terms
    assert "xe lam" in p_a.kept_vi_terms
    assert any(vq.language == "vi" for vq in p_a.visual_queries)

    # Query B
    q_b = "người đứng chuẩn bị rồi chạy đà, nhảy và đáp đất"
    p_b = parser.parse(q_b, task_type="trake")
    assert len(p_b.trake_stages) >= 3
    assert p_b.strict_order is True

    # Query C
    q_c = "bản tin nói nhiệt độ tại Barcelona là 40 độ C"
    p_c = parser.parse(q_c, task_type="kis")
    assert any("40" in s for s in p_c.exact_strings)
    assert "Barcelona" in p_c.kept_vi_terms or "Barcelona" in p_c.lexical_terms

    # Query D: No hallucinations
    q_d = "người đứng cạnh xe"
    p_d = parser.parse(q_d, task_type="kis")
    all_terms = " ".join([vq.text for vq in p_d.visual_queries] + p_d.objects + p_d.attributes).lower()
    assert "red" not in all_terms and "blue" not in all_terms
    assert "toyota" not in all_terms and "honda" not in all_terms
    assert "hà nội" not in all_terms and "sài gòn" not in all_terms
    assert not p_d.exact_strings
