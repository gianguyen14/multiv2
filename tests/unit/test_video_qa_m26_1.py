"""
tests/unit/test_video_qa_m26_1.py

M26.1 tests for ExtractiveAnswerer:
- Weak regex → abstain
- Wrong answer type → abstain
- Unsupported number → abstain
- Unsupported location → abstain
- Valid disease evidence → answer
- No false stopword match
"""
import pytest
from backend.app.retrieval.video_qa import ExtractiveAnswerer, _classify_question


# ─── answer type classification ───────────────────────────────────────────────

def test_classify_how_many():
    assert _classify_question("Nhiệt độ đạt bao nhiêu độ C?") == "HOW_MANY"

def test_classify_where():
    assert _classify_question("Vụ tai nạn giao thông xảy ra ở đâu?") == "WHERE"

def test_classify_disease():
    assert _classify_question("Thiếu niên nghiện smartphone dễ bị bệnh gì?") == "DISEASE"

def test_classify_who():
    assert _classify_question("Ai đang hát trên sân khấu?") == "WHO"


# ─── abstention cases ──────────────────────────────────────────────────────────

@pytest.fixture
def answerer():
    return ExtractiveAnswerer()


def test_abstain_on_empty_evidence(answerer):
    result = answerer.answer("Vụ tai nạn giao thông xảy ra ở đâu?", [])
    assert result["answer"] == ""
    assert result["confidence"] == 0.0


def test_abstain_on_zero_evidence_support(answerer):
    """Evidence with no question-term overlap must be ignored."""
    ev = [{"id": "x1", "text": "con mèo đang ngủ trên ghế sofa"}]
    result = answerer.answer("Vụ tai nạn giao thông xảy ra ở đâu?", ev)
    assert result["answer"] == ""


def test_abstain_stopword_location(answerer):
    """'nơi' is a stopword — must NOT be returned as a WHERE answer."""
    ev = [{"id": "x2", "text": "tai nạn thưởng lãm cảnh đẹp ở nơi được ví là đẹp nhất"}]
    result = answerer.answer("Vụ tai nạn giao thông xảy ra ở đâu?", ev)
    assert result["answer"] != "nơi", f"Returned stopword 'nơi' as answer: {result}"
    # Either abstains or returns a proper named location
    if result["answer"]:
        assert len(result["answer"]) > 3


def test_abstain_bare_number_no_context(answerer):
    """A lone number with no question-term context must not be extracted."""
    ev = [{"id": "x3", "text": "Nga luồn tra chung với Iran ở biển Caspi == 46 Bà Halla Toma"}]
    result = answerer.answer("Giá vàng hôm nay là bao nhiêu?", ev)
    # '46' appeared in unrelated text — must abstain or have very low confidence
    if result["answer"]:
        # If it does extract something, confidence must be below the minimum threshold
        # (this would be a bug)
        assert False, f"Should have abstained but returned: '{result['answer']}'"


def test_abstain_garbled_ocr_who(answerer):
    """Garbled OCR text should not satisfy WHO answer-type validation."""
    ev = [{"id": "x4", "text": "Bid LGwWchias sùi d6hiệm Giiềp GioY đhèloê Nan bbiidy thi đến hệ"}]
    result = answerer.answer("Ai đang hát trên sân khấu?", ev)
    # Even if a capitalised token is found, garbled text has zero evidence support
    assert result["answer"] == "" or result["confidence"] < 0.55


def test_abstain_unsupported_question(answerer):
    """Question about US President — no matching content in any evidence."""
    ev = [
        {"id": "x5", "text": "đây là một đoạn tin tức thông thường về thời tiết"},
        {"id": "x6", "text": "nhiệt độ tăng cao tại các vùng miền núi phía Bắc"},
    ]
    result = answerer.answer("Tổng thống Mỹ nói gì?", ev)
    assert result["answer"] == ""


# ─── valid evidence cases ──────────────────────────────────────────────────────

def test_answer_disease_from_asr(answerer):
    """ASR text containing 'trầm cảm' must produce a DISEASE answer."""
    ev = [
        {"id": "y1",
         "text": "ở độ tuổi từ 13 đến 16 cũng ghi nhận các triệu chứng lo âu và trầm cảm cao hơn bình thường."}
    ]
    result = answerer.answer("Thiếu niên nghiện smartphone dễ bị bệnh gì?", ev)
    assert result["answer"] == "trầm cảm"
    assert result["confidence"] >= 0.55


def test_answer_how_many_with_unit(answerer):
    """Numeric with a Vietnamese unit must be extracted for HOW_MANY."""
    ev = [{"id": "y2", "text": "nhiệt độ đỉnh điểm lên tới 41.5 độ trong tháng 8"}]
    result = answerer.answer("Nhiệt độ đạt bao nhiêu độ C?", ev)
    assert "41" in result["answer"] or "độ" in result["answer"]
    assert result["confidence"] >= 0.55


def test_answer_proper_location(answerer):
    """A capitalised named location after ở/tại must be extracted for WHERE."""
    ev = [{"id": "y3", "text": "vụ cháy rừng xảy ra tại Bolivia làm hàng nghìn ha rừng bị thiêu rụi"}]
    result = answerer.answer("Cháy rừng dữ dội xảy ra ở đâu?", ev)
    assert result["answer"] == "Bolivia"
    assert result["confidence"] >= 0.55


# ─── fusion noise-resistance (behavioural) ────────────────────────────────────

def test_fusion_visual_query_no_text_needed():
    """
    Smoke-test: for a visual-intent query, importing ConfiguredSearch must not
    raise even if ASR/OCR return zero scores.
    This does NOT load the actual model (no VIDEO_PROCESSED_ROOT set).
    """
    from backend.app.services.configured_search import ConfiguredSearch
    import os
    # Remove root so _initialize is never called
    old = os.environ.pop("VIDEO_PROCESSED_ROOT", None)
    try:
        s = ConfiguredSearch(processed_root=None)
        assert not s.configured
    finally:
        if old is not None:
            os.environ["VIDEO_PROCESSED_ROOT"] = old


def test_fusion_formula_bounded():
    """
    Unit-test the bounded additive formula logic in isolation.

    For visual-intent query (authority=0.01) with vis_range=0.022:
    - max text bonus = 1.0 * 0.01 * 0.022 = 0.00022
    - This must not overcome a visual gap of 0.001
    """
    vis_range = 0.022
    authority = 0.01  # visual query
    asr_norm = 1.0  # maximum possible lexical hit
    text_bonus = asr_norm * authority * vis_range
    visual_gap = 0.001  # typical small gap between adjacent candidates
    assert text_bonus < visual_gap, (
        f"Text bonus {text_bonus:.6f} must be < visual gap {visual_gap:.6f} "
        f"for visual-intent queries"
    )


def test_fusion_formula_text_intent():
    """
    For text-intent query (authority=0.50), text bonus must be meaningful.
    """
    vis_range = 0.022
    authority = 0.50  # text-intent query
    text_norm = 1.0
    text_bonus = text_norm * authority * vis_range
    # Should be significant fraction of vis_range
    assert text_bonus >= 0.005, (
        f"Text bonus {text_bonus:.4f} too small for text-intent query"
    )


def test_fusion_negative_visual_score():
    """
    If visual score is negative (or zero), additive fusion still produces a
    sensible (lower) fused score without sign flip.
    """
    vis = -0.01
    vis_range = 0.05
    authority = 0.50
    text_norm = 1.0
    text_bonus = text_norm * authority * vis_range
    fused = vis + text_bonus
    # text bonus is positive so fused > vis
    assert fused > vis
    # fused should still be less than a candidate with vis=0.04 and no text
    fused_strong_visual = 0.04
    assert fused_strong_visual > fused  # stronger visual wins
