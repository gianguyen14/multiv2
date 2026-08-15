"""
test_qa_generalized_validation.py — Unit tests for generalized answer-type taxonomy and unseen synthetic generalization.
"""

from types import SimpleNamespace

import pytest
from backend.app.retrieval.video_qa import (
    ExtractiveAnswerer,
    _classify_question,
    _evidence_support,
    _is_valid_answer,
)
from backend.app.services.configured_search import ConfiguredSearch


def test_classify_question():
    assert _classify_question("Nhiệt độ đạt bao nhiêu độ C?") == "HOW_MANY"
    assert _classify_question("Cháy rừng xảy ra ở đâu?") == "WHERE"
    assert _classify_question("Ai là đạo diễn của bộ phim?") == "WHO"
    assert _classify_question("Hội nghị diễn ra vào ngày nào?") == "WHEN"
    assert _classify_question("Bệnh nhân mắc bệnh gì?") == "DISEASE"
    assert _classify_question("Mũ bảo hiểm có màu gì?") == "COLOR"
    assert _classify_question("Menu hiển thị chữ gì?") == "WHAT_TEXT"
    assert _classify_question("Thủ tướng phát biểu gì?") == "WHAT_SAID"


def test_unseen_synthetic_who_adversarial():
    """Ensure non-person structural head nouns are rejected generically without literal blacklist words."""
    adversarial_who = [
        "trường đại học Bách Khoa",
        "sở y tế TP.HCM",
        "ban tổ chức Festival",
        "câu lạc bộ Bóng đá",
        "nhà máy Thủy điện",
        "viện nghiên cứu Biển Đông",
        "Lễ hội Văn hóa",
        "Ngày hội Việc làm",
        "Lần đầu Tiên",
        "ành âm nhạc",
        "dự án Metro số 1",
        "công ty Cổ phần",
        "hội đồng Quản trị",
    ]
    for non_person in adversarial_who:
        assert not _is_valid_answer(non_person, "WHO"), f"Failed to reject non-person: {non_person}"


def test_valid_who_names():
    """Ensure genuine person names with or without honorifics are accepted."""
    valid_names = [
        "Trần Đăng Khoa",
        "Bùi Xuân Cường",
        "Nguyễn Hữu Cảnh",
        "Nicola Benedetti",
        "James Cook",
        "Võ Văn Thưởng",
        "Phạm Minh Chính",
    ]
    for name in valid_names:
        assert _is_valid_answer(name, "WHO"), f"Failed to accept valid person: {name}"


def test_unseen_synthetic_where_adversarial():
    """Ensure non-location structural head nouns and time phrases are rejected generically."""
    adversarial_where = [
        "trường đại học Kinh tế",
        "sở y tế Cần Thơ",
        "ban tổ chức Hội nghị",
        "câu lạc bộ Thể thao",
        "nhà máy Sản xuất",
        "viện nghiên cứu Nông nghiệp",
        "ngày 15 tháng 8",
        "lúc 18 giờ 30",
        "bệnh viện Đa khoa",
        "dự án Đường cao tốc",
        "kế hoạch Phát triển",
    ]
    for non_loc in adversarial_where:
        assert not _is_valid_answer(non_loc, "WHERE"), f"Failed to reject non-location: {non_loc}"


def test_valid_where_locations():
    """Ensure genuine locations and geographic markers are accepted."""
    valid_locations = [
        "Bolivia",
        "Hàn Quốc",
        "TP.HCM",
        "thành phố Đà Lạt",
        "tỉnh Đắk Lắk",
        "vùng Biển Đông",
        "Chicago",
        "quận 1",
        "huyện Krông Bông",
    ]
    for loc in valid_locations:
        assert _is_valid_answer(loc, "WHERE"), f"Failed to accept valid location: {loc}"


def test_how_many_numeric_validation():
    assert _is_valid_answer("40 độ C", "HOW_MANY")
    assert _is_valid_answer("143 bác sĩ", "HOW_MANY")
    assert _is_valid_answer("1.5 triệu", "HOW_MANY")
    assert not _is_valid_answer("rất nhiều", "HOW_MANY")
    assert not _is_valid_answer("nhiệt độ ngoài trời", "HOW_MANY")


def test_directional_attribute_gating():
    # Question asks about price drop (giảm), evidence is general price/spread without drop
    q = "Giá vàng miếng SJC hôm nay giảm bao nhiêu triệu đồng?"
    t_mismatch = "giới khoảng 1,5 triệu đồng/lượng. SJC sẽ tiếp tục thu mua vàng miếng"
    assert _evidence_support(q, t_mismatch) == 0.0

    t_match = "giá vàng miếng SJC giảm 1,5 triệu đồng trong phiên giao dịch"
    assert _evidence_support(q, t_match) > 0.4


def test_proper_noun_constraint_gating():
    q = "Vụ tai nạn giao thông ở Đắk Lắk khiến bao nhiêu người bị thương?"
    t_other_province = "Lào Cai: Khởi tố vụ đứt cáp cần cẩu xây dựng làm 6 công nhân thương vong"
    assert _evidence_support(q, t_other_province) == 0.0

    t_correct = "tai nạn giao thông tại Đắk Lắk khiến 3 người bị thương"
    assert _evidence_support(q, t_correct) > 0.4


def test_extractive_answerer_abstains_on_unsupported_adversarial():
    answerer = ExtractiveAnswerer()
    evidence = [
        {"id": "ev1", "text": "Quy tụ những tên tuổi trong ngành âm nhạc và nghệ thuật biểu diễn."},
        {"id": "ev2", "text": "Khởi tố vụ án tại Lào Cai làm 6 người thương vong."},
        {"id": "ev3", "text": "SJC tiếp tục thu mua vàng miếng móp méo với khối lượng lớn."},
    ]
    # Unsupported who
    res1 = answerer.answer("Ai là ca sĩ biểu diễn trong lễ hội âm nhạc?", evidence)
    assert res1["answer"] == ""

    # Unsupported where
    res2 = answerer.answer("Tổng thống Mỹ phát biểu tại thành phố nào?", evidence)
    assert res2["answer"] == ""

    # Unsupported how many
    res3 = answerer.answer("Giá vàng miếng SJC hôm nay giảm bao nhiêu triệu đồng?", evidence)
    assert res3["answer"] == ""


def test_extractive_answerer_answers_supported_evidence():
    answerer = ExtractiveAnswerer()
    evidence = [
        {"id": "ev1", "text": "nhiệt độ ngoài trời hôm nay đạt 40 độ C tại nhiều khu vực"},
        {"id": "ev2", "text": "khởi công dự án cải tạo đền thờ Nguyễn Hữu Cảnh"},
        {"id": "ev3", "text": "thiếu niên nghiện smartphone có nguy cơ cao bị trầm cảm"},
    ]
    res1 = answerer.answer("Nhiệt độ đạt bao nhiêu độ C?", evidence)
    assert "40" in res1["answer"]

    res2 = answerer.answer("Dự án cải tạo đền thờ nào đang được khởi công?", evidence)
    assert "Nguyễn Hữu Cảnh" in res2["answer"]

    res3 = answerer.answer("Thiếu niên nghiện smartphone dễ bị bệnh gì?", evidence)
    assert "trầm cảm" in res3["answer"]


def test_configured_qa_does_not_copy_evidence_between_candidates(monkeypatch):
    search = ConfiguredSearch()
    rows = [
        {"video_id": "video-a", "source_frame_index_zero_based": 100, "frame_id": 100},
        {"video_id": "video-b", "source_frame_index_zero_based": 900, "frame_id": 900},
    ]
    monkeypatch.setattr(search, "search", lambda query, top_k: rows)
    search._ocr = [SimpleNamespace(video_id="video-a",
        source_frame_index_zero_based=100, frame_uid="video-a:000000100",
        raw_text="Nhiệt độ đạt 40 độ C")]
    search._asr = []

    results = search.handle({"query_type": "qa",
        "query": "Nhiệt độ đạt bao nhiêu độ C?", "top_k": 2})

    assert results[0]["answer"] == "40 độ C"
    assert results[0]["evidence_sources"] == ["video-a:000000100"]
    assert results[1]["answer"] == ""
    assert results[1]["evidence_sources"] == []
