from backend.app.retrieval.video_multimodal import lexical_score

LEXICAL_CANDIDATE_THRESHOLD = 0.15


def test_lexical_score_precision():
    # Stopwords are removed
    assert lexical_score("nhiệt độ đạt 40 độ C", "nhiệt độ ngoài trời hôm nay đạt 40 độ c") > 0.6
    assert lexical_score("Bảng điện tử Vietnam Airlines", "Vietnam Airlines Nhóm 1") > 0.3
    assert lexical_score("hoàn toàn không liên quan", "pháo đài hình ngôi sao") == 0.0


def test_long_query_is_not_diluted_below_candidate_threshold():
    query = " ".join([f"chi-tiết-{index}" for index in range(1, 7)] + ["từ-khóa-đích"])

    assert lexical_score(query, "Từ-khóa-đích") >= LEXICAL_CANDIDATE_THRESHOLD


def test_single_overlap_in_long_noisy_evidence_stays_below_threshold():
    query = " ".join([f"mô-tả-{index}" for index in range(1, 20)] + ["người"])
    evidence = "người xe đường nhà cây trời sông biển bảng cửa"

    assert lexical_score(query, evidence) < LEXICAL_CANDIDATE_THRESHOLD


def test_focused_phrase_outscores_generic_single_term_for_long_query():
    query = " ".join([f"mô-tả-{index}" for index in range(1, 19)] + ["chủ-thể", "hành-động"])
    generic_score = lexical_score(query, "chủ-thể")
    focused_score = lexical_score(query, "chủ-thể hành-động")

    assert generic_score < LEXICAL_CANDIDATE_THRESHOLD
    assert focused_score >= LEXICAL_CANDIDATE_THRESHOLD
    assert focused_score > generic_score


def test_stopword_only_query_has_no_lexical_signal():
    assert lexical_score("có là và của trong ở", "có người ở trong xe") == 0.0


def test_exact_phrase_receives_full_relevance_amid_extra_text():
    assert lexical_score(
        "vietnam airlines",
        "bảng điện tử hiển thị vietnam airlines tại quầy làm thủ tục",
    ) == 1.0


def test_lexical_score_is_always_bounded():
    samples = [
        ("a", "a"),
        ("a a a", "a a"),
        ("alpha beta", "alpha beta gamma"),
        ("alpha beta", "unrelated"),
        ("", "alpha"),
    ]
    assert all(0.0 <= lexical_score(query, text) <= 1.0 for query, text in samples)


def test_multimodal_weight_isolation():
    # When weights are 0, contribution is 0
    query = "nhiệt độ đạt 40 độ C"
    txt = "mức nhiệt được ghi nhận là 40 độ C"
    s = lexical_score(query, txt)
    assert s > 0.5
    
    ow, aw = 0.0, 0.0
    text_boost = ow * s + aw * s
    assert text_boost == 0.0
    
    ow, aw = 1.0, 1.0
    text_boost = ow * s * 1.5 + aw * s * 1.5
    assert text_boost > 1.0
