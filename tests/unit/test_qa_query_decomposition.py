from backend.app.retrieval.qa_query_decomposition import QAQueryDecomposer


def test_removes_dangling_vietnamese_copula_without_losing_constraints():
    result = QAQueryDecomposer().decompose(
        "Dòng chữ xuất hiện trên bảng hiệu quảng cáo là gì?"
    )

    assert result["retrieval_query"] == "Dòng chữ xuất hiện trên bảng hiệu quảng cáo"


def test_preserves_vietnamese_subject_action_and_location():
    result = QAQueryDecomposer().decompose(
        "Người đàn ông đang cầm gì trong tay bên bờ sông?"
    )

    assert result["retrieval_query"] == "Người đàn ông đang cầm trong tay bên bờ sông"


def test_removes_unknown_action_but_keeps_subject_and_location():
    result = QAQueryDecomposer().decompose(
        "Người phụ nữ đang làm gì bên đường?"
    )

    assert result["retrieval_query"] == "Người phụ nữ bên đường"


def test_removes_unknown_topic_scaffold():
    result = QAQueryDecomposer().decompose(
        "Người dẫn chương trình đang nói về chủ đề gì?"
    )

    assert result["retrieval_query"] == "Người dẫn chương trình"


def test_preserves_internal_vietnamese_copula():
    result = QAQueryDecomposer().decompose("Thành phố nào là thủ đô của Việt Nam?")

    assert result["retrieval_query"] == "Thành phố là thủ đô của Việt Nam"


def test_preserves_wh_shaped_token_at_start_of_proper_name():
    result = QAQueryDecomposer().decompose("Thủ đô của Ai Cập là gì?")

    assert result["retrieval_query"] == "Thủ đô của Ai Cập"


def test_preserves_wh_shaped_token_when_entity_starts_question():
    result = QAQueryDecomposer().decompose("Ai Cập nằm ở đâu?")

    assert result["retrieval_query"] == "Ai Cập nằm"


def test_removes_capitalized_who_pronoun_at_start_of_question():
    result = QAQueryDecomposer().decompose("Ai đang cầm chiếc ô đỏ?")

    assert result["retrieval_query"] == "đang cầm chiếc ô đỏ"


def test_preserves_english_holding_action_and_location():
    result = QAQueryDecomposer().decompose(
        "What is the man holding near the river?"
    )

    assert result["retrieval_query"] == "the man holding near the river"


def test_removes_unknown_english_doing_action():
    result = QAQueryDecomposer().decompose(
        "What is the woman doing beside the bus?"
    )

    assert result["retrieval_query"] == "the woman beside the bus"


def test_removes_english_fronted_auxiliary_but_keeps_constraints():
    result = QAQueryDecomposer().decompose(
        "Where did the red truck stop after crossing the bridge?"
    )

    assert result["retrieval_query"] == "the red truck stop after crossing the bridge"


def test_removes_dangling_english_copula():
    result = QAQueryDecomposer().decompose("The visible text is what?")

    assert result["retrieval_query"] == "The visible text"


def test_yes_no_question_is_not_rewritten_as_wh_question():
    result = QAQueryDecomposer().decompose("Is the red car crossing the bridge?")

    assert result["retrieval_query"] == "Is the red car crossing the bridge"


def test_output_contract_and_empty_decomposition_fallback_are_preserved():
    result = QAQueryDecomposer().decompose("Ai?")

    assert result == {
        "original_question": "Ai?",
        "retrieval_query": "Ai?",
        "answer_type": "unknown",
        "unknown_slot": "unknown",
        "known_constraints": {},
    }
