"""Corpus-independent regressions for focused query planning.

The examples in this module are synthetic on purpose.  They exercise linguistic
and ranking invariants without referring to a video id, frame id, or competition
ground truth.
"""

import pytest

from backend.app.retrieval.qa_query_decomposition import QAQueryDecomposer
from backend.app.retrieval.video_multimodal import lexical_score
from backend.app.services.query_refiner import DeterministicQueryParser


@pytest.mark.parametrize(
    ("query", "focus"),
    [
        ("Tìm kiếm cảnh sát đang điều tiết giao thông", "cảnh sát"),
        ("Tìm phim trường có diễn viên mặc áo đỏ", "phim trường"),
        ("Xác định cảnh quan vùng núi", "cảnh quan"),
    ],
)
def test_search_intent_cleanup_does_not_consume_a_noun_prefix(query, focus):
    plan = DeterministicQueryParser().parse(query)

    assert focus in plan.visual_queries[0].text.casefold()


def test_exact_text_cleanup_does_not_damage_a_containing_proper_name():
    plan = DeterministicQueryParser().parse(
        'Tìm người đứng trước ArtHouse với dòng chữ "ART"'
    )

    assert "ART" in plan.exact_strings
    assert "ArtHouse" in plan.visual_queries[0].text


def test_qa_cleanup_preserves_proper_noun_that_starts_like_a_wh_word():
    question = "Thủ đô của Ai Cập là gì?"

    decomposed = QAQueryDecomposer().decompose(question)["retrieval_query"]
    visual = DeterministicQueryParser().parse(
        question, task_type="qa"
    ).visual_queries[0].text

    assert "Ai Cập" in decomposed
    assert "Ai Cập" in visual
    assert "người Cập" not in visual


def test_long_query_prefers_multi_term_focus_over_a_generic_single_overlap():
    noise = " ".join(f"chi-tiết-{index}" for index in range(30))
    query = f"{noise} cầu tre"

    focused = lexical_score(query, "cầu tre")
    generic = lexical_score(query, "cầu")
    unrelated = lexical_score(query, "nhà ga")

    assert focused > generic > unrelated
    assert all(0.0 <= score <= 1.0 for score in (focused, generic, unrelated))
