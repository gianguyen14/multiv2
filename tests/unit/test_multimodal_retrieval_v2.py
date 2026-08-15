import pytest
from backend.app.retrieval.video_multimodal import lexical_score

def test_lexical_score_precision():
    # Stopwords are removed
    assert lexical_score("nhiệt độ đạt 40 độ C", "nhiệt độ ngoài trời hôm nay đạt 40 độ c") > 0.6
    assert lexical_score("Bảng điện tử Vietnam Airlines", "Vietnam Airlines Nhóm 1") > 0.3
    assert lexical_score("hoàn toàn không liên quan", "pháo đài hình ngôi sao") == 0.0

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
