"""Human-first annotations for the public Videos_L24_a review set.

These labels were created from the GitHub Actions review artifact (16-frame overview
plus p20/p40/p60/p80 detail frames). They are intentionally anchored to the p40
sample and do not reconstruct authoritative frame IDs from timestamp * FPS.
"""

from __future__ import annotations

HUMAN_ANNOTATIONS = [
    ("L24A-Q001", "L24_V002", 281.19, "Thi múa rồng: rồng vàng-xanh, đội biểu diễn mặc xanh, sân thi đấu có khán giả.", "đầu rồng", "Trong cảnh đội mặc xanh đang điều khiển rồng vàng-xanh, có bao nhiêu đầu rồng hiện rõ?", 1, "high"),
    ("L24A-Q002", "L24_V003", 305.27, "Thi múa rồng trên sân xanh; rồng vàng-xanh, đội mặc đồ tối/đen-vàng.", "đầu rồng", "Trong cảnh cận động tác của rồng vàng-xanh, có bao nhiêu đầu rồng hiện rõ?", 1, "high"),
    ("L24A-Q003", "L24_V004", 374.52, "Thi múa rồng xanh; đội biểu diễn mặc quân phục/rằn ri xanh.", "người mặc rằn ri", "Có bao nhiêu người mặc đồ rằn ri xanh đang trực tiếp ở trên sàn biểu diễn trong cảnh?", 7, "medium"),
    ("L24A-Q004", "L24_V005", 250.09, "Lân vàng-đỏ thi mai hoa thung ban ngày, góc toàn sân.", "bàn phủ khăn xanh", "Trong cảnh toàn sân khi lân vàng-đỏ ở trên cọc cao, có bao nhiêu bàn phủ khăn xanh nhạt nằm quanh sân?", 3, "high"),
    ("L24A-Q005", "L24_V006", 227.51, "Lân đỏ biểu diễn trên dãy cọc mai hoa thung ban ngày.", "con lân", "Trong cảnh lân đỏ ở trước phông nền màu xanh, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q006", "L24_V007", 228.19, "Lân vàng biểu diễn trên cọc ban ngày, khán giả và cờ tam giác ở hậu cảnh.", "cờ tam giác", "Trong cảnh lân vàng đứng cao trên cọc, có bao nhiêu lá cờ tam giác lớn ở phía sau bên phải?", 2, "high"),
    ("L24A-Q007", "L24_V008", 224.84, "Lân vàng/kem trên dãy cọc ban ngày; góc toàn sân.", "cờ tam giác", "Trong cảnh toàn sân có lân vàng/kem trên cọc, có bao nhiêu lá cờ tam giác lớn ở phía phải?", 2, "high"),
    ("L24A-Q008", "L24_V009", 206.98, "Lân vàng biểu diễn trên cọc trước phông xanh của giải.", "con lân", "Trong cảnh lân vàng đứng trước màn hình nền xanh, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q009", "L24_V010", 187.50, "Lân trắng-đỏ nhảy giữa các cọc ban ngày.", "con lân", "Trong cảnh lân trắng-đỏ đang ở trên cao giữa dãy cọc, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q010", "L24_V011", 259.80, "Lân trắng thi trên cọc; có đội nhạc cụ/cymbal ở sát sân.", "người mặc đồng phục cam-đỏ", "Trong cảnh các nhạc công cạnh sân, có bao nhiêu người mặc đồng phục cam-đỏ hiện rõ ở tiền cảnh?", 3, "high"),
    ("L24A-Q011", "L24_V012", 38.89, "Đoạn phỏng vấn ngắn ngoài trời với các thành viên trẻ mặc áo trắng và đỏ.", "người mặc áo đỏ", "Có bao nhiêu người mặc áo đỏ đứng rõ phía sau người đang được phỏng vấn?", 2, "high"),
    ("L24A-Q012", "L24_V013", 208.87, "Lân đỏ thi mai hoa thung ban ngày; xen cảnh khán giả.", "trẻ em đội mũ lưỡi trai", "Trong cảnh khán giả, có bao nhiêu trẻ em đội mũ lưỡi trai nhìn rõ ở hàng phía trước?", 3, "medium"),
    ("L24A-Q013", "L24_V014", 210.28, "Lân trắng thi trên cọc ban ngày.", "con lân", "Trong cảnh lân trắng ở trên dãy cọc, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q014", "L24_V015", 238.86, "Lân trắng họa tiết đen thi ban đêm; có người hỗ trợ mặc vàng-đen và đạo cụ rồng ở nền.", "người áo vàng-đen", "Trong cảnh lân trắng ở gần mặt đất, có bao nhiêu người mặc áo vàng-đen đứng cạnh lân?", 2, "high"),
    ("L24A-Q015", "L24_V016", 225.40, "Lân trắng thi trên cọc ban đêm.", "con lân", "Trong cảnh lân trắng đứng trên cọc ban đêm, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q016", "L24_V017", 45.91, "Clip ngắn lân trắng trên cọc cao ban đêm, có đoạn chuyển sang lân vàng.", "con lân", "Trong cảnh lân trắng ở trên cọc cao, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q017", "L24_V018", 263.90, "Lân vàng-đen thi mai hoa thung ban đêm.", "cờ tam giác", "Trong cảnh lân vàng-đen trên cọc, có bao nhiêu lá cờ tam giác lớn ở hậu cảnh?", 2, "high"),
    ("L24A-Q018", "L24_V019", 208.16, "Lân vàng-xanh thi trên cọc ban ngày.", "con lân", "Trong cảnh cận lân vàng-xanh, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q019", "L24_V020", 198.91, "Lân vàng-đỏ thi trên sân mai hoa thung ban ngày; góc toàn sân.", "tấm đệm đỏ", "Trong cảnh toàn sân, có bao nhiêu tấm đệm đỏ lớn được ghép ở khu đáp xuống bên phải?", 3, "high"),
    ("L24A-Q020", "L24_V021", 226.87, "Lân vàng thi trên cọc ban ngày; góc toàn sân có khu đáp đệm đỏ.", "tấm đệm đỏ", "Trong cảnh toàn sân khi lân vàng đứng trên cọc, có bao nhiêu tấm đệm đỏ lớn ở khu đáp xuống bên phải?", 3, "high"),
    ("L24A-Q021", "L24_V022", 236.26, "Lân vàng thi trên cọc ban ngày.", "con lân", "Trong cảnh lân vàng đứng thẳng trên cọc, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q022", "L24_V023", 241.85, "Lân đen thi trên cọc ban ngày.", "con lân", "Trong cảnh lân đen đang biểu diễn trên cọc, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q023", "L24_V024", 225.66, "Lân trắng thi trên cọc ban ngày, có nhạc công ở cạnh sân.", "con lân", "Trong cảnh lân trắng ở trên cọc, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q024", "L24_V025", 224.98, "Lân trắng họa tiết đen thi trên cọc ban ngày; góc toàn sân.", "tấm đệm đỏ", "Trong cảnh toàn sân, có bao nhiêu tấm đệm đỏ lớn ở khu đáp xuống bên phải?", 3, "high"),
    ("L24A-Q025", "L24_V026", 227.95, "Lân vàng thi trên cọc ban ngày.", "con lân", "Trong cảnh cận lân vàng trên cọc, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q026", "L24_V027", 227.44, "Lân trắng-đen thi trên cọc ban đêm.", "con lân", "Trong cảnh cận lân trắng-đen ở trên cọc, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q027", "L24_V028", 227.66, "Lân vàng thi mai hoa thung ban đêm; góc toàn sân.", "tấm đệm đỏ", "Trong cảnh toàn sân ban đêm, có bao nhiêu tấm đệm đỏ lớn ở khu đáp xuống bên phải?", 3, "high"),
    ("L24A-Q028", "L24_V029", 221.80, "Lân trắng thi trên cọc ban đêm.", "con lân", "Trong cảnh lân trắng đang ở trên một cọc cao, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q029", "L24_V030", 258.05, "Lân trắng-đỏ thi ban đêm; xen cảnh đội trống/cymbal.", "con lân", "Trong cảnh cận lân trắng-đỏ ban đêm, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q030", "L24_V031", 227.62, "Lân trắng-đỏ thi trên cọc ban đêm.", "con lân", "Trong cảnh cận đầu lân trắng-đỏ, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q031", "L24_V032", 286.35, "Lân vàng thi trên cọc ban đêm.", "con lân", "Trong cảnh cận lân vàng trên dãy cọc ban đêm, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q032", "L24_V033", 234.64, "Lân trắng-xanh thi trên cọc ban đêm.", "con lân", "Trong cảnh lân trắng-xanh trên cọc, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q033", "L24_V035", 264.71, "Lân vàng thi trên cọc ban đêm.", "con lân", "Trong cảnh lân vàng đứng cao trên cọc, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q034", "L24_V036", 117.50, "Lân trắng-hồng thi ban đêm; góc toàn sân với dãy cọc và đệm đáp.", "tấm đệm đỏ", "Trong cảnh toàn sân ban đêm, có bao nhiêu tấm đệm đỏ lớn ở khu đáp xuống bên phải?", 3, "high"),
    ("L24A-Q035", "L24_V037", 255.55, "Lân vàng thi trên cọc ban đêm.", "con lân", "Trong cảnh lân vàng trên cọc ban đêm, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q036", "L24_V038", 203.22, "Lân trắng-xanh thi trên cọc ban đêm.", "con lân", "Trong cảnh lân trắng-xanh đứng cao trên cọc, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q037", "L24_V039", 208.03, "Lân trắng-đỏ thi trên cọc ban đêm.", "con lân", "Trong cảnh cận lân trắng-đỏ trên cọc, có bao nhiêu con lân hiện rõ?", 1, "high"),
    ("L24A-Q038", "L24_V040", 15.11, "Clip dọc ngắn: lân trắng thi trên cọc cao ban đêm.", "con lân", "Trong clip dọc, có bao nhiêu con lân trắng hiện rõ trong cảnh này?", 1, "high"),
    ("L24A-Q039", "L24_V041", 14.34, "Clip dọc ngắn: lân trắng-xanh trên cọc ban đêm.", "con lân", "Trong clip dọc, có bao nhiêu con lân trắng-xanh hiện rõ trong cảnh này?", 1, "high"),
    ("L24A-Q040", "L24_V042", 222.56, "Lân trắng-đỏ thi trên cọc ban ngày; góc toàn sân.", "tấm đệm đỏ", "Trong cảnh toàn sân, có bao nhiêu tấm đệm đỏ lớn ở khu đáp xuống bên phải?", 3, "high"),
    ("L24A-Q041", "L24_V043", 275.24, "Lân trắng-cam thi trên cọc ban ngày; góc toàn sân.", "tấm đệm đỏ", "Trong cảnh toàn sân khi lân trắng-cam đứng trên cọc, có bao nhiêu tấm đệm đỏ lớn ở khu đáp xuống bên phải?", 3, "high"),
    ("L24A-Q042", "L24_V044", 14.14, "Clip dọc ngắn: lân vàng thi trên cọc.", "con lân", "Trong clip dọc, có bao nhiêu con lân vàng hiện rõ trong cảnh này?", 1, "high"),
    ("L24A-Q043", "L24_V045", 13.02, "Clip dọc ngắn: lân vàng thực hiện động tác trên cọc.", "con lân", "Trong clip dọc, có bao nhiêu con lân vàng hiện rõ trong cảnh này?", 1, "high"),
]


def records():
    for qid, video_id, ts, summary, target, question, answer, confidence in HUMAN_ANNOTATIONS:
        yield {
            "id": qid,
            "task": "counting_qa",
            "video_id": video_id,
            "human_summary_vi": summary,
            "question_vi": question,
            "answer": str(answer),
            "answer_count": answer,
            "count_target": target,
            "evidence_timestamp_s": ts,
            "evidence_sample": "p40",
            "annotation_confidence": confidence,
            "ground_truth_scope": "single_sampled_frame",
            "authoritative_frame_id": None,
            "frame_id_note": "Do not derive authoritative frame ID from timestamp/FPS.",
        }
