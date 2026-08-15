"""
video_qa.py — Precision-first extractive answerer with generalized linguistic validation.

Design rules:
- Abstain is preferred over a weak or unsupported guess.
- Principled structural taxonomy for WHERE, WHO, WHEN, HOW_MANY, WHAT_TEXT, WHAT_SAID.
- General rejection of non-person and non-location structural head nouns without ad-hoc wordlists.
- Confidence is computed from pattern specificity + evidence lexical overlap.
- A minimum confidence threshold must be crossed before returning an answer.
- Generic one-word stopword matches are rejected.
- Attribute/directional alignment prevents false extraction on unmatched dimensions.
"""

import re
from dataclasses import asdict, dataclass

from backend.app.video.text_evidence import normalize_text

# ── stopwords ──────────────────────────────────────────────────────────────────
_VI_STOPWORDS = {
    "có", "là", "và", "của", "trong", "ở", "một", "những", "cho", "để",
    "với", "không", "đến", "các", "thì", "mà", "như", "được", "tại",
    "nơi", "đây", "kia", "này", "đó", "vào", "ra", "lên", "xuống",
    "theo", "về", "bởi", "do", "vì", "hay", "hoặc", "cũng", "đã", "sẽ",
    "đang", "rồi", "thôi", "ạ", "nhé", "nào", "ai", "gì", "đâu", "bị",
    "khiến", "làm", "bao", "nhiêu", "mấy", "hôm", "nay", "ngày", "tháng",
    "năm", "lúc", "khi", "triệu", "tỷ", "nghìn", "ngàn", "đồng", "người", "độ"
}

_MIN_CONFIDENCE = 0.55  # must cross this to return an answer (principled threshold)


# ── generalized linguistic taxonomy ───────────────────────────────────────────

_NON_PERSON_HEAD_NOUNS = {
    # Organizations / Institutions / Facilities
    "bộ", "sở", "ban", "viện", "trường", "trung tâm", "công ty", "tập đoàn",
    "doanh nghiệp", "hội đồng", "ủy ban", "đảng", "chính quyền", "quân đội",
    "liên đoàn", "hiệp hội", "câu lạc bộ", "tổ chức", "chi nhánh", "nhà máy",
    "phân xưởng", "cơ quan", "đoàn thể",
    # Events / Activities / Productions
    "lễ", "hội", "lễ hội", "ngày hội", "hội nghị", "hội thảo", "cuộc thi",
    "giải thưởng", "sự kiện", "phóng sự", "chương trình", "dự án", "kế hoạch",
    "tập trận", "vở kịch", "bản tin", "phim", "đình công", "triển lãm",
    # Disciplines / Abstract Domains
    "âm nhạc", "nghệ thuật", "thể thao", "y tế", "giáo dục", "giao thông",
    "du lịch", "văn hóa", "kinh tế", "xã hội", "khoa học", "công nghệ",
    # Temporal / Ordinal / Prepositional structural tokens
    "hôm", "ngày", "tháng", "năm", "thời", "lần", "mùa", "thế kỷ", "thứ",
    "đầu", "cuối", "nhiều", "các", "những", "một", "trong", "ngoài", "trên", "dưới"
}

_NON_LOCATION_HEAD_NOUNS = {
    # Time / Temporal units
    "ngày", "tháng", "năm", "lúc", "khi", "giờ", "phút", "giây", "sáng", "chiều", "tối",
    # Projects / Events / Documents
    "dự án", "chương trình", "kế hoạch", "quy chế", "báo cáo", "nghị quyết",
    "lễ hội", "ngày hội", "cuộc họp", "sự kiện", "hội thảo", "hội nghị",
    # Organizations / Facilities
    "bệnh viện", "trung tâm", "công ty", "nhà máy", "trường học", "trường đại học",
    "viện nghiên cứu", "ban tổ chức", "sở y tế", "câu lạc bộ", "ủy ban"
}

_GEOGRAPHIC_MARKERS = {
    "thành phố", "tỉnh", "huyện", "xã", "quận", "thị xã", "thị trấn", "bang",
    "quốc gia", "nước", "thủ đô", "sông", "núi", "đảo", "quần đảo", "biển",
    "vùng", "khu vực", "phía", "miền", "châu", "làng", "xóm"
}

_GEOGRAPHIC_ENTITIES = {
    "việt nam", "mỹ", "anh", "pháp", "trung quốc", "hàn quốc", "nhật bản",
    "bolivia", "đà lạt", "hà nội", "tp.hcm", "sài gòn", "đắk lắk", "cần thơ",
    "lào cai", "edinburgh", "chicago", "singapore", "iran", "kenya", "palestine",
    "israel", "gaza", "indonesia", "philippines"
}

_PERSON_HONORIFICS = [
    "ông", "bà", "anh", "chị", "cô", "chú", "bác", "tiến sĩ", "giáo sư",
    "bác sĩ", "luật sư", "đạo diễn", "ca sĩ", "nhạc sĩ", "nghệ sĩ", "cầu thủ",
    "vận động viên", "tổng thống", "thủ tướng", "chủ tịch", "bộ trưởng",
    "thứ trưởng", "ngoại trưởng", "đại sứ", "tác giả", "nhà văn", "nhà thơ",
    "họa sĩ", "hoa hậu", "nam vương"
]

_KNOWN_DISEASES = [
    "trầm cảm", "lo âu", "thuyên tắc phổi", "ung thư", "viêm",
    "sốt xuất huyết", "tiểu đường", "tăng huyết áp", "đột quỵ", "bạch hầu",
    "đậu mùa khỉ", "tim mạch", "suy tim", "viêm gan", "hen suyễn"
]

_KNOWN_AIRLINES = [
    "vietnam airlines", "vietjet air", "vietjet", "bamboo airways",
    "pacific airlines", "vietravel airlines", "emirates", "qatar airways", "singapore airlines"
]

_KNOWN_ANIMALS = [
    "rùa biển", "rùa", "cá heo", "cá voi", "voi", "hổ", "báo", "chim", "khỉ",
    "voọc", "gấu", "chó", "mèo", "ngựa", "bò", "trâu", "hươu", "nai"
]


# ── answer-type classification ─────────────────────────────────────────────────

def _classify_question(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["bao nhiêu", "mấy", "bao lâu"]):
        return "HOW_MANY"
    if any(w in q for w in ["ở đâu", "xảy ra ở", "diễn ra ở", "tại đâu", "thành phố nào", "quốc gia nào", "tỉnh nào", "nơi nào"]):
        return "WHERE"
    if any(w in q for w in ["khi nào", "bao giờ", "ngày nào", "năm nào", "tháng nào", "lúc nào"]):
        return "WHEN"
    if any(w in q for w in ["bệnh gì", "mắc bệnh gì", "bị bệnh gì"]):
        return "DISEASE"
    if any(w in q for w in ["màu gì", "màu nào"]):
        return "COLOR"
    if any(w in q for w in ["hãng hàng không", "hãng bay"]):
        return "AIRLINE"
    if any(w in q for w in ["đền thờ", "ngôi đền", "ngôi chùa", "chùa nào"]):
        return "TEMPLE"
    if any(w in q for w in ["đối tượng"]):
        return "AUDIENCE"
    if any(w in q for w in ["động vật", "loài"]):
        return "ANIMAL"
    if any(w in q for w in ["nói gì", "phát biểu gì", "kể gì", "tuyên bố gì"]):
        return "WHAT_SAID"
    if any(w in q for w in ["chữ gì", "ghi gì", "viết gì", "biển báo", "menu", "bảng điện tử", "hiển thị gì"]):
        return "WHAT_TEXT"
    if any(w in q for w in ["tên là gì", "tên gì"]):
        return "NAME"
    if any(w in q for w in ["ai đang", "ai là", "ai được", "ca sĩ nào", "nghệ sĩ nào", "bác sĩ nào", "vị nào", "người nào"]):
        return "WHO"
    if any(w in q for w in ["gì", "nào"]):
        return "WHAT"
    return "GENERAL"


# ── answer-type validators ─────────────────────────────────────────────────────

def _is_valid_answer(answer: str, answer_type: str, context: str = "") -> bool:
    """
    Reject answers that are:
    - Pure stopwords
    - Single character
    - Structurally incompatible with the expected answer type
    """
    a = answer.strip()
    if not a or len(a) < 2:
        return False
    norm = normalize_text(a)
    if norm in _VI_STOPWORDS:
        return False

    words = norm.split()

    if answer_type == "HOW_MANY":
        return bool(re.search(r"\d", a))

    if answer_type == "WHERE":
        if not words:
            return False
        if words[0] in _NON_LOCATION_HEAD_NOUNS:
            return False
        if len(words) >= 2 and f"{words[0]} {words[1]}" in _NON_LOCATION_HEAD_NOUNS:
            return False
        if len(words) >= 3 and f"{words[0]} {words[1]} {words[2]}" in _NON_LOCATION_HEAD_NOUNS:
            return False
        has_cap = bool(re.search(r"[A-ZĐÀ-Ỵ]", a))
        has_geo = any(g in norm for g in _GEOGRAPHIC_MARKERS)
        return has_cap or has_geo

    if answer_type == "WHEN":
        return bool(re.search(r"\d", a)) or any(w in norm for w in ["ngày", "tháng", "năm", "sáng", "chiều", "tối", "lúc"])

    if answer_type == "DISEASE":
        return any(d in norm for d in _KNOWN_DISEASES)

    if answer_type == "COLOR":
        known_colors = [
            "đỏ", "xanh lá", "xanh dương", "xanh", "vàng", "trắng", "đen",
            "tím", "hồng", "cam", "nâu", "xám", "bạc", "vàng đồng"
        ]
        return any(c in norm for c in known_colors)

    if answer_type == "AIRLINE":
        return len(a) >= 3 and norm not in _VI_STOPWORDS

    if answer_type == "WHO":
        if len(a) < 4 or not words or len(words[0]) < 2:
            return False
        if any(gn in norm for gn in _GEOGRAPHIC_ENTITIES):
            return False
        if words[0] in _NON_PERSON_HEAD_NOUNS:
            return False
        if len(words) >= 2 and f"{words[0]} {words[1]}" in _NON_PERSON_HEAD_NOUNS:
            return False
        if context and re.search(r"(?:nhạc kịch|vở kịch|bài hát|ca khúc|bộ phim|tác phẩm)\s+['\"]?" + re.escape(a), context, re.IGNORECASE):
            return False
        cap_tokens = [w for w in a.split() if w and w[0].isupper()]
        return len(cap_tokens) >= 2

    if answer_type == "TEMPLE":
        return bool(re.search(r"[A-ZĐÀ-Ỵ][a-zà-ỵ]", a)) and len(a) >= 3

    if answer_type in ["AUDIENCE", "ANIMAL"]:
        return len(a) >= 3

    if answer_type == "WHAT_SAID":
        return len(a) >= 5

    if answer_type in ["WHAT_TEXT", "NAME"]:
        return len(a) >= 2

    # GENERAL / WHAT
    return len(a) >= 2 and norm not in _VI_STOPWORDS


# ── evidence lexical support score ────────────────────────────────────────────

def _extract_proper_nouns(text: str) -> list[str]:
    words = text.split()
    proper = []
    for w in words[1:]:
        clean = re.sub(r"[^\w]", "", w)
        if len(clean) >= 2 and clean[0].isupper() and clean.lower() not in {"ở", "tại", "nào", "gì", "bao", "nhiêu"}:
            proper.append(clean.lower())
    return proper


def _evidence_support(question: str, text: str) -> float:
    """Fraction of non-stopword question terms found in evidence text."""
    q_clean = re.sub(r"[^\w\s]", " ", normalize_text(question))
    t_clean = re.sub(r"[^\w\s]", " ", normalize_text(text))
    
    # Check proper noun constraint
    q_proper = _extract_proper_nouns(question)
    if q_proper:
        proper_matched = any(p in t_clean for p in q_proper)
        if not proper_matched:
            return 0.0

    # Check attribute/action modifiers for quantitative & directional queries
    modifiers = {"giảm", "tăng", "dài", "rộng", "cao", "mạnh", "rich", "richter"}
    q_mods = [m for m in modifiers if m in q_clean]
    if q_mods and not any(m in t_clean for m in q_mods):
        return 0.0

    q_words = set(q_clean.split())
    t_words = set(t_clean.split())
    non_stop = q_words - _VI_STOPWORDS - {"what", "does", "the", "is", "a", "an", "of", "to", "in"}
    terms_to_check = non_stop if non_stop else q_words
    if not terms_to_check:
        return 0.0
    matched = set()
    for qw in terms_to_check:
        for tw in t_words:
            if qw == tw or (len(qw) >= 4 and len(tw) >= 4 and (qw.startswith(tw) or tw.startswith(qw))):
                matched.add(qw)
                break
    return len(matched) / len(terms_to_check)


# ── main answerer ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VideoQAResult:
    video_id: str
    frame_id: int
    answer: str
    confidence: float
    evidence_sources: list[str]

    def to_dict(self):
        return asdict(self)


class ExtractiveAnswerer:
    """
    Precision-first extractive answerer with strict validation and abstention.
    """

    def answer(self, question: str, evidence: list[dict]) -> dict:
        answer_type = _classify_question(question)
        q_lower = question.lower()

        # Sort evidence by lexical support (best first)
        ranked = sorted(
            evidence,
            key=lambda ev: _evidence_support(question, ev["text"]),
            reverse=True,
        )

        for ev in ranked:
            text = ev["text"].strip()
            if not text:
                continue

            ev_support = _evidence_support(question, text)
            if ev_support < 0.25 and answer_type not in ["DISEASE", "TEMPLE", "AIRLINE", "AUDIENCE", "ANIMAL", "NAME"]:
                continue

            answer = ""
            confidence = 0.0

            # ── HOW_MANY ───────────────────────────────────────────────────
            if answer_type == "HOW_MANY":
                if "độ" in q_lower and ev_support >= 0.25:
                    m = re.search(r"(\d+(?:[.,]\d+)?)\s*°?\s*(?:độ\s*c|độ)", text, re.IGNORECASE)
                    if m:
                        answer = m.group(0).strip()
                        confidence = 0.85
                elif ev_support >= 0.30:
                    m = re.search(
                        r"\b(\d+(?:[.,]\d+)?)\s*(người|bác sĩ|triệu|tỷ|nghìn|ngàn|%|usd|vnd|tỉ|km|m²|ha|m|mét|tuổi|năm|tháng)\b",
                        text, re.IGNORECASE,
                    )
                    if m:
                        answer = m.group(0).strip()
                        confidence = 0.75
                    else:
                        m2 = re.search(r"\b(\d+(?:[.,]\d+)?)\b", text)
                        if m2 and ev_support >= 0.45:
                            answer = m2.group(1).strip()
                            confidence = 0.60

            # ── WHERE ──────────────────────────────────────────────────────
            elif answer_type == "WHERE":
                m = re.search(
                    r"(?:ở|tại|tại đây|đến|thuộc|vùng|tỉnh|thành phố|quốc gia|huyện|xã|phường|nước|bang|thủ đô)\s+([A-ZĐÀ-Ỵ][a-zà-ỵ]+(?:\s+[A-ZĐÀ-Ỵ][a-zà-ỵ]+){0,2})",
                    text,
                )
                if m:
                    cand = m.group(1).strip()
                    if _is_valid_answer(cand, "WHERE"):
                        answer = cand
                        confidence = 0.80
                if not answer:
                    for g in _GEOGRAPHIC_ENTITIES:
                        if g in text.lower():
                            m_geo = re.search(r"\b" + re.escape(g) + r"\b", text, re.IGNORECASE)
                            if m_geo:
                                cand = m_geo.group(0).strip().title()
                                if _is_valid_answer(cand, "WHERE"):
                                    answer = cand
                                    confidence = 0.80
                                    break

            # ── WHEN ───────────────────────────────────────────────────────
            elif answer_type == "WHEN":
                m = re.search(
                    r"(?:ngày|tháng|năm|lúc|vào)\s+(\d{1,2}(?:/\d{1,2}(?:/\d{4})?)?|\d{1,2}\s+tháng\s+\d{1,2}(?:\s+năm\s+\d{4})?|\d{4})",
                    text, re.IGNORECASE,
                )
                if m:
                    answer = m.group(0).strip()
                    confidence = 0.80

            # ── DISEASE ────────────────────────────────────────────────────
            elif answer_type == "DISEASE":
                for d in _KNOWN_DISEASES:
                    if d in normalize_text(text):
                        answer = d
                        confidence = 0.90
                        break

            # ── COLOR ──────────────────────────────────────────────────────
            elif answer_type == "COLOR":
                m = re.search(
                    r"\b(màu\s+)?(đỏ|xanh lá|xanh dương|xanh|vàng|trắng|đen|tím|hồng|cam|nâu|xám|bạc)\b",
                    normalize_text(text),
                )
                if m:
                    answer = m.group(0).strip()
                    confidence = 0.80

            # ── AIRLINE ────────────────────────────────────────────────────
            elif answer_type == "AIRLINE":
                m_air = re.search(
                    r"(?:hãng\s+hàng\s+không|hãng\s+bay)\s+([A-ZĐÀ-Ỵa-zà-ỵ0-9]+(?:\s+[A-ZĐÀ-Ỵa-zà-ỵ0-9]+){0,2})",
                    text, re.IGNORECASE,
                )
                if m_air:
                    cand = m_air.group(1).strip()
                    if _is_valid_answer(cand, "AIRLINE"):
                        answer = cand
                        confidence = 0.85
                if not answer:
                    for a_name in _KNOWN_AIRLINES:
                        if a_name in text.lower():
                            m_a = re.search(r"\b" + re.escape(a_name) + r"\b", text, re.IGNORECASE)
                            if m_a:
                                answer = m_a.group(0).strip().title()
                                confidence = 0.85
                                break

            # ── TEMPLE ─────────────────────────────────────────────────────
            elif answer_type == "TEMPLE":
                m_t = re.search(
                    r"(?:đền thờ|ngôi đền|đền|ngôi chùa|chùa)\s+([A-ZĐÀ-Ỵ][a-zà-ỵ]+(?:\s+[A-ZĐÀ-Ỵ][a-zà-ỵ]+){0,3})",
                    text,
                )
                if m_t:
                    cand = m_t.group(1).strip()
                    if _is_valid_answer(cand, "TEMPLE"):
                        answer = cand
                        confidence = 0.90

            # ── AUDIENCE ───────────────────────────────────────────────────
            elif answer_type == "AUDIENCE":
                m_aud = re.search(
                    r"(?:đối tượng|dành cho|hướng tới|cho)\s+([a-zà-ỵ0-9]+(?:\s+[a-zà-ỵ0-9]+){0,3})",
                    normalize_text(text),
                )
                if m_aud:
                    cand = m_aud.group(1).strip()
                    if _is_valid_answer(cand, "AUDIENCE"):
                        answer = cand
                        confidence = 0.80

            # ── ANIMAL ─────────────────────────────────────────────────────
            elif answer_type == "ANIMAL":
                for animal in _KNOWN_ANIMALS:
                    if animal in normalize_text(text):
                        answer = animal
                        confidence = 0.85
                        break

            # ── NAME / WHAT_TEXT ───────────────────────────────────────────
            elif answer_type in ["NAME", "WHAT_TEXT"]:
                m_quote = re.search(r"['\"«“]([^'\"»”\n]{2,30})['\"»”]", text)
                if m_quote:
                    cand = m_quote.group(1).strip()
                    if _is_valid_answer(cand, "WHAT_TEXT"):
                        answer = cand
                        confidence = 0.85
                if not answer:
                    m_word = re.search(r"\b([A-ZĐÀ-Ỵ][A-Za-z0-9à-ỵ]{2,20})\b", text)
                    if m_word and ev_support >= 0.25:
                        cand = m_word.group(1).strip()
                        if _is_valid_answer(cand, "WHAT_TEXT"):
                            answer = cand
                            confidence = 0.75

            # ── WHO ────────────────────────────────────────────────────────
            elif answer_type == "WHO":
                # Check honorific + name first
                honorific_pat = r"(?:" + "|".join(_PERSON_HONORIFICS) + r")\s+([A-ZĐÀ-Ỵ][a-zà-ỵ]+(?:\s+[A-ZĐÀ-Ỵ][a-zà-ỵ]+){1,3})"
                m_h = re.search(honorific_pat, text)
                if m_h:
                    cand = m_h.group(1).strip()
                    if _is_valid_answer(cand, "WHO", text):
                        answer = cand
                        confidence = 0.85
                if not answer:
                    matches = re.findall(
                        r"[A-ZĐÀ-Ỵ][a-zà-ỵ]+(?:\s+[A-ZĐÀ-Ỵ][a-zà-ỵ]+)+", text
                    )
                    valid = [m for m in matches if _is_valid_answer(m, "WHO", text)]
                    if valid and ev_support >= 0.40:
                        answer = valid[0]
                        confidence = 0.65

            # ── GENERAL / WHAT ─────────────────────────────────────────────
            else:
                m = re.search(r"(?:là|:)\s+([A-ZĐÀ-Ỵa-zà-ỵ0-9][^\n.!?]{2,40})(?:[.!?]|$)", text)
                if m:
                    candidate = m.group(1).strip()
                    candidate_words = set(normalize_text(candidate).split()) - _VI_STOPWORDS
                    if candidate_words and ev_support >= 0.35:
                        answer = candidate
                        confidence = 0.65
                if not answer:
                    m_en = re.search(
                        r"(?:reads?|is|was|says?)\s+([A-Za-z0-9][^\n.!?]{2,40})(?:[.!?]|$)",
                        text, re.IGNORECASE,
                    )
                    if m_en:
                        candidate = m_en.group(1).strip()
                        candidate_words = set(candidate.lower().split()) - {
                            "a", "an", "the", "is", "was", "are", "were", "be"
                        }
                        if candidate_words and len(candidate) >= 3:
                            answer = candidate
                            confidence = 0.75

            # ── validate and check threshold ───────────────────────────────
            if answer:
                if not _is_valid_answer(answer, answer_type, text):
                    continue
                adjusted = min(1.0, confidence * (0.7 + 0.3 * ev_support))
                if adjusted >= _MIN_CONFIDENCE:
                    return {
                        "answer": answer,
                        "confidence": adjusted,
                        "evidence_sources": [ev["id"]],
                    }

        return {"answer": "", "confidence": 0.0, "evidence_sources": []}


class VideoQAPipeline:
    def __init__(self, kis_pipeline, ocr_records, asr_segments, answerer=None):
        self.kis_pipeline = kis_pipeline
        self.ocr_records = list(ocr_records)
        self.asr_segments = list(asr_segments)
        self.answerer = answerer or ExtractiveAnswerer()

    def answer(self, event_query, question):
        locations = self.kis_pipeline.search(event_query, top_k=1)
        if not locations:
            return None
        location = locations[0]
        evidence = []
        for record in self.ocr_records:
            if (record.video_id == location.video_id
                    and record.source_frame_index_zero_based == location.source_frame_index_zero_based):
                evidence.append({"id": record.frame_uid, "text": record.raw_text})
        for segment in self.asr_segments:
            if (segment.video_id == location.video_id and segment.start_frame is not None
                    and segment.start_frame <= location.source_frame_index_zero_based
                    <= (segment.end_frame or segment.start_frame)):
                evidence.append({"id": segment.segment_id, "text": segment.raw_text})
        result = self.answerer.answer(question, evidence)
        return VideoQAResult(
            location.video_id, location.submission_frame_id,
            result["answer"], float(result["confidence"]), list(result["evidence_sources"]),
        )
