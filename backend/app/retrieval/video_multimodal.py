from dataclasses import dataclass, field
from functools import lru_cache

from backend.app.video.text_evidence import normalize_text

_STOPWORDS = frozenset({
    "có", "là", "và", "của", "trong", "ở", "một", "những", "cho", "để", "với",
    "không", "đến", "các", "thì", "mà", "như",
})
def _content_tokens(value):
    return [token for token in normalize_text(value).split() if token not in _STOPWORDS]


@lru_cache(maxsize=256)
def _query_features(query):
    tokens = tuple(_content_tokens(query))
    return tokens, frozenset(tokens)


def _phrase_relevance(query_tokens, text_tokens):
    """Measure exact ordered phrase evidence in linear time."""
    if len(query_tokens) < 2 or len(text_tokens) < 2:
        return 0.0
    separator = "\0"
    query_phrase = separator + separator.join(query_tokens) + separator
    text_phrase = separator + separator.join(text_tokens) + separator
    return 1.0 if query_phrase in text_phrase else 0.0


def lexical_score(query, text):
    """Score lexical relevance while resisting dilution from long queries.

    Dice overlap balances query coverage against evidence precision, so one
    generic shared term cannot dominate a long query. Contiguous multi-token
    phrases receive an additional signal.
    """
    query_tokens, query_terms = _query_features(query)
    text_tokens = _content_tokens(text)
    text_terms = set(text_tokens)
    if not query_terms or not text_terms:
        return 0.0
    overlap = query_terms & text_terms
    if not overlap:
        return 0.0

    dice_overlap = 2 * len(overlap) / (len(query_terms) + len(text_terms))
    phrase_relevance = _phrase_relevance(query_tokens, text_tokens)
    return min(1.0, max(dice_overlap, phrase_relevance))


def minmax(values):
    values = list(values)
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


@dataclass
class VideoSegmentCandidate:
    video_id: str
    start_frame: int
    end_frame: int
    representative_frame: int
    visual_score: float = 0.0
    asr_score: float = 0.0
    ocr_score: float = 0.0
    fused_score: float = 0.0
    supporting_evidence_ids: list[str] = field(default_factory=list)


class MultimodalVideoRetriever:
    def __init__(self, visual_search, ocr_records, asr_segments, weights=None):
        self.visual_search = visual_search
        self.ocr_records = list(ocr_records)
        self.asr_segments = list(asr_segments)
        self.weights = weights or {"visual": 1 / 3, "ocr": 1 / 3, "asr": 1 / 3}

    def search(self, query, top_k=100, modalities=("visual", "ocr", "asr")):
        candidates = {}
        if "visual" in modalities:
            for hit in self.visual_search(query, top_k):
                payload = hit.get("payload", hit)
                frame = payload["source_frame_index_zero_based"]
                key = (payload["video_id"], frame)
                candidates[key] = VideoSegmentCandidate(payload["video_id"], frame, frame, frame,
                    visual_score=float(hit.get("score", 0)), supporting_evidence_ids=[payload["frame_uid"]])
        if "ocr" in modalities:
            for record in self.ocr_records:
                score = lexical_score(query, record.normalized_text)
                if score <= 0:
                    continue
                key = (record.video_id, record.source_frame_index_zero_based)
                candidate = candidates.setdefault(key, VideoSegmentCandidate(record.video_id,
                    record.source_frame_index_zero_based, record.source_frame_index_zero_based,
                    record.source_frame_index_zero_based))
                candidate.ocr_score = max(candidate.ocr_score, score)
                candidate.supporting_evidence_ids.append(record.frame_uid)
        if "asr" in modalities:
            for segment in self.asr_segments:
                score = lexical_score(query, segment.normalized_text)
                if score <= 0 or segment.start_frame is None:
                    continue
                key = (segment.video_id, segment.start_frame)
                candidate = candidates.setdefault(key, VideoSegmentCandidate(segment.video_id,
                    segment.start_frame, segment.end_frame or segment.start_frame, segment.start_frame))
                candidate.asr_score = max(candidate.asr_score, score)
                candidate.supporting_evidence_ids.append(segment.segment_id)
        result = list(candidates.values())
        normalized = {name: minmax(getattr(item, f"{name}_score") for item in result)
            for name in modalities}
        for index, item in enumerate(result):
            item.fused_score = sum(self.weights[name] * normalized[name][index] for name in modalities)
        return sorted(result, key=lambda item: (-item.fused_score, item.video_id, item.representative_frame))[:top_k]
