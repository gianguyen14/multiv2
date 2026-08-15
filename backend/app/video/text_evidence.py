import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Optional


def normalize_text(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip().casefold()


@dataclass(frozen=True)
class OCRRecord:
    video_id: str
    frame_uid: str
    source_frame_index_zero_based: int
    timestamp_seconds: Optional[float]
    raw_text: str
    normalized_text: str
    boxes: list
    confidence: Optional[float]

    @classmethod
    def create(cls, record, raw_text, boxes=None, confidence=None):
        return cls(record.video_id, record.frame_uid, record.source_frame_index_zero_based,
            record.timestamp_seconds, raw_text, normalize_text(raw_text), boxes or [], confidence)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class ASRSegment:
    video_id: str
    segment_id: str
    start_seconds: float
    end_seconds: float
    start_frame: Optional[int]
    end_frame: Optional[int]
    raw_text: str
    normalized_text: str
    language: Optional[str]
    confidence: Optional[float]

    @classmethod
    def create(cls, video_id, index, start_seconds, end_seconds, start_frame, end_frame,
               raw_text, language=None, confidence=None):
        return cls(video_id, f"{video_id}:asr:{index:06d}", float(start_seconds), float(end_seconds),
            start_frame, end_frame, raw_text, normalize_text(raw_text), language, confidence)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
