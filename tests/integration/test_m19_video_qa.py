from backend.app.retrieval.kis_pipeline import KISResult
from backend.app.retrieval.video_qa import VideoQAPipeline
from backend.app.video.frame_record import FrameRecord
from backend.app.video.text_evidence import ASRSegment, OCRRecord


class KIS:
    def search(self, query, top_k):
        return [KISResult("video", 10, 11, 1, 8)]


def record():
    return FrameRecord.create(video_id="video", source_frame_index_zero_based=10,
        submission_frame_id=11, timestamp_seconds=1, pts=10, width=1, height=1,
        image_path="x", sample_interval_seconds=1, ingestion_version="test")


def test_qa_localizes_and_answers_from_ocr_evidence():
    ocr = [OCRRecord.create(record(), "The sign reads HCMC 2026.", confidence=1)]
    result = VideoQAPipeline(KIS(), ocr, []).answer("person near sign", "What does the sign read?")
    assert result.video_id == "video" and result.frame_id == 11
    assert result.answer == "HCMC 2026"
    assert result.evidence_sources == ["video:000000010"]


def test_qa_uses_asr_and_abstains_without_evidence():
    asr = [ASRSegment.create("video", 0, 0, 2, 8, 12, "The code is alpha seven.", "en", 1)]
    result = VideoQAPipeline(KIS(), [], asr).answer("speaker", "What is the code?")
    assert result.answer == "alpha seven"
    empty = VideoQAPipeline(KIS(), [], []).answer("speaker", "What is said?")
    assert empty.answer == "" and empty.confidence == 0 and empty.evidence_sources == []
