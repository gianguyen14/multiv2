from backend.app.retrieval.ranking_metrics import mrr_at_k, recall_at_k
from backend.app.retrieval.video_multimodal import MultimodalVideoRetriever
from backend.app.video.frame_record import FrameRecord
from backend.app.video.text_evidence import ASRSegment, OCRRecord


def frame(video_id, index):
    return FrameRecord.create(video_id=video_id, source_frame_index_zero_based=index,
        submission_frame_id=index, timestamp_seconds=float(index), pts=index, width=10, height=10,
        image_path=f"{video_id}-{index}.jpg", sample_interval_seconds=1, ingestion_version="test")


def test_each_modality_and_fusion_retrieve_their_evidence():
    ocr = [OCRRecord.create(frame("ocr_video", 10), "HCMC 2026", confidence=1)]
    asr = [ASRSegment.create("asr_video", 0, 2, 3, 20, 30, "spoken secret phrase", "en", 1)]

    def visual_search(query, top_k):
        return [{"score": 0.9, "payload": frame("visual_video", 5).to_dict()}]

    retriever = MultimodalVideoRetriever(visual_search, ocr, asr)
    assert retriever.search("red shirt", modalities=("visual",))[0].video_id == "visual_video"
    assert retriever.search("HCMC 2026", modalities=("ocr",))[0].video_id == "ocr_video"
    assert retriever.search("spoken secret", modalities=("asr",))[0].video_id == "asr_video"
    fused = retriever.search("HCMC spoken", modalities=("visual", "ocr", "asr"))
    assert {item.video_id for item in fused} == {"visual_video", "ocr_video", "asr_video"}
    assert all(0 <= item.fused_score <= 1 for item in fused)


def test_multimodal_evaluation_has_complete_candidate_coverage():
    records = [OCRRecord.create(frame("v", 1), "screen code", confidence=1)]
    segments = [ASRSegment.create("v", 0, 1, 2, 2, 3, "spoken clue", "en", 1)]
    retriever = MultimodalVideoRetriever(lambda query, top_k: [], records, segments)
    ranked = retriever.search("screen code", modalities=("ocr", "asr"))
    predictions = [candidate.video_id for candidate in ranked]
    assert recall_at_k(predictions, ["v"], 1) == 1
    assert mrr_at_k(predictions, ["v"], 1) == 1
