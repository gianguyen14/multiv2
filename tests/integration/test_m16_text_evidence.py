import numpy as np

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.video.m15_ingestion_pipeline import VideoIngestionPipeline
from backend.app.video.m16_text_pipeline import M16TextPipeline
from backend.app.video.text_evidence import normalize_text
from tests.m15_support import MeanRGBEncoder, create_video


class FakeOCR:
    def __init__(self, backend="tesseract", languages="eng+vie"):
        self.calls = 0
        self.backend = backend
        self.languages = languages

    def identity(self):
        return f"{self.backend}:{self.languages}"

    def info(self):
        return {"backend": self.backend, "languages": self.languages}

    def extract(self, image_paths):
        self.calls += 1
        return [
            {"text": "  HCMC  2026  ", "boxes": [[1, 2, 3, 4]], "confidence": 0.9},
            {"text": "HCMC 2026", "boxes": [], "confidence": 0.8},
        ]


class FakeASR:
    def __init__(self, segments=None, model="small", compute_type="int8", revision="rev1"):
        self.calls = 0
        self.segments = segments or []
        self.model = model
        self.compute_type = compute_type
        self.revision = revision

    def identity(self):
        return f"faster-whisper:{self.model}:{self.compute_type}:{self.revision}"

    def info(self):
        return {"backend": "faster-whisper", "model": self.model, "compute_type": self.compute_type, "revision": self.revision}

    def transcribe(self, video_path):
        self.calls += 1
        return self.segments



def ingest(video, root):
    VideoIngestionPipeline(MeanRGBEncoder(), VideoIngestConfig(processed_root=root)).ingest_video(video)


def test_ocr_asr_persistence_mapping_and_resume(tmp_path):
    video = tmp_path / "mixed.mp4"
    create_video(video, "red_car")
    root = tmp_path / "processed"
    ingest(video, root)
    ocr = FakeOCR()
    asr = FakeASR([{"start_seconds": 0.26, "end_seconds": 1.24,
        "text": "  Xin  CHÀO world ", "language": "vi", "confidence": 0.7}])
    pipeline = M16TextPipeline(root, ocr, asr)

    result = pipeline.run_video(video)
    assert result == {"video_id": "mixed", "status": "completed", "ocr_count": 1, "asr_count": 1}
    ocr_record = pipeline.store.load_ocr("mixed")[0]
    assert ocr_record.raw_text == "  HCMC  2026  "
    assert ocr_record.normalized_text == "hcmc 2026"
    assert ocr_record.frame_uid == "mixed:000000000"
    segment = pipeline.store.load_asr("mixed")[0]
    assert segment.raw_text == "  Xin  CHÀO world "
    assert segment.normalized_text == "xin chào world"
    assert (segment.start_frame, segment.end_frame) == (1, 5)

    assert pipeline.run_video(video)["status"] == "resumed"
    assert (ocr.calls, asr.calls) == (1, 1)


def test_modality_resume_does_not_suppress_later_asr(tmp_path):
    video = tmp_path / "mixed.mp4"
    create_video(video, "red_car")
    root = tmp_path / "processed"
    ingest(video, root)
    ocr = FakeOCR()
    asr = FakeASR([{"start_seconds": 0.1, "end_seconds": 0.2,
        "text": "speech", "language": "en", "confidence": 0.8}])

    ocr_only = M16TextPipeline(root, ocr, asr, use_ocr=True, use_asr=False)
    assert ocr_only.run_video(video)["status"] == "completed"
    assert ocr.calls == 1 and asr.calls == 0
    assert not ocr_only.store._path("mixed", "asr.json").exists()

    asr_only = M16TextPipeline(root, ocr, asr, use_ocr=False, use_asr=True)
    result = asr_only.run_video(video)
    assert result["status"] == "completed" and result["asr_count"] == 1
    assert ocr.calls == 1 and asr.calls == 1
    assert asr_only.run_video(video)["status"] == "resumed"
    assert asr.calls == 1


def test_silent_no_text_and_failure_isolation(tmp_path):
    valid = tmp_path / "valid.mp4"
    create_video(valid, "blue_object")
    root = tmp_path / "processed"
    ingest(valid, root)
    pipeline = M16TextPipeline(root, type("EmptyOCR", (), {"extract": lambda self, paths: [
        {"text": "", "boxes": [], "confidence": None} for _ in paths]})(), FakeASR())
    report = pipeline.run_path([valid, tmp_path / "missing.mp4"])
    assert report["succeeded"] == 1 and report["failed"] == 1
    assert pipeline.store.load_ocr("valid") == []
    assert pipeline.store.load_asr("valid") == []


def test_unicode_normalization_preserves_vietnamese():
    assert normalize_text("  Xin\tchào  ") == "xin chào"


def test_ocr_asr_fingerprint_invalidation_and_selective_rerun(tmp_path):
    video = tmp_path / "mixed.mp4"
    create_video(video, "red_car")
    root = tmp_path / "processed"
    ingest(video, root)

    ocr_1 = FakeOCR(backend="tesseract", languages="eng+vie")
    asr_1 = FakeASR(model="small", compute_type="int8", revision="rev1")
    p1 = M16TextPipeline(root, ocr_1, asr_1)
    res1 = p1.run_video(video)
    assert res1["status"] == "completed"
    assert (ocr_1.calls, asr_1.calls) == (1, 1)

    # 1. Unchanged config -> both reused
    ocr_2 = FakeOCR(backend="tesseract", languages="eng+vie")
    asr_2 = FakeASR(model="small", compute_type="int8", revision="rev1")
    p2 = M16TextPipeline(root, ocr_2, asr_2)
    res2 = p2.run_video(video)
    assert res2["status"] == "resumed"
    assert (ocr_2.calls, asr_2.calls) == (0, 0)

    # 2. Changed OCR language -> OCR invalidated and rerun, ASR reused
    ocr_3 = FakeOCR(backend="tesseract", languages="eng")
    asr_3 = FakeASR(model="small", compute_type="int8", revision="rev1")
    p3 = M16TextPipeline(root, ocr_3, asr_3)
    res3 = p3.run_video(video)
    assert res3["status"] == "completed"
    assert (ocr_3.calls, asr_3.calls) == (1, 0)

    # 3. Changed OCR backend -> OCR invalidated and rerun
    ocr_4 = FakeOCR(backend="easyocr", languages="en+vi")
    asr_4 = FakeASR(model="small", compute_type="int8", revision="rev1")
    p4 = M16TextPipeline(root, ocr_4, asr_4)
    res4 = p4.run_video(video)
    assert res4["status"] == "completed"
    assert (ocr_4.calls, asr_4.calls) == (1, 0)

    # 4. Changed ASR model/revision -> ASR invalidated and rerun, OCR reused
    ocr_5 = FakeOCR(backend="easyocr", languages="en+vi")
    asr_5 = FakeASR(model="medium", compute_type="int8", revision="rev2")
    p5 = M16TextPipeline(root, ocr_5, asr_5)
    res5 = p5.run_video(video)
    assert res5["status"] == "completed"
    assert (ocr_5.calls, asr_5.calls) == (0, 1)

    # 5. Changed ASR compute type -> ASR invalidated and rerun
    ocr_6 = FakeOCR(backend="easyocr", languages="en+vi")
    asr_6 = FakeASR(model="medium", compute_type="float16", revision="rev2")
    p6 = M16TextPipeline(root, ocr_6, asr_6)
    res6 = p6.run_video(video)
    assert res6["status"] == "completed"
    assert (ocr_6.calls, asr_6.calls) == (0, 1)


def test_corrupt_ocr_asr_json_safely_invalidates(tmp_path):
    video = tmp_path / "mixed.mp4"
    create_video(video, "red_car")
    root = tmp_path / "processed"
    ingest(video, root)

    ocr = FakeOCR()
    asr = FakeASR()
    pipeline = M16TextPipeline(root, ocr, asr)
    pipeline.run_video(video)

    # Corrupt OCR JSON
    ocr_path = root / "mixed" / "ocr.json"
    ocr_path.write_text("{corrupted_json_syntax")
    ocr_new = FakeOCR()
    asr_new = FakeASR()
    p_rec = M16TextPipeline(root, ocr_new, asr_new)
    res = p_rec.run_video(video)
    assert res["status"] == "completed"
    assert ocr_new.calls == 1
    assert asr_new.calls == 0

    # Corrupt ASR JSON
    asr_path = root / "mixed" / "asr.json"
    asr_path.write_text("[not a valid segment dict]")
    asr_path.write_text("{\"not\": \"a list\"}")
    ocr_new2 = FakeOCR()
    asr_new2 = FakeASR()
    p_rec2 = M16TextPipeline(root, ocr_new2, asr_new2)
    res2 = p_rec2.run_video(video)
    assert res2["status"] == "completed"
    assert ocr_new2.calls == 0
    assert asr_new2.calls == 1


def test_legacy_ocr_asr_cache_without_meta_is_safely_accepted(tmp_path):
    video = tmp_path / "mixed.mp4"
    create_video(video, "red_car")
    root = tmp_path / "processed"
    ingest(video, root)

    ocr = FakeOCR()
    asr = FakeASR()
    pipeline = M16TextPipeline(root, ocr, asr)
    pipeline.run_video(video)

    # Remove meta files to simulate legacy artifacts
    ocr_meta = root / "mixed" / "ocr_meta.json"
    asr_meta = root / "mixed" / "asr_meta.json"
    if ocr_meta.exists():
        ocr_meta.unlink()
    if asr_meta.exists():
        asr_meta.unlink()

    # Legacy cache is safely accepted when manifest matches
    ocr_legacy = FakeOCR()
    asr_legacy = FakeASR()
    p_legacy = M16TextPipeline(root, ocr_legacy, asr_legacy)
    res = p_legacy.run_video(video)
    assert res["status"] == "resumed"
    assert (ocr_legacy.calls, asr_legacy.calls) == (0, 0)

