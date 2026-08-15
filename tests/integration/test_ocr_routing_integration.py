import json
from pathlib import Path
import pytest

from backend.app.video.frame_store import FrameRecord, FrameStore
from backend.app.video.ingest_manifest import IngestManifest
from backend.app.video.m16_text_pipeline import (
    M16TextPipeline,
    TextEvidenceStore,
    compute_ocr_fingerprint,
    compute_asr_fingerprint,
)
from backend.app.video.text_backends import (
    AdaptiveOCRBackend,
    OCRBackend,
    TesseractOCRBackend,
)


class MockIntegrationPrimaryPaddle(OCRBackend):
    def __init__(self, texts=None):
        self.texts = texts or ["TRƯỜNG ĐẠI HỌC BÁCH KHOA"]
        self.call_count = 0

    def extract(self, image_paths):
        self.call_count += len(image_paths)
        return [{
            "text": self.texts[i % len(self.texts)],
            "boxes": [[0, 0, 100, 50]],
            "confidence": 0.96,
            "backend": "paddleocr",
            "language": "vi",
        } for i, _ in enumerate(image_paths)]

    def info(self):
        return {"backend": "paddleocr", "languages": "vi", "device": "cuda:0"}

    def identity(self):
        return "paddleocr:vi:cuda:0"


class MockIntegrationFallbackTesseract(OCRBackend):
    def __init__(self, texts=None):
        self.texts = texts or ["FALLBACK TEXT"]
        self.call_count = 0

    def extract(self, image_paths):
        self.call_count += len(image_paths)
        return [{
            "text": self.texts[i % len(self.texts)],
            "boxes": [[0, 0, 100, 50]],
            "confidence": 0.85,
            "backend": "tesseract",
            "language": "eng+vie",
        } for i, _ in enumerate(image_paths)]

    def info(self):
        return {"backend": "tesseract", "languages": "eng+vie"}

    def identity(self):
        return "tesseract:eng+vie"


class MockIntegrationFailingPaddle(OCRBackend):
    def extract(self, image_paths):
        raise RuntimeError("Paddle GPU failed")

    def info(self):
        return {"backend": "paddleocr", "languages": "vi", "device": "cuda:0"}

    def identity(self):
        return "paddleocr:vi:cuda:0"


def test_adaptive_ocr_pipeline_integration_primary_success(tmp_path):
    root = tmp_path / "processed"
    video_id = "L22_V001"
    v_dir = root / video_id
    v_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = v_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy frame image
    frame_path = frames_dir / "000000000.jpg"
    frame_path.write_bytes(b"dummy image data")
    
    dummy_video_file = tmp_path / f"{video_id}.mp4"
    dummy_video_file.write_bytes(b"dummy video")
    
    frame_store = FrameStore(root)
    manifest = IngestManifest(
        video_id=video_id,
        source_path=str(dummy_video_file),
        source_size=1000,
        source_mtime_ns=1000,
        source_hash="srchash_v001",
        ingestion_version="1.0",
        schema_version=2,
        status="frames_ready",
        completed_stage="frames",
        frames_fingerprint="frames_fp_v001",
        sampled_frame_count=1,
    )
    frame_store.save_manifest(manifest)
    record = FrameRecord.create(
        video_id=video_id,
        source_frame_index_zero_based=0,
        submission_frame_id=0,
        image_path=str(frame_path),
        timestamp_seconds=0.0,
        pts=0,
        width=1920,
        height=1080,
        sample_interval_seconds=1.0,
        ingestion_version="1.0",
    )
    frame_store.save_records(video_id, [record])
    
    primary = MockIntegrationPrimaryPaddle(texts=["UBND THÀNH PHỐ HỒ CHÍ MINH"])
    fallback = MockIntegrationFallbackTesseract()
    adaptive = AdaptiveOCRBackend(primary=primary, fallback=fallback)
    
    asr_dummy = type("DummyASR", (), {
        "transcribe": lambda self, path: [],
        "info": lambda self: {"backend": "dummy"},
        "identity": lambda self: "dummy",
    })()
    
    pipeline = M16TextPipeline(root, adaptive, asr_dummy, use_ocr=True, use_asr=False)
    
    res = pipeline.run_video(dummy_video_file)
    assert res["status"] == "completed"
    assert res["ocr_count"] == 1
    assert primary.call_count == 1
    assert fallback.call_count == 0
    
    store = TextEvidenceStore(root)
    records = store.load_ocr(video_id)
    assert len(records) == 1
    assert "ubnd thành phố hồ chí minh" in records[0].normalized_text
    
    meta = store.load_ocr_meta(video_id)
    assert meta is not None
    assert meta["ocr_backend"]["backend"] == "adaptive"
    assert meta["ocr_backend"]["primary"]["backend"] == "paddleocr"
    
    # Test resume: running pipeline again should resume without calling extract
    res_resume = pipeline.run_video(dummy_video_file)
    assert res_resume["status"] == "resumed"
    assert primary.call_count == 1  # Not incremented!
    assert fallback.call_count == 0


def test_adaptive_ocr_pipeline_integration_fallback_on_primary_error(tmp_path):
    root = tmp_path / "processed"
    video_id = "L22_V002"
    v_dir = root / video_id
    v_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = v_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    frame_path = frames_dir / "000000000.jpg"
    frame_path.write_bytes(b"dummy image data")
    
    dummy_video_file = tmp_path / f"{video_id}.mp4"
    dummy_video_file.write_bytes(b"dummy video")
    
    frame_store = FrameStore(root)
    manifest = IngestManifest(
        video_id=video_id,
        source_path=str(dummy_video_file),
        source_size=1000,
        source_mtime_ns=1000,
        source_hash="srchash_v002",
        ingestion_version="1.0",
        schema_version=2,
        status="frames_ready",
        completed_stage="frames",
        frames_fingerprint="frames_fp_v002",
        sampled_frame_count=1,
    )
    frame_store.save_manifest(manifest)
    record = FrameRecord.create(
        video_id=video_id,
        source_frame_index_zero_based=0,
        submission_frame_id=0,
        image_path=str(frame_path),
        timestamp_seconds=0.0,
        pts=0,
        width=1920,
        height=1080,
        sample_interval_seconds=1.0,
        ingestion_version="1.0",
    )
    frame_store.save_records(video_id, [record])
    
    primary = MockIntegrationFailingPaddle()
    fallback = MockIntegrationFallbackTesseract(texts=["CỬA HÀNG TIỆN LỢI"])
    adaptive = AdaptiveOCRBackend(primary=primary, fallback=fallback, fallback_on_error=True)
    
    asr_dummy = type("DummyASR", (), {
        "transcribe": lambda self, path: [],
        "info": lambda self: {"backend": "dummy"},
        "identity": lambda self: "dummy",
    })()
    
    pipeline = M16TextPipeline(root, adaptive, asr_dummy, use_ocr=True, use_asr=False)
    
    res = pipeline.run_video(dummy_video_file)
    assert res["status"] == "completed"
    assert res["ocr_count"] == 1
    assert fallback.call_count == 1
    
    store = TextEvidenceStore(root)
    records = store.load_ocr(video_id)
    assert len(records) == 1
    assert "cửa hàng tiện lợi" in records[0].normalized_text
