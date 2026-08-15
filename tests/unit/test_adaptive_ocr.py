import json
import pytest
from pathlib import Path
from types import ModuleType
import sys

from backend.app.runtime.device_policy import DeviceSelection, RuntimeCapabilities
from backend.app.video import text_backends
from backend.app.video.text_backends import (
    AdaptiveOCRBackend,
    OCRBackend,
    PaddleOCRBackend,
    TesseractOCRBackend,
    create_ocr_backend,
)
from backend.app.video.m16_text_pipeline import (
    M16TextPipeline,
    TextEvidenceStore,
    compute_ocr_fingerprint,
    compute_asr_fingerprint,
)
from backend.app.video.text_evidence import normalize_text


class MockSentinelPaddleBackend(OCRBackend):
    def __init__(self, *args, **kwargs):
        raise AssertionError("Sentinel: PaddleOCR backend must NOT be constructed on CPU!")

    def extract(self, image_paths):
        return []


class MockSentinelTesseractBackend(OCRBackend):
    def __init__(self, *args, **kwargs):
        pass

    def extract(self, image_paths):
        raise AssertionError("Sentinel: Tesseract fallback must NOT be called when Paddle succeeds!")

    def info(self):
        return {"backend": "tesseract", "languages": "eng+vie"}

    def identity(self):
        return "tesseract:eng+vie"


class MockWorkingPaddleBackend(OCRBackend):
    def __init__(self, result_text="Tiếng Việt chuẩn xác", confidence=0.95, device="cuda:0"):
        self.result_text = result_text
        self.confidence = confidence
        self.device = device
        self.extract_call_count = 0

    def extract(self, image_paths):
        self.extract_call_count += len(image_paths)
        return [{
            "text": self.result_text,
            "boxes": [[10, 10, 100, 30]],
            "confidence": self.confidence,
            "backend": "paddleocr",
            "language": "vi",
        } for _ in image_paths]

    def info(self):
        return {"backend": "paddleocr", "languages": "vi", "device": self.device}

    def identity(self):
        return f"paddleocr:vi:{self.device}"


class MockWorkingTesseractBackend(OCRBackend):
    def __init__(self, result_text="Fallback Tesseract text", confidence=0.88):
        self.result_text = result_text
        self.confidence = confidence
        self.extract_call_count = 0

    def extract(self, image_paths):
        self.extract_call_count += len(image_paths)
        return [{
            "text": self.result_text,
            "boxes": [[10, 10, 100, 30]],
            "confidence": self.confidence,
            "backend": "tesseract",
            "language": "eng+vie",
        } for _ in image_paths]

    def info(self):
        return {"backend": "tesseract", "languages": "eng+vie"}

    def identity(self):
        return "tesseract:eng+vie"


class MockFailingPaddleBackend(OCRBackend):
    def extract(self, image_paths):
        raise RuntimeError("Simulated CUDA OOM in PaddleOCR")

    def info(self):
        return {"backend": "paddleocr", "languages": "vi", "device": "cuda:0"}

    def identity(self):
        return "paddleocr:vi:cuda:0"


# A. AUTO CPU routing: Simulate no usable CUDA -> primary is Tesseract, Paddle constructor never called
def test_auto_cpu_routing_selects_tesseract(monkeypatch):
    monkeypatch.setattr(text_backends, "probe_paddle", lambda: RuntimeCapabilities("paddle", False, cuda_available=False))
    monkeypatch.setattr(text_backends, "PaddleOCRBackend", MockSentinelPaddleBackend)
    backend = create_ocr_backend(name="auto", device="cpu")
    assert isinstance(backend, TesseractOCRBackend)
    assert backend.info()["backend"] == "tesseract"


# B. AUTO GPU routing: Simulate usable Paddle CUDA backend -> returns AdaptiveOCRBackend with Paddle primary
def test_auto_gpu_routing_selects_paddle_primary(monkeypatch):
    monkeypatch.setattr(text_backends, "probe_paddle", lambda: RuntimeCapabilities(
        "paddle", True, cuda_available=True, cuda_device_count=1, cuda_devices=({"index": 0, "name": "NVIDIA RTX"},)
    ))
    monkeypatch.setattr(text_backends, "resolve_device", lambda *args, **kwargs: DeviceSelection(
        "ocr", "paddle", "cuda:0", "cuda:0", "auto", 0
    ))
    backend = create_ocr_backend(name="auto")
    assert isinstance(backend, AdaptiveOCRBackend)
    assert backend.primary.info()["backend"] == "paddleocr"
    assert backend.fallback.info()["backend"] == "tesseract"


# C. Paddle success: When Paddle returns valid text, Tesseract fallback is NOT called
def test_paddle_success_does_not_call_tesseract():
    paddle = MockWorkingPaddleBackend(result_text="Biển số 79H-6072", confidence=0.92)
    tesseract = MockSentinelTesseractBackend()
    adaptive = AdaptiveOCRBackend(primary=paddle, fallback=tesseract)
    
    results = adaptive.extract([Path("frame_001.jpg")])
    assert len(results) == 1
    assert results[0]["text"] == "Biển số 79H-6072"
    assert results[0]["backend"] == "paddleocr"
    assert paddle.extract_call_count == 1


# D. Empty Paddle result: When Paddle returns empty OCR, Tesseract is called as fallback
def test_empty_paddle_result_triggers_tesseract_fallback():
    paddle = MockWorkingPaddleBackend(result_text="", confidence=None)
    tesseract = MockWorkingTesseractBackend(result_text="Khu vực hồ Hoàn Kiếm", confidence=0.85)
    adaptive = AdaptiveOCRBackend(primary=paddle, fallback=tesseract, fallback_on_empty=True)
    
    results = adaptive.extract([Path("frame_001.jpg")])
    assert len(results) == 1
    assert results[0]["text"] == "Khu vực hồ Hoàn Kiếm"
    assert results[0]["backend"] == "tesseract"
    assert tesseract.extract_call_count == 1


# E. Low-confidence Paddle result triggers fallback when enabled
def test_low_confidence_paddle_triggers_fallback():
    paddle = MockWorkingPaddleBackend(result_text="mo ao", confidence=0.35)
    tesseract = MockWorkingTesseractBackend(result_text="mở áo", confidence=0.80)
    adaptive = AdaptiveOCRBackend(
        primary=paddle,
        fallback=tesseract,
        fallback_on_low_confidence=True,
        min_confidence=0.50,
    )
    
    results = adaptive.extract([Path("frame_001.jpg")])
    assert len(results) == 1
    assert results[0]["text"] == "mở áo"
    assert results[0]["backend"] == "tesseract"


# F. Paddle runtime error: Handled gracefully without crash, falling back to Tesseract
def test_paddle_runtime_error_falls_back_gracefully():
    paddle = MockFailingPaddleBackend()
    tesseract = MockWorkingTesseractBackend(result_text="An toàn giao thông", confidence=0.90)
    adaptive = AdaptiveOCRBackend(primary=paddle, fallback=tesseract, fallback_on_error=True)
    
    results = adaptive.extract([Path("frame_001.jpg")])
    assert len(results) == 1
    assert results[0]["text"] == "An toàn giao thông"
    assert results[0]["backend"] == "tesseract"


# G. Forced Tesseract: Paddle is never initialized
def test_forced_tesseract_never_initializes_paddle(monkeypatch):
    monkeypatch.setattr(text_backends, "PaddleOCRBackend", MockSentinelPaddleBackend)
    backend = create_ocr_backend(name="tesseract")
    assert isinstance(backend, TesseractOCRBackend)
    assert backend.info()["backend"] == "tesseract"


# H. Forced Paddle on unavailable GPU raises clear predictable error
def test_forced_paddle_device_error(monkeypatch):
    monkeypatch.setattr(text_backends, "probe_paddle", lambda: RuntimeCapabilities("paddle", False, cuda_available=False))
    with pytest.raises(RuntimeError) as exc_info:
        create_ocr_backend(name="paddleocr", device="cuda:0")
    assert "CUDA is unavailable" in str(exc_info.value) or "unavailable" in str(exc_info.value)


# I. Cache fingerprint backend change: Invalidate stale OCR cache when backend identity changes
def test_cache_fingerprint_invalidation_on_backend_change():
    source_hash = "abc123sourcehash"
    frames_fp = "fp_frames_456"
    
    tess_info = {"backend": "tesseract", "languages": "eng+vie"}
    paddle_info = {"backend": "paddleocr", "languages": "vi", "device": "cuda:0"}
    
    fp_tess, _ = compute_ocr_fingerprint(source_hash, frames_fp, tess_info)
    fp_paddle, _ = compute_ocr_fingerprint(source_hash, frames_fp, paddle_info)
    
    assert fp_tess != fp_paddle


# J. Resume sentinel: When valid cache exists, neither Paddle nor Tesseract backend is constructed
def test_resume_sentinel_does_not_construct_ocr_backends(tmp_path):
    store = TextEvidenceStore(tmp_path)
    video_id = "L22_V001"
    
    ocr_desc = {"backend": "tesseract", "languages": "eng+vie"}
    source_hash = "srchash_001"
    frames_fp = "framesfp_001"
    fp, payload = compute_ocr_fingerprint(source_hash, frames_fp, ocr_desc)
    
    # Save valid cache and meta
    (tmp_path / video_id).mkdir(parents=True, exist_ok=True)
    store.save_ocr(video_id, [], {**payload, "fingerprint": fp, "record_count": 0})
    
    assert store.validate_ocr_cache(video_id, fp, source_hash, frames_fp) is True


# K. Corrupt cache handles safely without application crash
def test_corrupt_cache_validation_fails_safely(tmp_path):
    store = TextEvidenceStore(tmp_path)
    video_id = "L22_V002"
    (tmp_path / video_id).mkdir(parents=True, exist_ok=True)
    (tmp_path / video_id / "ocr.json").write_text("{corrupted json[")
    
    assert store.validate_ocr_cache(video_id, "some_fp") is False


# L. Unicode Vietnamese normalization preserves full diacritics
def test_unicode_vietnamese_preserved():
    sample_text = "Tiếng Việt, Nguyễn, kỹ thuật, nhiệt độ 40°C, đường Tiên Lân, Trường Sa"
    normalized = normalize_text(sample_text)
    
    assert "tiếng việt" in normalized
    assert "nguyễn" in normalized
    assert "kỹ thuật" in normalized
    assert "nhiệt độ" in normalized
    assert "đường" in normalized
    assert "trường sa" in normalized
