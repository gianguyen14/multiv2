import sys
from types import ModuleType

from backend.app.runtime.device_policy import RuntimeCapabilities
from backend.app.video import text_backends


def test_tesseract_is_default_and_cpu():
    assert isinstance(text_backends.create_ocr_backend(), text_backends.TesseractOCRBackend)
    try:
        text_backends.create_ocr_backend("tesseract", "cuda")
    except RuntimeError as exc:
        assert "CPU-only" in str(exc)
    else:
        raise AssertionError("CUDA Tesseract must fail")


def test_faster_whisper_uses_cuda_index(monkeypatch):
    calls = []
    package = ModuleType("faster_whisper")
    package.WhisperModel = lambda *args, **kwargs: calls.append((args, kwargs)) or object()
    monkeypatch.setitem(sys.modules, "faster_whisper", package)
    monkeypatch.setattr(text_backends, "resolve_device", lambda *args, **kwargs:
        __import__("backend.app.runtime.device_policy", fromlist=["DeviceSelection"]).DeviceSelection(
            "asr", "ctranslate2", "cuda:1", "cuda:1", "argument", 1))
    backend = text_backends.FasterWhisperASRBackend(device="cuda:1")
    assert calls[0][1]["device"] == "cuda"
    assert calls[0][1]["device_index"] == 1
    assert calls[0][1]["compute_type"] == "float16"
    assert backend.info()["device"] == "cuda:1"


def test_easyocr_gpu_flag_comes_from_policy(monkeypatch):
    calls = []
    package = ModuleType("easyocr")
    package.Reader = lambda languages, gpu: calls.append((languages, gpu)) or object()
    monkeypatch.setitem(sys.modules, "easyocr", package)
    monkeypatch.setattr(text_backends, "resolve_device", lambda *args, **kwargs:
        __import__("backend.app.runtime.device_policy", fromlist=["DeviceSelection"]).DeviceSelection(
            "ocr", "torch", "auto", "cpu", "default", fallback=True))
    text_backends.create_ocr_backend("easyocr")
    assert calls == [(["en", "vi"], False)]
