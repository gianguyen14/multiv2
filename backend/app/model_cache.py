import os
from pathlib import Path


def model_cache_dir(provider, explicit=None):
    if explicit:
        return Path(explicit)
    root = os.getenv("MODEL_CACHE_DIR")
    return Path(root) / provider if root else None


def visual_model_name():
    from backend.app.core.config import SIGLIP2_MODEL
    return SIGLIP2_MODEL


def asr_model_name():
    from backend.app.core.config import FASTER_WHISPER_MODEL
    return FASTER_WHISPER_MODEL


def _status(backend, model, path=None, error=None):
    return {
        "backend": backend,
        "model": model,
        "cached": path is not None,
        "cache_path": str(path) if path else None,
        **({"error": error} if error else {}),
    }


def visual_status():
    model = visual_model_name()
    cache = model_cache_dir("huggingface")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=model, cache_dir=cache, local_files_only=True)
        return _status("siglip2", model, path)
    except Exception as exc:
        return _status("siglip2", model, error=str(exc))


def _download_asr(model, local_files_only):
    from faster_whisper.utils import download_model
    kwargs = {"local_files_only": local_files_only}
    cache = model_cache_dir("faster-whisper")
    if cache:
        kwargs["cache_dir"] = str(cache)
    return download_model(model, **kwargs)


def asr_status(model=None):
    model = model or asr_model_name()
    try:
        return _status("faster-whisper", model, _download_asr(model, True))
    except Exception as exc:
        return _status("faster-whisper", model, error=str(exc))


def paddleocr_status():
    import importlib.util
    if importlib.util.find_spec("paddleocr") is None:
        return _status("paddleocr", "PP-OCRv4-vi", error="paddleocr is not installed; Tesseract OCR is active fallback")
    cache = model_cache_dir("paddleocr")
    # Check default paddleocr paths
    default_dir = Path.home() / ".paddleocr"
    target_dir = cache or default_dir
    if target_dir.exists() and any(target_dir.iterdir()):
        return _status("paddleocr", "PP-OCRv4-vi", path=target_dir)
    return _status("paddleocr", "PP-OCRv4-vi", error="PaddleOCR models not downloaded")


def prepare_paddleocr(lang="vi"):
    import importlib.util
    if importlib.util.find_spec("paddleocr") is None:
        raise RuntimeError("paddleocr is not installed; install paddleocr to prepare OCR models")
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False, show_log=False)


def prepare_visual():
    from huggingface_hub import snapshot_download
    cache = model_cache_dir("huggingface")
    return snapshot_download(repo_id=visual_model_name(), cache_dir=cache)


def prepare_asr(model=None):
    return _download_asr(model or asr_model_name(), False)


def query_refiner_model_name():
    from backend.app.core.config import QUERY_REFINER_MODEL
    return QUERY_REFINER_MODEL


def query_refiner_status():
    model = query_refiner_model_name()
    cache = model_cache_dir("huggingface")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=model, cache_dir=cache, local_files_only=True)
        return _status("query_refiner", model, path)
    except Exception as exc:
        return _status("query_refiner", model, error=str(exc))


def prepare_query_refiner():
    from huggingface_hub import snapshot_download
    cache = model_cache_dir("huggingface")
    return snapshot_download(repo_id=query_refiner_model_name(), cache_dir=cache)


def inventory(whisper_model=None, tesseract_languages=()):
    from backend.app.core.config import OCR_BACKEND
    return {
        "visual": visual_status(),
        "asr": asr_status(whisper_model),
        "query_refiner": query_refiner_status(),
        "ocr": {
            "backend": OCR_BACKEND,
            "tesseract": {
                "local_weights": "system language packs",
                "eng": "eng" in tesseract_languages,
                "vie": "vie" in tesseract_languages,
            },
            "paddleocr": paddleocr_status(),
        },
    }

