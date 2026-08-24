import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from backend.app.core.config import (
    FASTER_WHISPER_MODEL,
    OCR_BACKEND,
    OCR_FALLBACK_ON_EMPTY,
    OCR_FALLBACK_ON_ERROR,
    OCR_FALLBACK_ON_LOW_CONFIDENCE,
    OCR_PADDLE_MIN_CONFIDENCE,
)
from backend.app.runtime.device_policy import probe_paddle, resolve_device


DEFAULT_WHISPER_MODEL = FASTER_WHISPER_MODEL


def resolve_whisper_revision(model_name: str = DEFAULT_WHISPER_MODEL, cache_dir: Path | None = None) -> str:
    try:
        from faster_whisper.utils import download_model
        from backend.app.model_cache import model_cache_dir
        actual_cache = model_cache_dir("faster-whisper", cache_dir)
        kwargs = {"local_files_only": True}
        if actual_cache:
            kwargs["cache_dir"] = str(actual_cache)
        path = download_model(model_name, **kwargs)
        p = Path(path)
        if p.parent.name == "snapshots":
            return p.name
        return str(path)
    except Exception:
        return "default"


class OCRBackend(ABC):
    @abstractmethod
    def extract(self, image_paths):
        pass

    def identity(self):
        return type(self).__name__

    def info(self):
        return {"backend": type(self).__name__}


class TesseractOCRBackend(OCRBackend):
    def __init__(self, languages="eng+vie", executable="tesseract"):
        self.languages = languages
        self.executable = executable

    def identity(self):
        return f"tesseract:{self.languages}"

    def info(self):
        return {
            "backend": "tesseract",
            "languages": self.languages,
            "executable": self.executable,
        }

    def extract(self, image_paths):
        output = []
        for path in image_paths:
            command = [self.executable, str(path), "stdout", "-l", self.languages, "tsv"]
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                raise RuntimeError(f"OCR failed for {Path(path).name}") from exc
            words, boxes, confidences = [], [], []
            lines = result.stdout.splitlines()
            if lines:
                header = lines[0].split("\t")
                for line in lines[1:]:
                    row = dict(zip(header, line.split("\t")))
                    text = row.get("text", "").strip()
                    confidence = float(row.get("conf", -1))
                    if text and confidence >= 0:
                        words.append(text)
                        boxes.append([int(row[key]) for key in ("left", "top", "width", "height")])
                        confidences.append(confidence / 100)
            output.append({
                "text": " ".join(words),
                "boxes": boxes,
                "confidence": sum(confidences) / len(confidences) if confidences else None,
                "backend": "tesseract",
                "languages": self.languages,
            })
        return output


class PaddleOCRBackend(OCRBackend):
    def __init__(self, languages="vi", device=None, min_confidence=0.50, model_dir=None, local_files_only=False):
        self.languages = languages
        self.min_confidence = float(min_confidence)
        self.model_dir = model_dir
        self.local_files_only = local_files_only
        self.device_selection = resolve_device("ocr", "paddle", device, component_env="OCR_DEVICE")
        self._reader = None

    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        import importlib.util
        if importlib.util.find_spec("paddleocr") is None:
            raise RuntimeError("PaddleOCR is not installed. Install paddleocr to use PaddleOCR backend.")
        from paddleocr import PaddleOCR
        use_gpu = self.device_selection.device.startswith("cuda")
        kwargs = {
            "use_angle_cls": True,
            "lang": self.languages,
            "use_gpu": use_gpu,
            "show_log": False,
        }
        if self.model_dir:
            model_path = Path(self.model_dir)
            if (model_path / "det").exists():
                kwargs["det_model_dir"] = str(model_path / "det")
            if (model_path / "rec").exists():
                kwargs["rec_model_dir"] = str(model_path / "rec")
            if (model_path / "cls").exists():
                kwargs["cls_model_dir"] = str(model_path / "cls")
        self._reader = PaddleOCR(**{k: v for k, v in kwargs.items() if v is not None})
        return self._reader

    def identity(self):
        return f"paddleocr:{self.languages}:{self.device_selection.device}"

    def info(self):
        return {
            "backend": "paddleocr",
            "languages": self.languages,
            "device": self.device_selection.device,
            "min_confidence": self.min_confidence,
        }

    def extract(self, image_paths):
        reader = self._get_reader()
        output = []
        for path in image_paths:
            result = reader.ocr(str(path), cls=True)
            words, boxes, confidences = [], [], []
            if result and result[0]:
                for line in result[0]:
                    if not line or len(line) < 2:
                        continue
                    box, text_info = line[0], line[1]
                    txt = text_info[0].strip() if text_info and len(text_info) > 0 else ""
                    conf = float(text_info[1]) if text_info and len(text_info) > 1 else 0.0
                    if txt:
                        words.append(txt)
                        boxes.append(box)
                        confidences.append(conf)
            avg_conf = sum(confidences) / len(confidences) if confidences else None
            output.append({
                "text": " ".join(words),
                "boxes": boxes,
                "confidence": avg_conf,
                "backend": "paddleocr",
                "languages": self.languages,
            })
        return output


class AdaptiveOCRBackend(OCRBackend):
    def __init__(
        self,
        primary: OCRBackend,
        fallback: OCRBackend,
        fallback_on_empty: bool = True,
        fallback_on_error: bool = True,
        fallback_on_low_confidence: bool = True,
        min_confidence: float = 0.50,
        circuit_breaker_threshold: int = 5,
    ):
        self.primary = primary
        self.fallback = fallback
        self.fallback_on_empty = fallback_on_empty
        self.fallback_on_error = fallback_on_error
        self.fallback_on_low_confidence = fallback_on_low_confidence
        self.min_confidence = min_confidence
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self._consecutive_failures = 0
        self._tripped = False

    def identity(self):
        return f"adaptive:{self.primary.identity()}=>{self.fallback.identity()}"

    def info(self):
        return {
            "backend": "adaptive",
            "primary": self.primary.info(),
            "fallback": self.fallback.info(),
            "routing_mode": "auto",
            "fallback_on_empty": self.fallback_on_empty,
            "fallback_on_error": self.fallback_on_error,
            "fallback_on_low_confidence": self.fallback_on_low_confidence,
            "min_confidence": self.min_confidence,
        }

    def extract(self, image_paths):
        output = []
        for path in image_paths:
            if self._tripped:
                res = self.fallback.extract([path])[0]
                output.append(res)
                continue

            try:
                primary_res = self.primary.extract([path])[0]
                self._consecutive_failures = 0

                text = primary_res.get("text", "").strip()
                conf = primary_res.get("confidence")

                # Check if primary result is empty
                if not text and self.fallback_on_empty:
                    fallback_res = self.fallback.extract([path])[0]
                    if fallback_res.get("text", "").strip():
                        output.append(fallback_res)
                        continue

                # Check if confidence is below threshold
                if self.fallback_on_low_confidence and conf is not None and conf < self.min_confidence:
                    fallback_res = self.fallback.extract([path])[0]
                    output.append(fallback_res)
                    continue

                output.append(primary_res)
            except Exception:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.circuit_breaker_threshold:
                    self._tripped = True
                if self.fallback_on_error:
                    fallback_res = self.fallback.extract([path])[0]
                    output.append(fallback_res)
                else:
                    raise
        return output


class EasyOCRBackend(OCRBackend):
    def __init__(self, languages=("en", "vi"), device=None):
        import easyocr
        self.languages = list(languages)
        self.device_selection = resolve_device("ocr", "torch", device,
            component_env="OCR_DEVICE")
        self.reader = easyocr.Reader(self.languages, gpu=self.device_selection.device.startswith("cuda"))

    def identity(self):
        return f"easyocr:{','.join(self.languages)}"

    def info(self):
        return {
            "backend": "easyocr",
            "languages": self.languages,
            "device": self.device_selection.device,
        }

    def extract(self, image_paths):
        output = []
        for path in image_paths:
            results = self.reader.readtext(str(path))
            output.append({
                "text": " ".join(item[1] for item in results),
                "boxes": [item[0] for item in results],
                "confidence": sum(item[2] for item in results) / len(results) if results else None,
                "backend": "easyocr",
                "languages": self.languages,
            })
        return output


def create_ocr_backend(name=None, device=None, languages=None):
    name = (name or OCR_BACKEND).lower()
    if name == "auto":
        paddle_caps = probe_paddle()
        if paddle_caps.installed and paddle_caps.cuda_available and device not in ("cpu", "unavailable"):
            primary = PaddleOCRBackend(languages="vi", device=device or "cuda:0", min_confidence=OCR_PADDLE_MIN_CONFIDENCE)
            fallback = TesseractOCRBackend(languages=languages or "eng+vie")
            return AdaptiveOCRBackend(
                primary=primary,
                fallback=fallback,
                fallback_on_empty=OCR_FALLBACK_ON_EMPTY,
                fallback_on_error=OCR_FALLBACK_ON_ERROR,
                fallback_on_low_confidence=OCR_FALLBACK_ON_LOW_CONFIDENCE,
                min_confidence=OCR_PADDLE_MIN_CONFIDENCE,
            )
        else:
            langs = languages or "eng+vie"
            return TesseractOCRBackend(languages=langs)

    if name == "tesseract":
        if device not in (None, "auto", "cpu"):
            raise RuntimeError("Tesseract is CPU-only")
        langs = languages or "eng+vie"
        return TesseractOCRBackend(languages=langs)

    if name == "paddleocr":
        return PaddleOCRBackend(languages=languages or "vi", device=device, min_confidence=OCR_PADDLE_MIN_CONFIDENCE)

    if name == "easyocr":
        langs = tuple(languages) if languages else ("en", "vi")
        return EasyOCRBackend(languages=langs, device=device)

    raise ValueError(f"unsupported OCR backend {name!r}; use auto, tesseract, paddleocr, or easyocr")


class ASRBackend(ABC):
    @abstractmethod
    def transcribe(self, video_path):
        pass

    def identity(self):
        return type(self).__name__

    def info(self):
        return {"backend": type(self).__name__}


class FasterWhisperASRBackend(ASRBackend):
    def __init__(self, model_name=DEFAULT_WHISPER_MODEL, device=None, compute_type=None,
            cache_dir=None, local_files_only=False, revision=None):
        from faster_whisper import WhisperModel
        from backend.app.model_cache import model_cache_dir
        self.model_name = model_name
        self.device_selection = resolve_device("asr", "ctranslate2", device,
            component_env="ASR_DEVICE")
        self.compute_type = compute_type or ("float16" if self.device_selection.device.startswith("cuda") else "int8")
        self.cache_dir = cache_dir
        self.revision = revision or resolve_whisper_revision(self.model_name, self.cache_dir)
        download_root = model_cache_dir("faster-whisper", cache_dir)
        kwargs = {"device": "cuda" if self.device_selection.device.startswith("cuda") else "cpu",
            "compute_type": self.compute_type,
            "download_root": str(download_root) if download_root else None,
            "local_files_only": local_files_only}
        if self.device_selection.device_index is not None:
            kwargs["device_index"] = self.device_selection.device_index
        try:
            self.model = WhisperModel(model_name, **kwargs)
        except Exception as exc:
            if local_files_only:
                raise RuntimeError(f"Faster Whisper model {model_name} is not available locally. Run: python projectctl.py models --prepare --asr") from exc
            raise

    def identity(self):
        return f"faster-whisper:{self.model_name}:{self.compute_type}:{self.revision}"

    def info(self):
        return {
            "backend": "faster-whisper",
            "model": self.model_name,
            "revision": self.revision,
            "requested_device": self.device_selection.requested,
            "device": self.device_selection.device,
            "device_source": self.device_selection.source,
            "compute_type": self.compute_type,
            "fallback": self.device_selection.fallback,
        }

    def transcribe(self, video_path):
        segments, info = self.model.transcribe(str(video_path), vad_filter=True)
        return [{"start_seconds": segment.start, "end_seconds": segment.end,
            "text": segment.text, "language": info.language,
            "confidence": None if segment.avg_logprob is None else float(segment.avg_logprob)}
            for segment in segments]


class SidecarASRBackend(ASRBackend):
    def transcribe(self, video_path):
        path = Path(video_path).with_suffix(".asr.json")
        return json.loads(path.read_text()) if path.is_file() else []

    def identity(self):
        return "sidecar-asr"

    def info(self):
        return {"backend": "sidecar-asr"}
