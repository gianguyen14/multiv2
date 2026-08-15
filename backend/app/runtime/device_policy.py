import importlib.util
import os
import re
import shutil
from dataclasses import asdict, dataclass


_DEVICE = re.compile(r"^(auto|cpu|cuda(?::([0-9]+))?)$")


@dataclass(frozen=True)
class RuntimeCapabilities:
    backend: str
    installed: bool
    cpu_available: bool = True
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_devices: tuple = ()
    compute_types: tuple = ()
    cudnn_version: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class DeviceSelection:
    component: str
    backend: str
    requested: str
    device: str
    source: str
    device_index: int | None = None
    fallback: bool = False
    reason: str | None = None


def normalize_device(value):
    value = "auto" if value is None else str(value).strip().lower()
    match = _DEVICE.fullmatch(value)
    if not match:
        raise ValueError(f"unsupported compute device {value!r}; use auto, cpu, cuda, or cuda:N")
    return value


def device_request(override=None, component_env=None, compatibility_env=None, environ=None):
    environ = os.environ if environ is None else environ
    if override is not None:
        return normalize_device(override), "argument"
    for name in (compatibility_env, component_env, "COMPUTE_DEVICE"):
        if name and name in environ:
            return normalize_device(environ[name]), name
    return "auto", "default"


def probe_torch():
    if importlib.util.find_spec("torch") is None:
        return RuntimeCapabilities("torch", False, error="torch is not installed")
    try:
        import torch
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        devices = []
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            total = getattr(properties, "total_memory", None)
            capability = torch.cuda.get_device_capability(index)
            devices.append({"index": index, "name": properties.name,
                "compute_capability": list(capability), "total_vram_bytes": total})
        cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        return RuntimeCapabilities("torch", True, cuda_available=available,
            cuda_device_count=count, cuda_devices=tuple(devices),
            cudnn_version=str(cudnn) if cudnn else None)
    except Exception as exc:
        return RuntimeCapabilities("torch", True, error=str(exc))


def probe_ctranslate2():
    if importlib.util.find_spec("ctranslate2") is None:
        return RuntimeCapabilities("ctranslate2", False, error="ctranslate2 is not installed")
    try:
        import ctranslate2
        count = int(ctranslate2.get_cuda_device_count())
        cpu_types = tuple(sorted(ctranslate2.get_supported_compute_types("cpu")))
        cuda_types = tuple(sorted(ctranslate2.get_supported_compute_types("cuda"))) if count else ()
        return RuntimeCapabilities("ctranslate2", True, cuda_available=count > 0,
            cuda_device_count=count, compute_types=(cpu_types, cuda_types))
    except Exception as exc:
        return RuntimeCapabilities("ctranslate2", True, error=str(exc))


def probe_paddle():
    if importlib.util.find_spec("paddle") is None:
        return RuntimeCapabilities("paddle", False, error="paddle is not installed")
    try:
        import paddle
        available = bool(paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)
        count = int(paddle.device.cuda.device_count()) if available else 0
        devices = []
        if available:
            for index in range(count):
                name = getattr(paddle.device.cuda, "get_device_name", lambda idx: f"cuda:{idx}")(index)
                devices.append({"index": index, "name": str(name)})
        return RuntimeCapabilities("paddle", True, cuda_available=available,
            cuda_device_count=count, cuda_devices=tuple(devices))
    except Exception as exc:
        return RuntimeCapabilities("paddle", True, error=str(exc))


def resolve_device(component, backend, override=None, component_env=None,
        compatibility_env=None, capabilities=None, environ=None):
    requested, source = device_request(override, component_env, compatibility_env, environ)
    if backend == "cpu":
        if requested.startswith("cuda"):
            raise RuntimeError(f"{component} is CPU-only and cannot use {requested}")
        return DeviceSelection(component, backend, requested, "cpu", source)
    if capabilities is None:
        if backend == "torch":
            capabilities = probe_torch()
        elif backend == "paddle":
            capabilities = probe_paddle()
        else:
            capabilities = probe_ctranslate2()
    if requested == "cpu":
        if not capabilities.installed:
            raise RuntimeError(f"{component} runtime unavailable: {capabilities.error}")
        return DeviceSelection(component, backend, requested, "cpu", source)
    index = int(requested.split(":", 1)[1]) if ":" in requested else 0
    if requested.startswith("cuda"):
        if not capabilities.cuda_available:
            raise RuntimeError(f"{component} requested {requested} but {backend} CUDA is unavailable")
        if index >= capabilities.cuda_device_count:
            raise RuntimeError(f"{component} requested cuda:{index} but only {capabilities.cuda_device_count} CUDA device(s) are available")
        return DeviceSelection(component, backend, requested, f"cuda:{index}", source, index)
    if capabilities.cuda_available:
        return DeviceSelection(component, backend, requested, "cuda:0", source, 0)
    return DeviceSelection(component, backend, requested, "cpu", source,
        fallback=True, reason=capabilities.error or f"{backend} CUDA unavailable")


def _diagnostic_selection(component, backend, capabilities, component_env=None,
        compatibility_env=None, environ=None):
    try:
        return resolve_device(component, backend, component_env=component_env,
            compatibility_env=compatibility_env, capabilities=capabilities, environ=environ)
    except RuntimeError as exc:
        requested, source = device_request(None, component_env, compatibility_env, environ)
        return DeviceSelection(component, backend, requested, "unavailable", source,
            fallback=False, reason=str(exc))


def runtime_summary(environ=None):
    torch = probe_torch()
    ct2 = probe_ctranslate2()
    paddle = probe_paddle()
    visual = _diagnostic_selection("visual", "torch", torch, "VISUAL_DEVICE",
        "SEARCH_MODEL_DEVICE", environ)
    asr = _diagnostic_selection("asr", "ctranslate2", ct2, "ASR_DEVICE", environ=environ)
    ocr_backend = (os.environ if environ is None else environ).get("OCR_BACKEND", "auto").lower()
    if ocr_backend == "tesseract":
        ocr_capabilities = RuntimeCapabilities("cpu", True)
        ocr_backend_type = "cpu"
    elif ocr_backend == "paddleocr" or ocr_backend == "auto":
        ocr_capabilities = paddle
        ocr_backend_type = "paddle"
    else:
        ocr_capabilities = torch
        ocr_backend_type = "torch"
    ocr = _diagnostic_selection("ocr", ocr_backend_type, ocr_capabilities, "OCR_DEVICE", environ=environ)
    devices = {selection.device for selection in (visual, asr, ocr) if selection.device != "unavailable"}
    return {
        "mode": "mixed" if len(devices) > 1 else ("gpu" if (devices and next(iter(devices)).startswith("cuda")) else "cpu"),
        "capabilities": {
            "cpu_available": True,
            "torch": asdict(torch),
            "ctranslate2": asdict(ct2),
            "paddle": asdict(paddle),
            "ffmpeg_available": shutil.which("ffmpeg") is not None,
            "tesseract_available": shutil.which("tesseract") is not None,
            "paddleocr_available": importlib.util.find_spec("paddleocr") is not None,
            "easyocr_available": importlib.util.find_spec("easyocr") is not None,
        },
        "components": {
            "visual": asdict(visual),
            "asr": asdict(asr),
            "ocr": {**asdict(ocr), "ocr_backend": ocr_backend},
            "faiss": asdict(DeviceSelection("faiss", "cpu", "cpu", "cpu", "fixed")),
        },
    }
