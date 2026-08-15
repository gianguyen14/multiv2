import pytest

from backend.app.runtime.device_policy import (DeviceSelection, RuntimeCapabilities,
    device_request, normalize_device, resolve_device)


def capability(backend="torch", cuda=False, count=0, installed=True):
    return RuntimeCapabilities(backend, installed, cuda_available=cuda, cuda_device_count=count)


def test_device_syntax():
    assert normalize_device(None) == "auto"
    assert normalize_device(" CUDA:1 ") == "cuda:1"
    for value in ("gpu", "cuda:-1", "cuda:x", ""):
        with pytest.raises(ValueError):
            normalize_device(value)


def test_precedence_keeps_explicit_auto():
    env = {"SEARCH_MODEL_DEVICE": "cuda", "VISUAL_DEVICE": "cuda:1", "COMPUTE_DEVICE": "cpu"}
    assert device_request("auto", "VISUAL_DEVICE", "SEARCH_MODEL_DEVICE", env) == ("auto", "argument")
    assert device_request(None, "VISUAL_DEVICE", "SEARCH_MODEL_DEVICE", env) == ("cuda", "SEARCH_MODEL_DEVICE")
    assert device_request(None, "VISUAL_DEVICE", None, env) == ("cuda:1", "VISUAL_DEVICE")


def test_auto_cpu_and_cuda_are_backend_specific():
    cpu = resolve_device("visual", "torch", capabilities=capability(), environ={})
    gpu = resolve_device("visual", "torch", capabilities=capability(cuda=True, count=2), environ={})
    assert cpu.device == "cpu" and cpu.fallback
    assert gpu.device == "cuda:0"
    ct2 = resolve_device("asr", "ctranslate2", capabilities=capability("ctranslate2"), environ={})
    assert ct2.device == "cpu"


def test_explicit_cuda_errors_and_index_selection():
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_device("visual", "torch", "cuda", capabilities=capability())
    selected = resolve_device("visual", "torch", "cuda:1", capabilities=capability(cuda=True, count=2))
    assert selected == DeviceSelection("visual", "torch", "cuda:1", "cuda:1", "argument", 1)
    with pytest.raises(RuntimeError, match="only 1"):
        resolve_device("visual", "torch", "cuda:1", capabilities=capability(cuda=True, count=1))


def test_cpu_only_components_reject_cuda():
    assert resolve_device("faiss", "cpu", "auto").device == "cpu"
    with pytest.raises(RuntimeError, match="CPU-only"):
        resolve_device("tesseract", "cpu", "cuda")
