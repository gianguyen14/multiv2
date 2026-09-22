import numpy as np
import pytest

from backend.app.runtime.ingest_policy import GPUCapability, initial_batch_size, select_dtype
from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.embeddings.ingest_encoder import QwenImageIngestEncoder, _validate_vectors


def test_dtype_policy_distinguishes_v100_from_ampere():
    v100 = GPUCapability(True, compute_capability=(7, 0), bf16_supported=False)
    ampere = GPUCapability(True, compute_capability=(8, 6), bf16_supported=True)
    assert select_dtype("auto", v100) == "float16"
    assert select_dtype("auto", ampere) == "bfloat16"
    assert select_dtype("auto", None) == "float32"


def test_batch_policy_covers_low_mid_and_high_memory_profiles():
    profiles = {
        "3060_8gb": GPUCapability(True, total_vram_bytes=8 * 1024**3, free_vram_bytes=7 * 1024**3),
        "mid_16gb": GPUCapability(True, total_vram_bytes=16 * 1024**3, free_vram_bytes=14 * 1024**3),
        "a40_48gb": GPUCapability(True, total_vram_bytes=48 * 1024**3, free_vram_bytes=44 * 1024**3),
        "pro_96gb": GPUCapability(True, total_vram_bytes=96 * 1024**3, free_vram_bytes=90 * 1024**3),
    }
    assert initial_batch_size(profiles["3060_8gb"], 1, 128) == 1
    assert initial_batch_size(profiles["mid_16gb"], 1, 128) == 4
    assert initial_batch_size(profiles["a40_48gb"], 1, 128) == 16
    assert initial_batch_size(profiles["pro_96gb"], 1, 128) == 24


def test_qwen_config_requires_1024_dimensions():
    with pytest.raises(ValueError, match="1024"):
        VideoIngestConfig(ingest_backend="qwen3_vl", qwen_embedding_dim=768)


def test_vector_contract_rejects_wrong_shape_and_norm():
    with pytest.raises(RuntimeError, match="contract"):
        _validate_vectors(np.ones((2, 4), dtype=np.float32), 2, 1024)
    values = np.zeros((1, 1024), dtype=np.float32)
    with pytest.raises(RuntimeError, match="normalized"):
        _validate_vectors(values, 1, 1024)


def test_qwen_oom_retry_repeats_same_offset_without_skip_or_duplicate(monkeypatch):
    encoder = QwenImageIngestEncoder.__new__(QwenImageIngestEncoder)
    encoder.device = "cuda:0"
    encoder.batch_size = None
    encoder.dtype = "float16"
    encoder.capability = GPUCapability(True, index=0, total_vram_bytes=16 * 1024**3)
    encoder._embedder = type("Embedder", (), {})()
    encoder._embedder.instruction = "instruction"
    encoder._embedder._ensure_loaded = lambda: None
    calls = []

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, item):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def to(self, dtype):
            return self

        def numpy(self):
            value = np.zeros(1024, dtype=np.float32)
            value[self.value] = 1.0
            return value

    def process(payloads, normalize=False):
        ids = [int(item["image"].split("-")[1]) for item in payloads]
        calls.append(ids)
        if len(ids) == 4 and len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return [FakeTensor(index) for index in ids]

    encoder._embedder._embedder = type("Model", (), {"process": staticmethod(process)})()
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)

    result = encoder.encode_image([f"frame-{index}" for index in range(6)], batch_size=4)

    assert result.shape == (6, 1024)
    assert np.argmax(result, axis=1).tolist() == list(range(6))
    assert calls == [[0, 1, 2, 3], [0, 1], [2, 3], [4, 5]]
    assert encoder.effective_batch_size == 2
