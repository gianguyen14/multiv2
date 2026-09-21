import numpy as np
import pytest

from backend.app.runtime.ingest_policy import GPUCapability, initial_batch_size, select_dtype
from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.embeddings.ingest_encoder import _validate_vectors


def test_dtype_policy_distinguishes_v100_from_ampere():
    v100 = GPUCapability(True, compute_capability=(7, 0), bf16_supported=False)
    ampere = GPUCapability(True, compute_capability=(8, 6), bf16_supported=True)
    assert select_dtype("auto", v100) == "float16"
    assert select_dtype("auto", ampere) == "bfloat16"
    assert select_dtype("auto", None) == "float32"


def test_batch_policy_is_bounded_by_visible_free_memory():
    small = GPUCapability(True, total_vram_bytes=8 * 1024**3, free_vram_bytes=7 * 1024**3)
    large = GPUCapability(True, total_vram_bytes=96 * 1024**3, free_vram_bytes=90 * 1024**3)
    assert initial_batch_size(small, 1, 32) == 1
    assert initial_batch_size(large, 1, 32) == 24


def test_qwen_config_requires_1024_dimensions():
    with pytest.raises(ValueError, match="1024"):
        VideoIngestConfig(ingest_backend="qwen3_vl", qwen_embedding_dim=768)


def test_vector_contract_rejects_wrong_shape_and_norm():
    with pytest.raises(RuntimeError, match="contract"):
        _validate_vectors(np.ones((2, 4), dtype=np.float32), 2, 1024)
    values = np.zeros((1, 1024), dtype=np.float32)
    with pytest.raises(RuntimeError, match="normalized"):
        _validate_vectors(values, 1, 1024)
