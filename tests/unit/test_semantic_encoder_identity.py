from backend.app.embeddings.identity import semantic_encoder_identity


def qwen(runtime=None, instruction="Retrieve the video frame that best matches the described visual scene.", revision="rev-1", dim=1024):
    return {
        "backend": "qwen3_vl", "model_name": "Qwen/Qwen3-VL-Embedding-2B",
        "revision": revision, "instruction": instruction,
        "instruction_sha256": "instruction-hash", "embedding_dim": dim,
        "normalization": "l2", "output_dtype": "float32",
        "contract_version": "qwen3-vl-image-ingest-v1",
        **(runtime or {}),
    }


def test_runtime_fields_do_not_change_semantic_identity():
    base = qwen({"model_dir": "/a", "dtype": "bfloat16", "device": "cuda:0", "selected_batch_size": 16})
    other = qwen({"model_dir": "/b", "dtype": "float16", "device": "cuda:0", "selected_batch_size": 4})
    assert semantic_encoder_identity(base) == semantic_encoder_identity(other)


def test_instruction_revision_backend_and_dimension_change_identity():
    base = semantic_encoder_identity(qwen())
    assert semantic_encoder_identity(qwen(instruction="different")) != base
    assert semantic_encoder_identity(qwen(revision="rev-2")) != base
    assert semantic_encoder_identity(qwen(dim=768)) != base
    assert semantic_encoder_identity({"provider": "siglip2", "model_name": "google/siglip2", "embedding_dim": 768, "normalization": "l2"}) != base
