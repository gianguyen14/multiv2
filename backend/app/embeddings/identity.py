"""Semantic embedding identity helpers."""
from __future__ import annotations

import hashlib
import json
from typing import Any


SEMANTIC_IDENTITY_FIELDS = (
    "backend", "model_name", "revision", "model_fingerprint", "instruction",
    "instruction_sha256", "embedding_dim", "normalization", "output_dtype",
    "contract_version",
)


def semantic_encoder_identity(identity: dict[str, Any], backend_override: str | None = None) -> dict[str, Any]:
    """Return only fields that define embedding-space compatibility."""
    result = {key: identity[key] for key in SEMANTIC_IDENTITY_FIELDS if key in identity}
    inferred_backend = identity.get("backend")
    if inferred_backend is None:
        model_name = str(identity.get("model_name", "")).lower()
        provider = str(identity.get("provider", "")).lower()
        if "siglip" in model_name or "siglip" in provider:
            inferred_backend = "siglip2"
    if backend_override is not None:
        inferred_backend = backend_override
    if inferred_backend is not None:
        result["backend"] = inferred_backend
    if "backend" not in result or "embedding_dim" not in result:
        raise ValueError("encoder identity requires backend and embedding_dim")
    return result


def semantic_identity_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    try:
        return semantic_encoder_identity(left) == semantic_encoder_identity(right)
    except ValueError:
        return False


def instruction_sha256(instruction: str) -> str:
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()


def identity_fingerprint(identity: dict[str, Any]) -> str:
    semantic = semantic_encoder_identity(identity)
    return hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
