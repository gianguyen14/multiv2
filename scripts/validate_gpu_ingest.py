#!/usr/bin/env python3
"""Validate GPU policy and one Qwen image embedding without ingesting a corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.embeddings.ingest_encoder import QwenImageIngestEncoder
from backend.app.runtime.ingest_policy import kernel_smoke, probe_gpu, select_dtype, initial_batch_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    args = parser.parse_args()
    capability = probe_gpu(args.device)
    payload = {"gpu": capability.to_dict(), "selected_dtype": select_dtype(args.dtype, capability),
        "selected_batch": initial_batch_size(capability), "kernel_smoke": None, "embedding": None}
    if not capability.available:
        payload["status"] = "PENDING_HARDWARE_VALIDATION"
        print(json.dumps(payload, indent=2))
        return 2
    payload["kernel_smoke"] = kernel_smoke(f"cuda:{capability.index}")
    if payload["kernel_smoke"]["status"] != "PASS":
        payload["status"] = "FAIL"
        print(json.dumps(payload, indent=2))
        return 1
    encoder = QwenImageIngestEncoder(model_dir=args.model, device=f"cuda:{capability.index}", dtype=args.dtype, strict_gpu=True)
    from PIL import Image
    with Image.open(Path(args.image)) as image:
        vector = encoder.encode_image(image, batch_size=1)
    payload["embedding"] = {"shape": list(vector.shape), "dtype": str(vector.dtype), "finite": bool(__import__('numpy').isfinite(vector).all()), "norm": float(__import__('numpy').linalg.norm(vector[0]))}
    payload["status"] = "PASS" if payload["embedding"]["shape"] == [1, 1024] and abs(payload["embedding"]["norm"] - 1.0) <= 1e-5 else "FAIL"
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
