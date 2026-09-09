#!/usr/bin/env python3
"""Minimal GPU validation for Qwen image search (run on an authorized GPU host).

Loads the existing Qwen3-VL-Embedding-2B model, encodes ONE known indexed
source frame from the raw video corpus, searches the packed Vision DB v2 index
(gen-001) read-only, reports the source-frame rank, timings, and then exercises
the real HTTP endpoint locally.

Does NOT ingest, re-encode, rewrite, or download anything. Requires:
  - GPU host with CUDA torch and the repo + model + DB v2 mounted
  - /video/video/L21_V001.mp4 present (or override VIDEO_PATH)
  - backend installed (PYTHONPATH) and dependencies (fastapi testclient)
"""

import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_PROCESSED = Path("/home/hermes/aic/data/aic-db-v1/runtime")
DEFAULT_VISION = Path("/home/hermes/aic/data/aic-db-v2/vision/runtime")
DEFAULT_MODEL = Path("/home/hermes/aic/models/Qwen3-VL-Embedding-2B")
DEFAULT_VIDEO = Path("/video/video/L21_V001.mp4")
INSTRUCTION = "Retrieve the video frame that best matches the described visual scene."
DIM = 1024
MIN_PIXELS = 4096
MAX_PIXELS = 1310720
MAX_LENGTH = 8192


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--vision", type=Path, default=DEFAULT_VISION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--frame-second", type=int, default=0)
    parser.add_argument("--http-top-k", type=int, default=3)
    parser.add_argument("--http", action="store_true", help="also exercise POST /api/search/image")
    return parser.parse_args(argv)


def extract_frame(path: Path, second: int):
    import av
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
    except IndexError:
        raise RuntimeError("no video stream")
    target = second
    prev = None
    frame = None
    import fractions
    for f in container.decode(stream):
        sec = float(f.pts) * float(stream.time_base) if f.pts is not None else None
        if sec is not None and sec >= target:
            frame = f
            break
        prev = f
    container.close()
    if frame is None:
        raise RuntimeError(f"no frame available near second={second}")
    pil = frame.to_image().convert("RGB")
    return pil


def main(argv=None):
    args = parse_args(argv)
    if not (args.video.is_file()):
        raise SystemExit(f"video not found: {args.video}")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this validation requires a GPU host")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    from backend.app.embeddings.qwen3_vl import Qwen3VlLocalEmbedder
    embedder = Qwen3VlLocalEmbedder(
        model_dir=args.model,
        max_length=MAX_LENGTH,
        instruction=INSTRUCTION,
        torch_dtype="float16",
        attn_implementation="eager",
        threads=2,
    )
    # Enforce the exact preprocessing contract used by the dense DB v2 ingestion.
    embedder.min_pixels = MIN_PIXELS
    embedder.max_pixels = MAX_PIXELS
    embedder.image_max_length = MAX_LENGTH

    # 1) Encode a known indexed source frame.
    from PIL import Image
    pil = extract_frame(args.video, args.frame_second)
    t0 = time.perf_counter()
    vector = embedder.encode_image(pil, DIM)
    t_encode = time.perf_counter() - t0
    pil.close()
    print(f"encoded vector: shape={vector.shape} dtype={vector.dtype} "
          f"finite={bool((vector == vector).all() and (vector != float('inf')).all() and (vector != float('-inf')).all())} "
          f"l2={float((vector**2).sum()) ** 0.5:.6f} encode_s={t_encode:.2f}", flush=True)
    if vector.shape != (DIM,) or not bool((vector ** 2).sum() > 0.99 and (vector ** 2).sum() < 1.01):
        raise SystemExit("vector contract mismatch (expected 1024-d L2-normalized)")

    # 2) Search the packed Vision DB v2 read-only.
    from backend.app.services.qwen_runtime_search import QwenRuntimeSearch
    provider = QwenRuntimeSearch(
        processed_root=args.processed,
        vision_processed_root=args.vision,
        model_dir=args.model,
    )
    t1 = time.perf_counter()
    rows = provider._search_visual_vector(vector, top_k=5, deduplicate=False)
    t_search = time.perf_counter() - t1
    print(f"search top-5 took {t_search:.2f}s:", flush=True)
    for row in rows:
        print("  ", row["frame_uid"], row["score"], flush=True)
    expected_uid = f"L21_V001:{str(args.frame_second * 30).zfill(9)}"
    rank = next((i for i, r in enumerate(rows) if r["frame_uid"] == expected_uid), None)
    print(f"source-frame rank of {expected_uid}: {'not-in-top-5' if rank is None else rank}", flush=True)
    if rank is None:
        raise SystemExit("EXPECTED SOURCE FRAME NOT IN TOP-5 — preprocessing identity mismatch")

    # 3) Real HTTP endpoint.
    if args.http:
        from fastapi.testclient import TestClient
        from backend.app.main import create_app
        os.environ["QWEN_IMAGE_INFERENCE"] = "ready"
        client = TestClient(create_app())
        import io
        buf = io.BytesIO()
        pil2 = extract_frame(args.video, args.frame_second)
        pil2.save(buf, "PNG")
        t2 = time.perf_counter()
        resp = client.post(
            f"/api/search/image?top_k={args.http_top_k}",
            files={"file": ("query.png", buf.getvalue(), "image/png")},
        )
        t_http = time.perf_counter() - t2
        print(f"HTTP status={resp.status_code} elapsed_s={t_http:.2f}", flush=True)
        print(resp.json() if resp.status_code != 200 else [r["frame_uid"] for r in resp.json()["results"]], flush=True)
        if resp.status_code != 200:
            raise SystemExit("HTTP image endpoint failed")
    print("GPU_VALIDATION_PASS", flush=True)


if __name__ == "__main__":
    main()