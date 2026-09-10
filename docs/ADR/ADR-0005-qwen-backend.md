# ADR-0005 — Qwen3-VL-Embedding-2B as Production Search Backend

**Status:** Accepted  
**Date:** 2026-09-10  
**Authors:** Engineering team (via agent audit)

## Context

The initial retrieval backend used SigLIP2 (`google/siglip2-base-patch16-224`, 768-d). As the AIC corpus grew to 873 videos and competition requirements evolved, SigLIP2 was found to produce a lower-dimensional embedding space that limited retrieval quality on the dense 1-fps frame database.

## Decision

Replace SigLIP2 as the production default with **Qwen3-VL-Embedding-2B** (`Qwen/Qwen3-VL-Embedding-2B`, revision `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`), a multimodal encoder producing 1024-d L2-normalised vectors with MRL (Matryoshka Representation Learning).

The FAISS index uses `IndexFlatIP` (inner-product, equivalent to cosine on L2-normalised vectors).

DB v1: 47,430 frames × 1024-d, 10-second sampling interval, 873 videos.  
DB v2: 470,833 frames × 1024-d, 1-second sampling interval, 873 videos.

SigLIP2 remains accessible as a legacy mode (`SEARCH_BACKEND=siglip2`) but no populated SigLIP2 FAISS index is present on the production host.

## Consequences

- **SEARCH_BACKEND=qwen3_vl** is the default (falls back via `SEARCH_ENCODER` alias).
- SigLIP2 backend is `LEGACY`; no new indexing pipeline targets it.
- Image search via Qwen3-VL visual queries is `EXPERIMENTAL` pending GPU validation on the production host.
- OCR/ASR lexical fusion continues to use DB v1 spool (VIDEO_PROCESSED_ROOT).
- Dense visual retrieval uses DB v2 (VISION_PROCESSED_ROOT) served read-only.
