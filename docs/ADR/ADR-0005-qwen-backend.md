# ADR-0005 — Qwen3-VL-Embedding-2B as Production Search Backend

**Status:** Accepted
**Date:** 2026-09-10
**Authors:** Engineering team (via agent audit)

## Context

The initial retrieval backend defaulted to SigLIP2 (`google/siglip2-base-patch16-224`, 768-d). During AIC 2026 development the production search backend was switched to a Qwen3-VL-based runtime.

## Decision

The production default search backend is **Qwen3-VL-Embedding-2B** (`Qwen/Qwen3-VL-Embedding-2B`, revision `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`), a multimodal encoder producing 1024-d L2-normalised vectors with MRL (Matryoshka Representation Learning).

The FAISS index uses `IndexFlatIP` (inner-product, equivalent to cosine on L2-normalised vectors).

The current `main` Qwen runtime serves **DB v1**: 47,430 frames × 1024-d, 10-second sampling interval, 873 videos.

SigLIP2 remains accessible as a legacy mode (`SEARCH_BACKEND=siglip2`) but no populated SigLIP2 FAISS index is present on the production host.

## Consequences

- **SEARCH_BACKEND=qwen3_vl** is the default (falls back via `SEARCH_ENCODER` alias).
- SigLIP2 backend is `LEGACY`; no new indexing pipeline targets it on `main`.
- Image search is **NOT available on current `main`** and is `EXPERIMENTAL` pending GPU validation (feature branch only).
- OCR/ASR lexical fusion uses the DB v1 spool via `VIDEO_PROCESSED_ROOT`.
- A dense DB v2 artifact (470,833 frames × 1024-d, 1-second interval) exists on disk and is used by the experimental image-search path. It is **not** served by the current `main` runtime and `VISION_PROCESSED_ROOT` is **not** an environment variable of the current `main` code.