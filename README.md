# Unified AIC Retrieval

Multimodal video retrieval system for **AIC 2026**, focused on fast and reliable retrieval from long-form video collections.

The project combines visual embeddings, speech, OCR, temporal reasoning, and evidence-aware ranking in a local/offline-friendly pipeline.

## What it supports

- **Textual KIS** — retrieve the most relevant video frames from a natural-language description.
- **Video Q&A** — retrieve supporting evidence and return the associated answer.
- **TRAKE** — retrieve an ordered sequence of semantic keyframes for multi-event temporal queries.
- **Image-to-frame search** — use an image as the query against indexed video frames.
- **OCR / ASR evidence** — combine text visible in frames with spoken content from video audio.

## Core stack

- **SigLIP2** — visual/text embeddings.
- **FAISS** — vector search over normalized frame embeddings.
- **Faster Whisper** — speech recognition.
- **Tesseract OCR** — OCR baseline for the current release path.
- **PyAV / FFmpeg** — authoritative video decoding and media processing.
- **FastAPI** — retrieval API.

## Retrieval pipeline

```text
Video
  │
  ├─ Decode / sample frames
  ├─ Visual embedding (SigLIP2)
  ├─ ASR (Faster Whisper)
  ├─ OCR (Tesseract)
  │
  ▼
FAISS + metadata index
  │
  ▼
Query intelligence
  │
  ├─ visual / semantic path
  ├─ OCR lexical path
  ├─ ASR lexical path
  └─ Vietnamese / English query variants
  │
  ▼
Fusion + evidence-aware reranking
  │
  ├─ KIS
  ├─ Q&A
  └─ TRAKE + temporal refinement
```

## Important frame-ID invariant

Authoritative frame IDs are **zero-based frame ordinals produced by sequential PyAV decoding in display order**.

They are never reconstructed from `timestamp × FPS`. This keeps indexed frames, evaluation output, and temporal refinement aligned with the source video.

## TRAKE coarse-to-fine retrieval

TRAKE first retrieves sparse global candidates, then performs dense temporal refinement only around promising regions. Ordered events are aligned monotonically inside the same video.

This keeps the permanent index compact while allowing higher temporal resolution where it matters.

## Current status

**1.1.0-rc2 prevalidation source**

The current release candidate is being validated on the target NVIDIA GPU environment before promotion. PaddleOCR GPU is not part of the accepted RC2 runtime path; the current OCR baseline is Tesseract.

## Development

Install the project in editable mode:

```bash
python -m pip install -e .
```

Run the test suite:

```bash
pytest
```

Verify imports:

```bash
python -c "import backend; import backend.app; print('imports: OK')"
```

## Project control CLI

The repository includes `projectctl.py` as the operator entry point for local project workflows.

```bash
python projectctl.py --help
```

Use the command help from the checked-out revision as the authoritative list of available operations.

## Repository layout

```text
backend/       application and retrieval pipeline
eval/          evaluation and benchmark utilities
scripts/       diagnostics, validation, dataset, and experiment helpers
tests/         unit and integration tests
docs/          additional project documentation
projectctl.py  operator CLI
```

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Engineering / agent rules](AGENTS.md)

## Design goals

The project prioritizes:

- deterministic frame identity;
- reproducible offline execution;
- sparse global indexing with bounded dense refinement;
- evidence-aware ranking instead of visual similarity alone;
- graceful fallback when optional components are unavailable;
- competition-oriented ranking quality and temporal diversity.

---

**Unified AIC Retrieval** is under active development for AIC 2026. Performance claims should be based on representative competition data and the corresponding validation results.
