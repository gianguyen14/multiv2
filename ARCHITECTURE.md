# System Architecture Overview

## High‑level components
```
frontend (React/Vite) ↔ FastAPI backend (2026‑Panel core)
    │
    ├─ loaders/                # ingest video, OCR, metadata, captions
    ├─ pipelines/              # orchestrate extraction, allow swapping implementations
    ├─ samplers/               # deterministic frame sampling strategies
    ├─ shot_detection/         # shot boundary detection
    ├─ embeddings/             # SigLIP2, DINOv3, multilingual‑E5 adapters
    ├─ indexes/                # FAISS sub‑indexes per modality, BM25 sparse index
    ├─ retrievers/             # hybrid RRF, TRAKE DP/beam
    ├─ strategies/             # QA answer extraction, KIS query expansion
    └─ core/                   # config, utilities, contracts
```

## Data contract
All modules exchange data through the `FrameData` schema (see `backend/app/schemas/frame.py`). This single source of truth guarantees consistent identifiers (`video_id`, `frame_id`, timestamps, shot IDs) across the pipeline.

## Sampling & Shot Detection (M2)
- **SamplingStrategy** interface defines deterministic frame extraction.
- **Fixed FPS strategies**: 0.25, 0.5, 1, 2 FPS.
- **ShotDetector** interface for shot boundary detection.
- **TransNetV2Adapter** implements ShotDetector.
- **Shot + 1 FPS** strategy: one frame per shot at 1 FPS.
- **Shot + adaptive dense** strategy:
  - scene < 4s → 2 FPS
  - 4-15s → 1 FPS
  - > 15s → 0.5 FPS
  - always include first/middle/last frame
- Every frame gets deterministic `frame_id`, exact `timestamp_ms`, and `shot_id` mapping.
- No duplicate frame IDs.

## Extensibility
New visual or text encoders can be added by implementing the corresponding **adapter interface** in `embeddings/` and registering it in the pipeline configuration.
New retrieval strategies are added as separate modules in `strategies/` and combined via RRF.

*Any architectural change must be recorded as an Architecture Decision Record (ADR) in `docs/ADR/` and approved before merging.*