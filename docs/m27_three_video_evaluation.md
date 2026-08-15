# M27 — Three-Video Development Evaluation

## 1. Corpus and Scope
- **Scope**: Exactly 3 videos: `L22_V001.mp4`, `L22_V002.mp4`, `L22_V003.mp4`
- **Active Source Directory**: `data/validation/m27_three_video_sources/`
- **Active Processed Root**: `data/processed-validation/three-video-final/`
- **Dataset Verification**: Validated (`Video Count: 3, Valid: True`)

## 2. Frozen Ground Truth Manifest
- **Path**: `data/validation/m27_three_video_gt/v1/`
- **GT Hash**: `b9bced4990bb04dcca6acb8b101d3dd918e1e25951d7c49c3d8d2c3542e62a50`
- **Counts**:
  - KIS: 19 items (8 visual, 3 OCR, 6 ASR, 2 mixed)
  - QA Positive: 11 items (2 visual, 4 OCR, 5 ASR)
  - QA Negative: 10 items (Adversarial unsupported)
  - TRAKE: 6 sequences (1 visual, 1 ASR, 4 mixed)
- **Window Distribution**:
  - Min: 250 frames (~8.3s)
  - P25: 300 frames (~10.0s)
  - Median: 360 frames (~12.0s)
  - P75: 825 frames (~27.5s)
  - Max: 4020 frames (~134.0s)
  - <10s: 2, 10-30s: 13, 30-60s: 3, >60s: 1
- **Frame ID Methodology**: Exact PyAV sequential display-order zero-based ordinal emitted directly by PyAV decode via `frames.json`. Zero timestamp-based FPS multiplication.
- **Review Statistics**: 100% (46/46 items) independently reviewed against raw video and contact sheets.

## 3. Evaluator Deterministic Validation
- Unit test suite: `tests/unit/test_evaluator_semantics.py`
- Tests passed: 100% (KIS hit/miss/boundary/multiple intervals, QA localization/answer/alias/abstention/false answer, TRAKE hit/partial/video match/monotonic ordering).

## 4. Frozen Baseline Metrics (M26.1 System on Frozen 3-Video GT)

| Metric | Visual Only | Visual + OCR | Visual + ASR | All Modalities (Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **KIS R@1** | 0.00 | 0.00 | 0.00 | **0.00** |
| **KIS R@5** | 0.05 | 0.05 | 0.05 | **0.05** |
| **KIS R@20** | 0.16 | 0.16 | 0.16 | **0.16** |
| **KIS R@50** | 0.26 | 0.26 | 0.26 | **0.26** |
| **KIS R@100** | 0.42 | 0.42 | 0.42 | **0.42** |
| **KIS Correct Video Rate** | 1.00 | 1.00 | 1.00 | **1.00** |
| **QA Pos Localization Rate** | 0.09 | 0.09 | 0.09 | **0.09** |
| **QA Pos Answer Accuracy** | 0.09 | 0.09 | 0.09 | **0.09** |
| **QA Full Condition Score** | 0.00 | 0.00 | 0.00 | **0.00** |
| **QA Negative Abstention** | 0.90 | 0.90 | 0.90 | **0.90** |
| **QA Negative False Answer** | 0.10 | 0.10 | 0.10 | **0.10** |
| **TRAKE Video Match** | 0.50 | 0.50 | 0.50 | **0.50** |
| **TRAKE Event Hit Rate** | 0.00 | 0.00 | 0.00 | **0.00** |
| **TRAKE Ordering Validity** | 0.00 | 0.00 | 0.00 | **0.00** |

## 5. Measured Failure Classification & Priorities for M28
- **P1: Multimodal Fusion & Text Authority Gating** (ConfiguredSearch used brittle keyword matching causing OCR/ASR to receive negligible 0.01 authority).
- **P2: Visual Generic Anchor Domination** (Generic news anchor shots dominating top ranks over event-specific shots).
- **P3: QA Evidence Window & Temporal Pooling** (Top-10 retrieval pool misses evidence when localization rank is deep).
- **P4: TRAKE Candidate Coverage** (Event hits blocked by coarse candidate pool recall).
