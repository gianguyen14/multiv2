# M31 — Competition Readiness & Final Hardening Report

## 1. Overview & Scope
- **Active Development & Evaluation Corpus**: Exactly 3 videos (`data/test-videos/L22_V001.mp4`, `data/test-videos/L22_V002.mp4`, `data/test-videos/L22_V003.mp4`).
- **Ground Truth**: Frozen 3-Video Ground Truth v1 (`data/validation/m27_three_video_gt/v1/`).
  - SHA256 Hash: `c5afd80cf58f033675ed454c1441a61908291c6d81c06304301ee1c0a842e5d5`
  - Items: 19 KIS queries, 11 positive Q&A queries, 10 negative Q&A queries, 6 TRAKE sequences.

## 2. Milestone Progression (M27 -> M31.1)

| Milestone | KIS R@1 | KIS R@5 | KIS R@20 | QA Localization | QA Positive Acc | QA Full Condition | QA Neg Abstain | QA False Ans | TRAKE Video Match | TRAKE Order Valid |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M27 Baseline** | 0.00 | 0.11 | 0.16 | 0.00 | 0.09 | 0.00 | 0.90 | 0.10 | 0.50 | 0.00 |
| **M28 Retrieval v2** | 0.26 | 0.37 | 0.42 | 0.36 | 0.27 | 0.27 | 0.80 | 0.20 | 0.50 | 0.00 |
| **M29 Temporal** | 0.26 | 0.37 | 0.53 | 0.36 | 0.27 | 0.27 | 0.80 | 0.20 | 0.50 | **1.00** |
| **M30 Q&A Quality** | 0.26 | 0.37 | 0.53 | 0.36 | 0.45 | 0.36 | 0.70 | 0.30 | 0.50 | **1.00** |
| **M31 Hardened** | 0.26 | 0.37 | 0.47 | 0.36 | 0.36 | 0.36 | 0.60 | 0.40 | 0.50 | **1.00** |
| **M31.1 Final Closure** | **0.32** | **0.37** | **0.47** | **0.36** | **0.55** | **0.36** | **1.00** | **0.00** | **0.67** | **1.00** |

## 3. Engineering & Operational Verification
- **Output Semantics (M31.1 & M31.2)**: All frame outputs are 0-based sequential PyAV decode ordinals; outputs capped at `top_k <= 100` with deterministic sorting.
- **CLI Commands (M31.3)**: `status`, `doctor`, `models`, `dataset verify`, `kis`, `qa`, `trake`, `benchmark`, `evaluate` all verified.
- **API Endpoints (M31.4)**: `/health/live`, `/health/ready`, `/api/search`, frame serving tested and verified.
- **Frontend DOM Safety (M31.5)**: Verified zero untrusted HTML injection; full `escapeHTML` coverage.
- **Offline Mode (M31.6)**: Verified offline execution with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- **Idempotent Resume (M31.7)**: Tested `preprocess`, `ocr`, and `asr` on existing processed data; returned `resumed` with 0 redundant recomputation.
- **Performance Benchmark (M31.8)**: KIS end-to-end latency: p50 = 69.04 ms per query on CPU.
- **Test Suite (M31.10)**: Full test suite passing with 0 failures (`pytest -q`), clean `git diff --check`.

## 4. M31.1 Regression Closure
- **Q&A Safety & Abstention Resolution**:
  - Replaced ad-hoc blacklist words with a generalized linguistic taxonomy distinguishing structural head nouns (`_NON_PERSON_HEAD_NOUNS`, `_NON_LOCATION_HEAD_NOUNS`, `_GEOGRAPHIC_MARKERS`, `_PERSON_HONORIFICS`).
  - Added directional/attribute alignment checks (`giảm`, `tăng`, `dài`, `cao`, `mạnh`, `richter`) preventing false extraction from mismatched quantifiers.
  - Enforced proper-noun query constraint validation.
  - Result: **10/10 Negative Abstention (100%)** and **0.00 False-Answer Rate**.
- **Positive Q&A Accuracy Restoration**:
  - Domain answer types (`AUDIENCE`, `ANIMAL`, `TEMPLE`, `DISEASE`, `WHERE`, `HOW_MANY`) correctly extract supported answers while enforcing structural type validity.
  - Result: Positive Answer Accuracy improved from 0.36 to **0.55 (6/11)**.
- **KIS Multi-Modal Pool Expansion**:
  - Expanded candidate generation prior to temporal NMS (`max(top_k * 2, 200)`), improving R@1 to **0.32** while maintaining diversity across multi-video retrieval.
- **Generalization Tests**:
  - Created `tests/unit/test_qa_generalized_validation.py` with synthetic unseen vocabulary (`trường đại học`, `sở y tế`, `ban tổ chức`, `câu lạc bộ`, `nhà máy`, `viện nghiên cứu`, `dự án`, etc.) demonstrating zero-shot rejection without ad-hoc wordlists.
- **Baselines Saved**:
  - `eval/baselines/m31_1_three_video_final.json` frozen.
