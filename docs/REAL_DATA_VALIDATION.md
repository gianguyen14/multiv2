# Real Data Validation Report (Release Candidate 1.1.0-rc1)

## Executive Summary
This document summarizes the end-to-end dataset validation, artifact integrity verification, multi-modal search smoke tests, strict offline execution, and performance benchmarks conducted on the representative real-video dataset for HCM City AI Challenge 2026.

## 1. Environment & Hardware Baseline
- **Operating System**: Linux Fedora (Kernel 7.1.8-200.fc44)
- **CPU**: 11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz (8 logical cores)
- **Memory**: 7.5 GiB Physical RAM
- **GPU / CUDA**: CPU-only execution (CUDA hardware unavailable on host)
- **Python**: 3.12.13 in isolated `.venv`
- **Docker Engine**: 29.7.2
- **Docker Compose**: v5.4.0

## 2. Authoritative Validation Subset Specification

The validation dataset consists of 12 real MP4 video files in `data/test-videos/` (all 1280x720, H.264 / AVC1).
All numbers in this table are authoritatively verified via sequential PyAV decoding and index payload audit:

| Video ID | Resolution | FPS | Codec | Duration (s) | Decoded Frames | Indexed Frames | Size (Bytes) | SHA-256 Digest |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `L22_V001` | 1280x720 | 30.00 | H.264 | 1,163.37 | 34,901 | 1,164 | 122,770,058 | `f6e675751bfba2b4...` |
| `L22_V002` | 1280x720 | 25.00 | H.264 | 1,187.24 | 29,681 | 1,188 | 262,066,874 | `1472857617e3d805...` |
| `L22_V003` | 1280x720 | 25.00 | H.264 | 1,225.04 | 30,626 | 1,226 | 146,580,160 | `a200a5b965af7578...` |
| `L22_V004` | 1280x720 | 30.00 | H.264 | 1,304.27 | 39,128 | 1,305 | 154,212,776 | `9ca8fc0afd05afe3...` |
| `L22_V005` | 1280x720 | 30.00 | H.264 | 1,040.60 | 31,218 | 1,041 | 95,178,585 | `7107556e4a50b527...` |
| `L22_V006` | 1280x720 | 30.00 | H.264 | 1,286.87 | 38,606 | 1,287 | 141,458,543 | `5cc16bd00f6789b1...` |
| `L22_V007` | 1280x720 | 25.00 | H.264 | 1,201.28 | 30,032 | 1,202 | 139,761,248 | `40ea1e0a3e84e2de...` |
| `L22_V008` | 1280x720 | 25.00 | H.264 | 1,095.72 | 27,393 | 1,096 | 126,829,790 | `9723d266965f5a44...` |
| `L22_V009` | 1280x720 | 25.00 | H.264 | 1,077.08 | 26,927 | 1,078 | 100,145,844 | `45de796cf02019ac...` |
| `L22_V010` | 1280x720 | 25.00 | H.264 | 1,091.04 | 27,276 | 1,092 | 135,184,670 | `501109e86ac6ea8a...` |
| `L22_V011` | 1280x720 | 30.00 | H.264 | 1,096.03 | 32,881 | 1,097 | 119,638,570 | `dd473d70a938ab7a...` |
| `L22_V012` | 1280x720 | 30.00 | H.264 | 1,204.77 | 36,143 | 1,205 | 130,263,202 | `58dedb9a6848f1a8...` |
| **Totals** | - | - | - | **13,973.30s** | **384,812** | **13,981** | **1,674,090,320** | - |

- **Subtotal V001–V005**: 165,554 decoded frames, 5,924 indexed frames.
- **Manifest Location**: `validation/subset_manifest.json`

## 3. PyAV Frame ID Integrity & Audit Methodology
- **Audited Videos**: `L22_V001`, `L22_V002`, `L22_V003`, `L22_V004`, `L22_V005`
- **Decoded Frames Traversed**: 165,554 total sequential frames
- **Mapping Check Methodology**: 75 sampled frame positions across start, middle, and end intervals were explicitly checked against sequential decode display ordinals.
- **Observed Mismatches**: Exactly **0 / 75** mismatches (0.0%).
- **Full Corpus Mapping Integrity**: All 13,981 FAISS index mappings in `gen-af90ddd46e7649acae805307a71683c6` were verified against resolver payloads with **0 mismatches**.

## 4. Artifact & Publication Verification
- **Generation ID**: `gen-af90ddd46e7649acae805307a71683c6`
- **Vector Count**: 13,981 (all finite, L2 unit normalized, dimension 768)
- **Mapping Count**: 13,981
- **Payload Count**: 13,981
- **Integrity**: `vector_count == mapping_count == payload_count == 13,981` -> `True`
- **Staging Cleanup**: `.staging` directory clean and absent.

## 5. Query Validation Results (20-Query Batch)
- **Validation Suite**: 10 KIS, 5 Q&A, 3 TRAKE, 2 Image Search
- **Latency Summary**:
  - **KIS**: Min 223.99 ms, p50 315.59 ms, p90 1,189.81 ms, p95 1,654.60 ms, Max 2,119.38 ms
  - **Q&A**: Min 415.99 ms, p50 423.04 ms, p90 474.51 ms, p95 485.00 ms, Max 495.49 ms
  - **TRAKE**: Min 568.88 ms, p50 788.05 ms, p90 972.97 ms, p95 996.08 ms, Max 1,019.20 ms
  - **Image Search**: Min 166.74 ms, p50 210.51 ms, p90 245.52 ms, Max 254.27 ms
- **Exact Self-Match Test**: Frame `000000000.jpg` of `L22_V001` retrieved rank 1 with cosine similarity `1.000000`.

## 6. Strict Offline Execution & Model Performance
- Executed under `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
- `Qwen/Qwen2.5-1.5B-Instruct` loaded locally from cache with zero network calls.
- **CPU Inference Latency**: Local LLM query refinement and text generation on 8-core CPU takes `~10–49s` per query on cold start / CPU inference.
- **GPU Inference**: `NOT MEASURED ON THIS HOST` (Host is CPU-only; no speculative sub-150ms claims).
- Deterministic fallback parser operates at `~1.56 ms` on CPU.
