# Data Contract: Ingestion Schemas, Database Artifacts & Integrity

This document establishes the authoritative data contracts for the AIC 2026 Multimodal Video Retrieval System. It formalizes directory structures, JSON schemas for offline OCR/ASR spools, the FAISS generation publication protocol, frame identification invariants, timestamp derivation formulas, and read-only operational guarantees.

---

## 1. Directory Structure & Environment Roots

The system decouples data storage into two primary directories governed by environment variables:

```
+----------------------------------------------------------------------------------------------------+
|                         ROOT DIRECTORY STRUCTURE & NAMESPACE TOPOLOGY                              |
|                                                                                                    |
|   VIDEO_PROCESSED_ROOT (Production DB v1: Coarse 10s Sampling)                                     |
|   ├── index/                                                                                       |
|   │   ├── CURRENT                          <-- JSON pointer to active generation                   |
|   │   ├── generations/                                                                             |
|   │   │   └── gen-A-b531063ff8ce48e1b0d62739fb290c23/                                             |
|   │   │       ├── frames.faiss             <-- 47,430 x 1024-d FAISS IndexFlatIP                   |
|   │   │       ├── mapping.json             <-- Vector index -> Frame UID mapping                   |
|   │   │       ├── payloads.json            <-- Frame metadata payloads                             |
|   │   │       └── generation.json          <-- Manifest & cryptographic digests                    |
|   │   └── .staging/                        <-- Temporary build directory (cleaned on boot)         |
|   ├── ocr/                                 <-- PaddleOCR extracted JSON spools per video           |
|   │   ├── L21_V001.json                                                                            |
|   │   └── ...                                                                                      |
|   └── asr/                                 <-- Faster-Whisper audio transcript JSON spools         |
|       ├── L21_V001.json                                                                            |
|       └── ...                                                                                      |
|                                                                                                    |
|   VISION_PROCESSED_ROOT (Experimental DB v2: Dense 1 FPS Sampling)                                 |
|   └── index/                                                                                       |
|       ├── CURRENT                          <-- JSON pointer to active generation (gen-001)         |
|       └── generations/                                                                             |
|           └── gen-001/                                                                             |
|               ├── frames.faiss             <-- 470,833 x 1024-d FAISS IndexFlatIP (~1.93 GB)       |
|               ├── mapping.json             <-- Vector index -> Frame UID mapping (14.9 MB)         |
|               ├── payloads.json            <-- Frame metadata payloads (156.4 MB)                  |
|               ├── generation.json          <-- Manifest metadata (33.2 KB)                         |
|               └── artifact_sha256.json     <-- Cryptographic SHA-256 sidecar                       |
+----------------------------------------------------------------------------------------------------+
```

### 1.1 Read-Only Runtime Guarantee

The production retrieval runtime operates under an absolute **read-only guarantee**:
- The application server mounts `$VIDEO_PROCESSED_ROOT` and `$VISION_PROCESSED_ROOT` with read permissions only.
- Ingestion pipelines (`m15_ingestion_pipeline.py`, `m16_text_pipeline.py`) are strictly offline batch utilities; they are never executed inside the serving path.
- The runtime never writes, mutates, moves, copies, re-encodes, or re-indexes database files.
- Stale staging directories (`.staging/`) are cleaned only when explicitly building new index generations via CLI utilities.

---

## 2. DB v1 Evidence Spool Schemas

The search runtime does not execute live OCR or ASR inference. Instead, it mounts pre-extracted JSON spools located in `$VIDEO_PROCESSED_ROOT/ocr/*.json` and `$VIDEO_PROCESSED_ROOT/asr/*.json`.

### 2.1 Optical Character Recognition (OCR) Spool Schema

Each video has a corresponding `$VIDEO_PROCESSED_ROOT/ocr/{video_id}.json` containing an array of OCR observation objects.

```json
[
  {
    "video_id": "L21_V001",
    "timestamp_seconds": 0.0,
    "source_frame_index_zero_based": 0,
    "frame_uid": "L21_V001:0",
    "raw_text": "HTVO HD 06:30:11 giay",
    "normalized_text": "htvo hd 06:30:11 giay",
    "confidence": 0.863752618432045,
    "backend_identity": "paddleocr:vi:cuda:3"
  }
]
```

#### Field Specifications:
- `video_id` (string, required): Stem of the video container (e.g. `"L21_V001"`). Must match regex `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`.
- `timestamp_seconds` (float, required): Container presentation timestamp in seconds where text was detected.
- `source_frame_index_zero_based` (integer, required): Zero-based sequential display frame index decoded from the video.
- `frame_uid` (string, required): Identifier linking the detection to a video frame.
- `raw_text` (string, required): Exact UTF-8 text recognized by the OCR engine.
- `normalized_text` (string, required): Lowercased, Unicode NFC normalized text without extraneous whitespace.
- `confidence` (float, required): Recognition confidence score $[0.0, 1.0]$. Filtered offline by `OCR_PADDLE_MIN_CONFIDENCE=0.50`.
- `backend_identity` (string, required): Provenance string declaring model and device (e.g. `"paddleocr:vi:cuda:3"` or `"tesseract:vie+eng"`).

### 2.2 Automatic Speech Recognition (ASR) Spool Schema

Audio speech transcripts are serialized in `$VIDEO_PROCESSED_ROOT/asr/{video_id}.json` as an array of speech segment objects.

```json
[
  {
    "video_id": "L21_V001",
    "start_seconds": 4.43,
    "end_seconds": 8.11,
    "raw_transcript": "Chào mừng quý vị đến với chương trình sống vây dây của Đại truyền hình thành phố Chị Mình",
    "normalized_transcript": "chào mừng quý vị đến với chương trình sống vây dây của đại truyền hình thành phố chị mình",
    "confidence": -0.3343446675724196,
    "backend_identity": "faster-whisper:small:float16:cuda:5"
  }
]
```

#### Field Specifications:
- `video_id` (string, required): Stem of the video container (e.g. `"L21_V001"`).
- `start_seconds` (float, required): Start boundary of the speech segment in seconds.
- `end_seconds` (float, required): End boundary of the speech segment in seconds.
- `raw_transcript` (string, required): Raw transcribed text segment.
- `normalized_transcript` (string, required): Lowercased, Unicode NFC normalized transcript text.
- `confidence` (float, required): Whisper log-probability or segment confidence.
- `backend_identity` (string, required): Provenance string declaring model and device (e.g. `"faster-whisper:small:float16:cuda:5"`).

### 2.3 Runtime In-Memory Ingestion & Temporal Mapping

When `QwenRuntimeSearch` initializes:
1. All JSON files in `ocr/` and `asr/` are parsed into flat memory structures.
2. For ASR segments spanning `[start_seconds, end_seconds]`, the representative timestamp is computed as the midpoint:
   $$	ext{timestamp\_seconds} = rac{	ext{start\_seconds} + 	ext{end\_seconds}}{2.0}$$
3. Stopwords are filtered using pre-compiled set `_STOPWORDS` (Vietnamese and English function words).
4. `nearest_uid(timeline, video_id, timestamp)` uses binary search (`bisect_left`) over sorted frame timestamps to map text hits to the closest indexed visual keyframe in that video.

---

## 3. Database Generation Schema & Artifacts

A published FAISS generation resides in `generations/{generation_id}/` and consists of five core artifacts.

### 3.1 `CURRENT` Active Generation Pointer
Located at `$VIDEO_PROCESSED_ROOT/index/CURRENT` or `$VISION_PROCESSED_ROOT/index/CURRENT`.
```json
{
  "schema_version": 1,
  "generation_id": "gen-A-b531063ff8ce48e1b0d62739fb290c23"
}
```
Validation rules:
- `schema_version` must equal `1`.
- `generation_id` must start with `"gen-"` and contain no path traversal characters (`/`, `\`, `..`).

### 3.2 `frames.faiss` (Vector Index)
- Binary serialized FAISS index.
- Index type: `IndexFlatIP` (inner product cosine similarity).
- Embedding vectors: Float32, L2 unit-normalized.
- Dimension: **1024-D** for Qwen3-VL (or 768-D for legacy SigLIP2).
- Count ($N_{	ext{total}}$): 47,430 vectors in DB v1; 470,833 vectors in DB v2.

### 3.3 `mapping.json` (Index-to-UID Mapping)
Maps sequential 0-based vector positions ($0 \dots N-1$) in `frames.faiss` to canonical `frame_uid` strings.
```json
{
  "frame_id_mapping": {
    "0": "L21_V001:000000000",
    "1": "L21_V001:000000300",
    "2": "L21_V001:000000600",
    "47429": "L30_V096:000018600"
  }
}
```
Validation invariants:
- Keys must form a contiguous sequence $0, 1, \dots, N-1$.
- Values must be non-empty and globally unique.
- Count must exactly match `index.ntotal`.

### 3.4 `payloads.json` (Frame Metadata Payloads)
Stores metadata for candidate resolution and competition submission generation.
```json
{
  "schema_version": 1,
  "payloads": {
    "L21_V001:000000000": {
      "candidate_id": "L21_V001:000000000",
      "video_id": "L21_V001",
      "source_frame_index_zero_based": 0,
      "timestamp_seconds": 0.0,
      "pts": 0,
      "frame_uid": "L21_V001:000000000",
      "source_path": "/aic/corpus/video/L21_V001.mp4",
      "sample_second": 0,
      "sampling_reason": "dense_1fps"
    }
  }
}
```
Validation invariants:
- `candidate_id` and `frame_uid` must match the dictionary key.
- `video_id` must match `frame_uid.split(':')[0]`.
- `source_frame_index_zero_based` must match integer conversion of `frame_uid.split(':')[1]`.

### 3.5 `generation.json` (Manifest & Checksums)
Metadata declaring encoder provenance and cryptographic hashes of sibling files.
```json
{
  "schema_version": 1,
  "generation_id": "gen-A-b531063ff8ce48e1b0d62739fb290c23",
  "generation_label": "GENERATION_A",
  "created_at_epoch": 1787945806.2910697,
  "source_video_count": 873,
  "pass_count": 873,
  "interval_seconds": 10.0,
  "index_type": "flat",
  "embedding_dim": 1024,
  "vector_count": 47430,
  "encoder_identity": {
    "provider": "sentence-transformers",
    "backend": "qwen3_vl_embedding_2b",
    "model": "Qwen/Qwen3-VL-Embedding-2B",
    "revision": "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
    "embedding_dim": 1024,
    "normalization": "l2",
    "instruction": "Retrieve the video frame that best matches the described visual scene.",
    "contract_version": "visual-encoder-v2",
    "fingerprint": "5835d5fd0517b47c4ba72ebb7b2e8af8f0ab5102aada0ed34d497fef5684cd99"
  },
  "artifact_sha256": {
    "frames.faiss": "b61ed8652f1661b8fd881cbeaa9c23516521fbe28872281fd925c77760768abb",
    "mapping.json": "db3cdbf6a3c184683a27c2feaa960fab8f83a6c1a2b66a2a3e9c1c4493e17d0c",
    "payloads.json": "1c566e07bc73bd87f09069d010ba7ec579e179085aea78463799ed6430c9c8ef"
  }
}
```

### 3.6 `artifact_sha256.json` (Sidecar Checksum File)
Used in packed DB v2 generations for rapid integrity validation:
```json
{
  "frames.faiss": "e37efa3ff6cb99de038a7e4378bdd7b1cf547c1397f70b72ad041957672825b4",
  "generation.json": "6f54b4a60033ae3538cfdce0df3037c8cd954f9f85ffca527b596adb24e1bd4c",
  "mapping.json": "e1112053c4563164cd27be5e139dd795c4626b0d0021c70b07f7de273d3e1477",
  "payloads.json": "473ccaf1c33ee95b64119be75e5b7483a29ba75c500258ca460c436a55f24ed5"
}
```

---

## 4. Frame Identification & Naming Contract

The canonical identifier for any indexed video keyframe is `frame_uid`.

### 4.1 Canonical Format
```
{video_id}:{source_frame_index_zero_based:09d}
```
Example: `L21_V001:000000000`, `L21_V001:000000300`, `L26_V120:000015000`.

- The prefix is the unique video identifier stem (`video_id`).
- The delimiter is a single colon (`:`).
- The suffix is the zero-based sequential decoded frame index padded with leading zeros to **exactly 9 digits**.

### 4.2 Submission Frame ID Policy (`FrameIdPolicy`)
The AIC 2026 competition evaluation system enforces submission IDs:
- Mode: `zero_based` (default in production) or `one_based`.
- Conversion logic in `backend/app/video/frame_id_policy.py`:
  ```python
  submission_frame_id = source_frame_index_zero_based + (mode == "one_based")
  ```
- The production database payloads explicitly record both:
  ```json
  "source_frame_index_zero_based": 300,
  "submission_frame_id": 300
  ```

---

## 5. Timestamp Derivation & Temporal Consistency

Video frames exhibit varying container formats, timebases, and frame rates. The system guarantees mathematical precision when converting container frames into temporal timestamps.

### 5.1 PyAV Presentation Timestamp (PTS) Derivation

In `backend/app/video/video_decoder.py:iter_frames`:
```python
time_base = frame.time_base or stream.time_base
timestamp = None
if frame.pts is not None and time_base is not None:
    timestamp = float(Fraction(frame.pts) * Fraction(time_base))
```
- Timestamps are computed strictly from the container presentation timestamp (`frame.pts`) multiplied by the rational timebase (`frame.time_base` or `stream.time_base`) using Python's exact `fractions.Fraction`.
- This ensures microsecond accuracy even with variable frame rate (VFR) containers or fractional frame rates (e.g. 29.97 FPS, $30000/1001$).

### 5.2 Nearest-Timestamp Periodic Frame Sampling

In `backend/app/video/frame_sampler.py:iter_sample_frames`:
- Sampling target initialized at `target = 0.0`.
- Increments by `interval_seconds` (e.g. 10.0s for DB v1, 1.0s for DB v2).
- When container timestamps advance past `target`, candidate selection picks the frame with minimum absolute temporal error:
  ```python
  if abs(current.timestamp_seconds - target) < abs(previous.timestamp_seconds - target):
      candidate = current
  ```
- Guaranteed single-frame selection per target step with chronological monotonicity.
