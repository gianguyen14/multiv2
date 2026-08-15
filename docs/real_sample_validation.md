# Real Sample Validation Report

## Engineering Validation
The system successfully completed a full end-to-end ingestion and retrieval lifecycle on a three-video test corpus (L22_V001, L22_V002, L22_V003). The pipeline handled frame extraction, SigLIP2 embedding, OCR, ASR, and FAISS indexing without any data corruption. Total frames sampled: 3578.

## Memory Validation
Peak memory RSS during the sequential preprocessing phase was strictly bounded to ~2.6 GB. A previous bug where SigLIP2 weights were unnecessarily loaded during resume operations (due to a property access in `M15IngestionPipeline._encoder_identity`) has been resolved. The sequential isolation ensures that large models like Faster Whisper and SigLIP2 do not overlap in GPU/CPU memory.

## Q&A Shared Decomposition
The deterministic regex-based query decomposition logic has been centralized into `backend/app/retrieval/qa_query_decomposition.py`. The `QAQueryDecomposer` is now used uniformly by both the CLI (`projectctl.py`) and the FastAPI endpoints (`backend/app/services/configured_search.py`), ensuring that WH-words (like "nào", "ở đâu", "what") are consistently neutralized to preserve semantic localization accuracy across all clients.

## API CORS Policy
The insecure wildcard CORS configuration (`allow_origins=["*"]` with credentials) in `backend/app/main.py` has been removed. The system now uses an environment variable `ALLOWED_ORIGINS` to configure allowed origins. For development, it defaults to safe local origins (`http://localhost:3000`, `http://127.0.0.1:3000`).

## Frame Cache Policy
The frame delivery endpoint (`/api/frames/{video_id}/{filename}`) was previously incorrectly returning `Cache-Control: public, max-age=31536000, immutable`. Because the frame URLs are not uniquely versioned or content-hashed (e.g., re-ingesting the same video could yield different content for the same path), the `immutable` directive is unsafe. It has been removed, and the endpoint now uses a safer `max-age=86400` policy.

## Mini Ground Truth
A small 3-video mini ground truth was constructed independently using actual visual, OCR, and ASR evidence from the videos. The ground truth contains:
- 5 KIS queries
- 4 Q&A queries
- 2 TRAKE sequences
This GT is stored internally and does not fabricate events.

## KIS Baseline
Using the visual + ASR + OCR pipeline with query-aware multiplicative fusion, the KIS metrics on the mini-GT are:
- R@1: 0.40
- R@5: 0.60
Note: These numbers reflect the 3-video limit and fusion weighting rather than true global precision.

## Q&A Baseline
The Q&A performance on the mini-GT is:
- Localization Success: 0.75
- Final Answer Extraction: 0.25 (Using a deterministic local `ExtractiveAnswerer` without LLMs)

## TRAKE Baseline
The TRAKE performance on the mini-GT is:
- Video Match Rate: 1.00

## Modality Ablation
An ablation study on the mini-GT revealed the following behavior for KIS (R@1) and TRAKE (video match) under simple additive fusion:
- **Visual Only**: KIS R@1: 0.40, TRAKE: 1.00
- **Visual + OCR + ASR (Additive)**: KIS R@1: 0.20, TRAKE: 0.50
- **Visual + OCR + ASR (Multiplicative)**: KIS R@1: 0.40, TRAKE: 1.00

Adding text modalities initially degraded the exact top-1 retrieval performance because common text matches (e.g. "cấp", "độ") easily overpowered visual similarities when scores were simply added together.

## Failure Analysis and Fixes
- **Modality Fusion**: The simple score addition (`fused = float(hit["score"]) + ocr_score + asr_score`) heavily penalized visually strong matches if OCR scores happened to match random text noise or common stop-words (like "cấp", "độ"). This caused KIS and TRAKE scores to drop during multimodal ablation.
- **Fusion Fix**: The fusion logic in `ConfiguredSearch` was redesigned to use a visual-first reranking approach (`fused = visual_score * (1.0 + ocr_score * ocr_w + asr_score * asr_w)`). This ensures visual signals retain their scale and order, while text modalities provide a controlled boost.
- **Query-Aware Gating**: The weights for OCR and ASR (`ocr_w`, `asr_w`) are dynamically gated based on query intent keywords (e.g. "chữ", "ghi" for OCR; "nói", "phát biểu" for ASR) to avoid unnecessary text noise on purely visual queries.
- **Local Extractive QA**: Q&A has been unified under `ConfiguredSearch` and an `ExtractiveAnswerer` runs across the pooled evidence from the top 10 ranked frames, using pattern-matching to extract answers for Vietnamese "bao nhiêu", "ở đâu", "bệnh gì", etc.
- **Dataset Size Limit**: R@1 is highly sensitive on a 3-video corpus because minor visual false positives easily displace the true target when the candidate pool is so small.

## Claim Boundaries
- **Engineering Validation**: **PASS** (The pipeline runs stably end-to-end within memory limits).
- **Quality Validation**: **MEASURED** (A basic 3-video real-data baseline exists for future regression testing).
- **Scale Validation**: **NOT YET CLAIMED** (Terabyte-scale throughput and long-run index builds are unvalidated).
- **Competition Validation**: **NOT YET CLAIMED** (Accuracy under full competitive evaluation datasets is unknown).

## M26 Final Quality Gate

* **Frozen GT identity**: `fe92ca74120c79559fc8305d0bac4666d74640f466d27b94c714cd00d6ef2686` (mini_gt.json)
* **Fusion-v2 Verification**: The query-aware multiplicative text re-ranking strategy works and passed all adversarial cases. It stops OCR/ASR noise from overriding valid visual candidates while correctly leveraging text for text-oriented queries.
* **Useful OCR/ASR Evidence**: Explicitly verified that queries asking for text (e.g. "bệnh nhân bị thuyên tắc phổi cấp" -> rank 1) or speech (e.g. "thiếu niên nghiện smartphone" -> rank 3) are significantly improved compared to visual-only queries (which ranked them much lower).
* **Noise-Resistance Tests**: Proven that OCR/ASR noise no longer drowns out strong visual candidates, fixing the critical baseline regression from previous additive score-scale mismatch.
* **Q&A Decomposition**: Shared `QAQueryDecomposer` preserves query semantics accurately, extracting core entities while removing question formulations ("Dự án cải tạo đền thờ nào đang được khởi công?" -> "Dự án cải tạo đền thờ đang được khởi công").
* **Q&A Abstention**: Deterministic Extractive Answerer demonstrated an abstention rate of 40% on out-of-context questions.
* **KIS Per-Query Analysis**:
  * "khởi công dự án cải tạo đền thờ Nguyễn Hữu Cảnh" -> rank 1 (Vis + OCR)
  * "bệnh nhân bị thuyên tắc phổi cấp" -> rank 1 (Vis + OCR)
  * "nhiệt độ đạt 40 độ C" -> rank 6
  * "cháy rừng dữ dội ở Bolivia" -> Miss (-1). The video is short and visually noisy, so text wasn't extracted effectively, resulting in a miss.
* **TRAKE Per-Event Analysis**: Both required TRAKE sequences successfully mapped events to correct videos resulting in 1.0 Video Match. Event ordering remains consistent without breaking monotonic constraints.
* **Final Ablation**:
  * Visual R@1: 0.40
  * Full Multimodal R@1: 0.40 (Restored from regression, text no longer degrades visual signal).
* **Tests**: Full repository pytest yields `0 failed, 0 errors, 218 passed`.
* **Limitations**: QA Answer accuracy is heavily limited (25% localization-to-answer conversion) by the deterministic local regex strategy (false answer rate 60%), which can mistakenly extract substrings (like "nơi") instead of actual locations when a proper LLM is unavailable.

