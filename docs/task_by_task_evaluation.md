# Task-by-Task Retrieval Evaluation

## Evaluation philosophy

This document compares **implemented architecture and task coverage**, not empirical quality. A component only counts as active when it is wired into the audited runtime path. Any unmeasured gain remains `TO_BE_MEASURED`.

Official QA scoring semantics are `UNRESOLVED` from the currently audited training-video evidence. Any exact-match answer metric in the benchmark plan is an **internal proposed metric**, not an asserted official scoring rule.

## 1. Textual KIS

Goal: retrieve the target video/frame or a valid frame interval from a text description.

### Audited execution patterns

- **AIC2024:** WordNet/translation-era query handling -> OpenCLIP -> Milvus, with OCR score combination in relevant paths.
- **AIC2025:** direct text query -> SigLIP2-384 -> FAISS keyframes, with OCR/transcript tools largely exposed as separate flows.
- **Panel2026:** CLIP-B/32 + mock BM25 -> RRF prototype.
- **CurrentSystem:** QueryRefiner/QueryPlan -> bilingual visual queries + OCR + ASR -> RRF -> deterministic CandidateReranker -> 60-frame temporal NMS.

CurrentSystem is the **most complete audited active KIS architecture** in this set. Comparative Recall/MRR remains `TO_BE_MEASURED`.

### Dense KIS gap

`DenseTemporalRefiner` exists in `backend/app/retrieval/kis_pipeline.py` but is not active in `ConfiguredSearch`.

The existing refiner calls `decode_frame_indices()` per coarse candidate. Since `decode_frame_indices()` iterates from frame zero until the furthest requested ordinal, naive activation would repeat sequential decode work. Actual quality and latency impact: `TO_BE_MEASURED`.

## 2. OCR-heavy search

Target examples: license plates, signs, names, numeric codes, prices, addresses.

### Audited capabilities

- **AIC2024:** EasyOCR + ElasticSearch text path.
- **AIC2025:** Vintern JSON substring matching.
- **Panel2026:** no real OCR path in the audited prototype.
- **CurrentSystem:** OCR evidence + exact/raw/normalized/separator-aware matching + lexical matching in the multi-channel plan.

CurrentSystem includes explicit separator-normalization logic useful for strings such as `50H-052.03` vs `50H 052.03`. Whether this produces higher Precision/Recall than historical systems is `TO_BE_MEASURED`.

### Required benchmark failures

- OCR detection miss;
- separator/normalization miss;
- short-substring false positive;
- candidate fusion miss;
- correct OCR evidence but wrong frame rank.

## 3. ASR-heavy search

- **AIC2024:** no audited ASR retrieval path.
- **AIC2025:** Whisper transcripts + MiniLM semantic matching.
- **Panel2026:** no audited ASR path.
- **CurrentSystem:** Faster-Whisper timestamped evidence + lexical/exact matching integrated into RRF.

AIC2025's semantic transcript embedding and CurrentSystem's lexical/exact path are different design choices. Neither is declared superior without a speech-query benchmark.

## 4. Video QA

Evaluation must separate:

1. evidence-video retrieval;
2. evidence-frame localization;
3. evidence extraction;
4. answer generation/extraction;
5. abstention behavior.

### Audited status

| System | QA implementation |
|---|---|
| AIC2024 | manual/proxy workflow rather than an active evidence-first QA engine |
| AIC2025 | manual retrieval workflow; no complete automated QA engine identified |
| Panel2026 | mocked/prototype QA |
| CurrentSystem | active `QAQueryDecomposer` + search + evidence gathering + `ExtractiveAnswerer` + abstention |

CurrentSystem is the only audited system with this complete active evidence-first structure. Answer accuracy remains `TO_BE_MEASURED`.

Internal benchmark metrics may include evidence recall, evidence-grounded answer rate, abstention metrics, and an internal normalized exact-match metric. Official competition scoring semantics are not asserted here.

## 5. TRAKE

Goal: retrieve an ordered sequence of events from the same video.

### Audited approaches

- **AIC2024:** heuristic relation between a main query and follow-up queries; strict monotonic event alignment is not the central audited mechanism.
- **AIC2025:** partial-sequence expansion constrained by keyframe gaps. A safe structural bound is `S_i <= S_(i-1) * G_i`.
- **Panel2026:** prototype video-frequency selection followed by chronological sorting.
- **CurrentSystem:** per-stage retrieval -> `TemporalRefiner` -> `TRAKEAligner` monotonic DP -> coherence diagnostics.

Current active defaults found in the audit:

- `transition_penalty = 0.0`
- `max_gap = None`

Therefore current DP enforces monotonic ordering but should not be described as actively using a non-zero transition penalty or finite max-gap unless configured.

Comparative sequence accuracy: `TO_BE_MEASURED`.

## 6. Image semantic search

- AIC2024: OpenCLIP image retrieval through Milvus.
- AIC2025: SigLIP2 keyframe image search.
- CurrentSystem: multipart image -> `SigLIP2Encoder.encode_image()` -> FAISS -> temporal NMS.

This task should be benchmarked separately from instance retrieval.

## 7. Instance retrieval

AIC2025 contains a dedicated DINOv3 index. CurrentSystem does not have a dedicated instance sidecar and uses SigLIP2 for image search.

DINOv3 is a **candidate representation for an isolated instance-retrieval experiment**, not a proven improvement.

Required subset examples:

- same logo;
- same vehicle;
- same landmark/building;
- unique clothing/package;
- near duplicate;
- semantically similar but different instance.

Metrics: Precision@1/5 or Recall@K on instance identity. Comparative DINOv3 vs SigLIP2 result: `TO_BE_MEASURED`.

## 8. Long-video retrieval

CurrentSystem's indexed sampled frames can be searched without re-decoding the entire source video, but on-demand access to arbitrary non-indexed ordinals uses sequential PyAV via `decode_frame_indices()`.

Structural cost grows with the furthest requested ordinal. Wall-clock seconds are **not inferred** from video duration.

This task requires separate retrieval and sparse-access measurements.

## 9. Sparse-frame access

Candidate decoder strategies for benchmark:

- D1: current sequential PyAV reference;
- D2: grouped sequential decode;
- D3: cache/predecoded lookup;
- D4: GOP-aware bounded-seek prototype;
- D5: optional GPU-assisted decode.

No candidate is a winner until it passes exact authoritative-ordinal parity and workload-specific latency/resource tests.

## 10. Near-duplicate retrieval / temporal dedup

CurrentSystem uses a **60-frame per-video suppression window**. That is the invariant. It is only approximately two seconds for ~30-fps material.

AIC2025 scene-midpoint sampling naturally reduces adjacent indexed frames but is not equivalent to CurrentSystem's post-retrieval temporal NMS.

Required ablation: temporal NMS ON/OFF, with target suppression failures classified explicitly.

## 11. Human operator workflow

The training video shows interactive candidate inspection, temporal neighbor browsing, and video verification as prominent workflow elements (`VIDEO_EVIDENCE`). No numerical share of operator time is asserted.

### Audited interaction features

| Feature | AIC2024 | AIC2025 | Panel2026 | CurrentSystem |
|---|---|---|---|---|
| result browsing | active | active | active | active |
| `/` focus | not identified | not identified | active | active |
| arrow grid navigation | not identified | not identified | active | missing |
| neighbor/timeline view | present concept | partial/dialog | active | missing |
| video player | present | present | present | missing |
| result-card “more like this” workflow | present concept | present/recommend flow | not central | missing in UI |
| copy/export interaction | submission/export oriented | CSV oriented | dry-run oriented | clipboard copy |

The operator baseline must measure **time to relevant result**, **time to exact frame**, **query reformulations**, and **click/keystroke count** before deciding UI benefit.

## 12. Failure taxonomy

Every failed benchmark item should receive one or more labels:

- `QUERY_PARSE_MISS`
- `TRANSLATION_MISS`
- `VISUAL_MODEL_MISS`
- `SAMPLING_MISS`
- `OCR_MISS`
- `ASR_MISS`
- `EXACT_TEXT_MISS`
- `FUSION_MISS`
- `RERANK_MISS`
- `DEDUP_MISS`
- `TEMPORAL_REFINEMENT_MISS`
- `TRAKE_ALIGNMENT_MISS`
- `FRAME_MAPPING_MISS`
- `RESULT_SERIALIZATION_MISS`
- `OPERATOR_UI_MISS`
- `DATASET_ANNOTATION_ISSUE`
- `AMBIGUOUS_QUERY`

The objective is not only “how often did retrieval fail?” but “which stage caused the failure?”.

## 13. Capability/status matrix

Status describes implementation, not measured quality.

| Task | AIC2024 | AIC2025 | Panel2026 | CurrentSystem | Next evaluation |
|---|---|---|---|---|---|
| Textual KIS | `IMPLEMENTED_PARTIAL` | `IMPLEMENTED_PARTIAL` | `MOCKED/PARTIAL` | `IMPLEMENTED_ACTIVE` | frozen baseline Recall/MRR |
| Exact-frame dense KIS | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `UNWIRED` | dense A/B after sparse-access baseline |
| OCR-heavy | `IMPLEMENTED_PARTIAL` | `IMPLEMENTED_PARTIAL` | `NOT_IMPLEMENTED` | `IMPLEMENTED_ACTIVE` | OCR exact/normalized benchmark |
| ASR-heavy | `NOT_IMPLEMENTED` | `IMPLEMENTED_ACTIVE` | `NOT_IMPLEMENTED` | `IMPLEMENTED_ACTIVE` | speech-query benchmark |
| QA | `MANUAL/PROXY` | `MANUAL` | `MOCKED` | `IMPLEMENTED_ACTIVE` | evidence + answer benchmark |
| TRAKE | `IMPLEMENTED_PARTIAL` | `IMPLEMENTED_PARTIAL` | `MOCKED/PARTIAL` | `IMPLEMENTED_ACTIVE` | CSA/event-hit baseline |
| Image semantic | `IMPLEMENTED_ACTIVE` | `IMPLEMENTED_ACTIVE` | `NOT_IDENTIFIED` | `IMPLEMENTED_ACTIVE` | semantic image baseline |
| Dedicated instance | `NOT_IDENTIFIED` | `IMPLEMENTED_ACTIVE` | `NOT_IMPLEMENTED` | `NOT_IMPLEMENTED` | DINO sidecar experiment |
| Sparse frame access | `IMPLEMENTED_PARTIAL` | `IMPLEMENTED_PARTIAL` | `PROTOTYPE` | `IMPLEMENTED_SEQUENTIAL` | D1-D5 decode benchmark |
| Timeline operator UI | `PRESENT` | `PARTIAL` | `IMPLEMENTED_ACTIVE` | `NOT_IMPLEMENTED` | P1-A operator study |

## 14. Ground-truth status

Ground-truth coverage is **PARTIAL / INSUFFICIENT FOR MASTER COMPARISON**. Existing validation material is not enough to support cross-system quality superlatives or architecture promotion decisions.

The next scientific step is the multi-task benchmark defined in `docs/benchmark_master_plan.md`.
