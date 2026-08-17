# Master Video Retrieval System Comparison

## Status

This document is an **evaluation baseline**, not a performance leaderboard. It compares four audited codebases and separates code/config facts from unmeasured quality claims.

Audited systems:

- **AIC2024** — audit artifact HEAD `33e49a0a3fb13a5bfd3c83f784d86ba4dd3e0e16`
- **AIC2025** — audit artifact HEAD `66c60ea8cc2bd0ec51a37ce7db60096578a23e6a`
- **Panel2026** — audit artifact HEAD `4db2f8c92b2f4ac3c025cf311ef5638dd3f908c6`
- **CurrentSystem / multiv2** — audited main `2e2a9ceecbaed197de4f63b1c22856cc219fb1ae`

Training source: `Tập huấn AIC 2026 - Buổi 4 [1080p].mp4` was used only for task/output/operator-workflow concepts. Video observations are not evidence that a repository implements a feature.

## Evidence labels

- `PROVEN_FROM_CODE`: directly supported by inspected source.
- `PROVEN_FROM_CONFIG`: directly supported by configuration/manifests.
- `PROVEN_FROM_ARTIFACT`: directly supported by an existing generated artifact/log.
- `PROVEN_FROM_RUNTIME`: supported by an existing runtime result; not inferred from source.
- `VIDEO_EVIDENCE`: observed in the training video.
- `DOCUMENTED_ONLY`: stated in docs/comments but not verified active.
- `INFERRED`: logical consequence of supported facts.
- `HYPOTHESIS`: candidate explanation/design; performance is `TO_BE_MEASURED`.
- `UNRESOLVED`: evidence is insufficient.

Architecture presence must not be converted into retrieval-quality superiority without a benchmark.

## Repository architecture summary

| System | Main retrieval shape | Visual representation | Text evidence | Temporal approach | Operator UI |
|---|---|---|---|---|---|
| AIC2024 | FastAPI + Milvus/ElasticSearch/MongoDB | OpenCLIP ViT-H/14 plus prototypes | EasyOCR / ElasticSearch | heuristic next-query time relation | React with video/neighbor concepts |
| AIC2025 | Streamlit + local files/FAISS | SigLIP2-384 + DINOv3 sidecar | Vintern JSON + Whisper/MiniLM | bounded keyframe-sequence expansion | Streamlit frame/video browsing |
| Panel2026 | FastAPI + React + VectorIndex abstraction | CLIP-B/32 | mock BM25 prototype | frequency/sort prototype | strongest audited interaction feature set |
| CurrentSystem | FastAPI + `ConfiguredSearch` + FAISS | SigLIP2-base-224 | OCR + Faster-Whisper ASR | TRAKE TemporalRefiner + monotonic DP | lightweight HTML/JS; lacks timeline/player |

Comparative retrieval accuracy across the four systems: **TO_BE_MEASURED**.

## CurrentSystem verified core

### Frame identity

`backend/app/video/video_decoder.py::iter_frames` assigns `source_frame_index_zero_based` using the ordinal from:

```python
for index, frame in enumerate(container.decode(stream)):
    ...
    yield DecodedFrame(index, ...)
```

`PROVEN_FROM_CODE`.

The authoritative identity therefore does not depend on `timestamp * FPS` reconstruction. This avoids FPS-derived reconstruction drift in the inspected authoritative path. End-to-end frame parity remains a runtime correctness gate.

`decode_frame_indices()` sorts requested ordinals, iterates `iter_frames(path)` from the start, and stops after the furthest requested ordinal. Structural decode work therefore grows with the furthest target ordinal. Wall-clock latency: **TO_BE_MEASURED**.

Current GPU video decoding in audited main: **NOT IMPLEMENTED**. GPU model inference must not be confused with GPU video decode.

### Sampling

The active audited sampler is timestamp-interval based, with `interval_seconds=1.0` as the default. “1 FPS” is therefore a default sampling setting, not an immutable frame-ID rule.

### Search orchestration

`ConfiguredSearch` implements active visual/OCR/ASR retrieval, QueryPlan/QueryRefiner routing, RRF fusion, deterministic token/rule candidate reranking, and a **60-frame** temporal NMS window.

The 60-frame threshold is a frame-ordinal invariant. “Approximately 2 seconds” is only true for ~30-fps material and is not the invariant.

### KIS dense refinement status

`backend/app/retrieval/kis_pipeline.py` contains `DenseTemporalRefiner` and `KISPipeline`. The refiner calls `decode_frame_indices()` for each coarse candidate and encodes that candidate's local neighborhood. It is not wired into the active `ConfiguredSearch` path in the audited baseline.

Status: `TEST_ONLY / UNWIRED`.

Naively wiring the existing implementation would repeat sequential sparse-frame decoding across candidates. Actual latency/quality impact: **TO_BE_MEASURED**.

### QA and TRAKE

CurrentSystem contains an active evidence-first QA architecture and an active TRAKE path with temporal refinement plus monotonic DP. `TRAKEAligner` defaults observed in the audit are `transition_penalty=0.0` and `max_gap=None`; therefore no active gap penalty should be claimed unless explicitly configured.

## Frame-ID forensics

| System | Audited frame identity | Assessment | Reason |
|---|---|---|---|
| AIC2024 | file/frame integer with FPS-derived time conversion in relevant paths | `RISKY` | susceptible to CFR/VFR assumptions; not proven universally wrong |
| AIC2025 | scene/keyframe coordinate mapped back through CSV | `ACCEPTABLE_WITH_MAPPING_RISK` | preserves original-frame mapping but adds a second coordinate space |
| Panel2026 | prototype uniform-sampling frame index | `RISKY` | tightly coupled to prototype sampling assumptions |
| CurrentSystem | PyAV emitted-frame zero-based ordinal | `ROBUST_ARCHITECTURAL_DESIGN` | authoritative ID avoids timestamp×FPS reconstruction; runtime parity still required |

## Visual representation comparison

| System | Model/path | Dim | Role |
|---|---|---:|---|
| AIC2024 | OpenCLIP ViT-H/14-378 (`dfn5b`) | 1024 | visual retrieval |
| AIC2024 alt | ALIGN-base | 640 | prototype/alternate path |
| AIC2025 | OpenCLIP SigLIP2-384 (`webli`) | 1536 | semantic keyframe retrieval |
| AIC2025 | DINOv3 ViT-L/16 | 1024 | dedicated image/instance-oriented sidecar path |
| Panel2026 | CLIP ViT-B/32 | 512 | prototype visual retrieval |
| CurrentSystem | `google/siglip2-base-patch16-224` | 768 | active visual retrieval |

AIC2025's 1536-dimension finding is source-supported by its actual embedding allocation; older comments reporting a different dimension must not override code.

For L2-normalized vectors, squared L2 distance and inner-product ranking are monotonic transformations of each other. This mathematical equivalence does not establish backend speed superiority.

## OCR / ASR / text retrieval

- **AIC2024:** EasyOCR + ElasticSearch text path.
- **AIC2025:** Vintern multimodal JSON substring search; Whisper transcript embeddings via MiniLM.
- **Panel2026:** mock BM25 text documents in the audited prototype.
- **CurrentSystem:** OCR/ASR evidence sidecars, separator-aware exact-term matching, lexical matching, and multi-channel RRF.

CurrentSystem has the **most complete audited active text-retrieval architecture** in this set, but comparative accuracy remains `TO_BE_MEASURED`.

Operational note: source code may contain a PaddleOCR CUDA path, but implementation existence is distinct from release-runtime validation. Do not infer validated production GPU OCR from source alone.

## Fusion and reranking

- AIC2024 uses direct score combination in audited paths.
- AIC2025 exposes modalities mostly through separate search flows.
- Panel2026 includes RRF over dense + mock BM25 results.
- CurrentSystem uses multi-channel RRF, deterministic token/string rule reranking, then temporal NMS.

`CandidateReranker` must be described as deterministic rule/token/string-based logic, not a learned semantic reranker.

## Temporal reasoning

- **AIC2024:** heuristic time-related next-query behavior.
- **AIC2025:** sequence expansion constrained by keyframe gaps. A safe growth description is `S_i <= S_(i-1) * G_i`; do not call it factorial without proof.
- **Panel2026:** prototype video-frequency selection and chronological sorting.
- **CurrentSystem:** coarse retrieval + `TemporalRefiner` + monotonic DP (`TRAKEAligner`) + coherence diagnostics.

CurrentSystem therefore has the **most complete audited temporal architecture**, not proven superior retrieval quality.

## Image and instance retrieval

AIC2025 contains a dedicated DINOv3 sidecar index. That makes it a useful **instance-retrieval experiment source**, not proof that DINOv3 outperforms SigLIP2 on the target corpus.

- DINOv3 vs SigLIP2 instance Precision@K: `TO_BE_MEASURED`.
- DINOv3 recommendation: `EXPERIMENT`.

CurrentSystem already has SigLIP2 image search and should retain it as the semantic baseline during any DINO sidecar experiment.

## Video decoding and sparse access

Candidate strategies for future benchmark, with no preselected winner:

1. current sequential PyAV reference;
2. grouped sequential decode for multiple requested regions;
3. cached/predecoded lookup for repeated operator access;
4. GOP-aware bounded seek prototype;
5. optional GPU-assisted sequential/batch decode.

Every candidate must pass exact authoritative-ordinal parity against the sequential reference before speed matters.

Cache vs bounded seek: **TO_BE_MEASURED BY WORKLOAD**. UI neighbor browsing and arbitrary uncached sparse access are different workloads.

## Operator workflow

Current UI provides `/` focus, `Ctrl+Enter` submission, result cards, score display, and copy behavior, but lacks:

- arrow-key result-grid navigation;
- neighbor/timeline strip;
- embedded video player/seek;
- frame stepping;
- direct “more like this” interaction from a result card;
- pinned candidate/session workflow.

Panel2026 and AIC2024 provide useful historical interaction concepts. The training video also demonstrates interactive inspection and temporal browsing as prominent workflow elements (`VIDEO_EVIDENCE`). No unsupported percentage of operator time is asserted.

## Diagnostics

CurrentSystem exposes richer runtime retrieval diagnostics than several audited systems (per-result channel scores and QueryPlan debugging), but lacks interactive dataset/embedding exploration, false-positive labeling, and hard-negative analysis tooling.

FiftyOne/VBS-style demonstrations from the training video are **capability references**, not recommendations to adopt a specific vendor.

## Result contracts and training-video evidence

The training video supports the value of strict result serialization, including `video_id`, integer frame identifiers, CSV handling, and QA quoting/escaping behavior. Official QA scoring semantics are **UNRESOLVED** from the currently audited video evidence.

Generic architecture should support typed KIS/QA/TRAKE result validation and JSON/JSONL/CSV adapters without making competition-specific submission code the retrieval core.

## Deployment and vector-backend position

CurrentSystem is configured for local model loading and local FAISS indexing. Complete air-gapped runtime verification is **not established by this audit** unless an existing runtime artifact is cited separately.

`KEEP FAISS DIRECT` is the current roadmap decision because there is no demonstrated requirement for another backend and additional backends add operational complexity. Comparative FAISS/Milvus/Qdrant performance: **NOT BENCHMARKED**. Milvus and Qdrant are not inherently cloud-only technologies.

## Historical concept decision matrix

| Concept | Source | Decision | Measurement status |
|---|---|---|---|
| Timeline / neighbor strip | Panel2026 | `ADOPT` as UI experiment | operator benefit `TO_BE_MEASURED` |
| Video player / seek | AIC2024 / Panel2026 | `ADOPT` as UI experiment | operator benefit `TO_BE_MEASURED` |
| DINOv3 sidecar index | AIC2025 | `EXPERIMENT` | quality `TO_BE_MEASURED` |
| TransNetV2 scene sampling | AIC2025 | `EXPERIMENT` | index-size/Recall tradeoff `TO_BE_MEASURED` |
| GPU decode concepts | AIC2025 / FFmpeg | `EXPERIMENT` | latency/throughput `TO_BE_MEASURED` |
| VectorStore backend expansion | Panel2026 | `REJECT FOR CURRENT ROADMAP` | no demonstrated need; perf not benchmarked |
| MongoDB spatial DB | AIC2024 | `REJECT FOR CURRENT ROADMAP` | no validated target-task requirement |
| WordNet expansion | AIC2024 | `REJECT FOR CURRENT ROADMAP` | semantic/query drift risk; impact unmeasured |
| hardcoded submission endpoints | historical | `REJECT` | product-architecture mismatch |

## Final evidence-grounded answers

1. **Frame-ID architecture:** CurrentSystem has the most robust audited design because authoritative identity comes from PyAV emitted-frame ordinal rather than FPS reconstruction.
2. **Ingestion idea worth testing:** AIC2025 TransNetV2 scene sampling, with Recall/index-cardinality tradeoff `TO_BE_MEASURED`.
3. **Temporal architecture:** CurrentSystem is the most complete audited implementation; quality remains unmeasured across systems.
4. **Text architecture:** CurrentSystem is the most complete audited active visual+OCR+ASR design; accuracy remains unmeasured.
5. **Operator workflow:** Panel2026 provides the richest audited interaction pattern.
6. **Unwired current component:** `DenseTemporalRefiner` / `KISPipeline`.
7. **Primary sparse-access structural issue:** `decode_frame_indices()` decodes sequentially from frame zero to the furthest target.
8. **DINOv3:** worth an isolated sidecar experiment; comparative quality `TO_BE_MEASURED`.
9. **TransNetV2:** worth a sampling experiment; gains `TO_BE_MEASURED`.
10. **Changing FAISS:** not justified by current requirements; backend performance is `NOT BENCHMARKED`.
11. **Most important missing benchmark:** broad multi-task ground truth plus frozen baseline metrics.
12. **Most important UI gap:** temporal neighbor inspection + video verification.
13. **GOP-aware seek:** experiment candidate only; correctness parity is mandatory.
14. **Cache vs seek winner:** `TO_BE_MEASURED BY WORKLOAD`.
15. **Implementation before evaluation:** none of Dense KIS, DINO, TransNet, decoder replacement, or score tuning should be promoted based on architecture alone.
