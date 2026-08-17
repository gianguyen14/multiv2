# Evidence-Grounded System Improvement Roadmap

## 1. Principles

This roadmap is evaluation-first. Architectural capability, historical precedent, or theoretical appeal is not sufficient to promote a feature into production.

Preserve these CurrentSystem invariants unless a dedicated correctness study proves an alternative:

- authoritative frame identity from zero-based PyAV emitted-frame ordinal;
- local-first deployment and local model loading;
- current SigLIP2 + FAISS baseline for semantic retrieval;
- task-specific KIS / QA / TRAKE behavior separation;
- explicit measurement before score/model/decoder tuning.

Comparative retrieval quality and latency claims remain `TO_BE_MEASURED` unless backed by benchmark artifacts.

## 2. Verified current strengths

- PyAV ordinal-based authoritative frame-ID policy avoids timestamp×FPS reconstruction in the inspected path.
- Active QueryRefiner/QueryPlan multi-path routing.
- Active visual + OCR + ASR fusion through RRF.
- Deterministic token/string/rule CandidateReranker.
- Active evidence-first QA structure.
- Active TRAKE TemporalRefiner + monotonic DP.
- Local FAISS index and local-files-only model-loading design.
- Existing CI/test infrastructure is present; exact total test count is not established by this evaluation baseline.

These are architectural facts, not empirical superiority claims.

## 3. Current gaps

Severity is an architectural planning judgment:

- **CRITICAL:** blocks scientific evaluation or a core operator task.
- **HIGH:** material architectural/operator/runtime risk that warrants targeted evaluation.
- **MEDIUM:** optional capability or configuration question with meaningful possible value.
- **LOW:** maintenance/cosmetic concern.

| Gap | Severity | Status | Required evidence |
|---|---|---|---|
| insufficient broad multi-task GT | CRITICAL | partial validation only | P0 annotation + baseline |
| missing timeline/neighbor UI | HIGH | not implemented | operator baseline/prototype study |
| missing video verification/player flow | HIGH | not implemented | operator study |
| sparse on-demand decode is sequential from frame zero | HIGH | active structural behavior | D1-D5 access benchmark |
| KIS DenseTemporalRefiner is unwired | HIGH | test-only/unwired | dense A/B after sparse-access baseline |
| no dedicated instance sidecar | MEDIUM | not implemented | DINO vs SigLIP2 instance study |
| TRAKE penalty/max-gap defaults may be untuned | MEDIUM | active defaults 0.0 / None | parameter study |
| limited dataset/embedding diagnostics | MEDIUM | partial diagnostics | benchmark/debug workflow study |
| no generic typed result/export validation layer | MEDIUM | partial/manual | serialization contract tests |

## 4. Existing-but-unwired KIS refinement

`backend/app/retrieval/kis_pipeline.py` contains `DenseTemporalRefiner` and `KISPipeline`.

The current refiner performs a local-frame request via `decode_frame_indices()` for each coarse candidate. Since `decode_frame_indices()` iterates from source ordinal zero to the furthest requested target, repeated candidate refinement repeats sequential decode work.

Structural risk: repeated decode work grows with candidate location and candidate count.

Actual wall-clock cost: `TO_BE_MEASURED`.

Actual FIR/MRR gain: `TO_BE_MEASURED`.

**Roadmap decision:** do not wire the old implementation directly into active KIS before the sparse-access benchmark.

## 5. Operator workflow gap

Training-video evidence and historical UI code both support evaluating a stronger operator loop:

```text
query
 -> result grid
 -> keyboard navigation
 -> inspect candidate
 -> neighbor/timeline strip
 -> video verification/frame step
 -> optional image/similarity refinement
 -> select/copy/export
```

Candidate capabilities:

- arrow-key result navigation;
- timeline/neighbor frames;
- embedded video player/seek;
- frame stepping;
- “more like this” from selected frame;
- pinned candidates/history.

No user-time percentage or productivity gain is assumed before operator measurement.

## 6. Diagnostics gap

CurrentSystem already exposes per-result score components and QueryPlan debugging. Missing developer/operator capabilities include:

- channel contribution inspector;
- false-positive/false-negative annotation;
- dataset browser;
- embedding-neighbor explorer;
- duplicate/hard-negative investigation;
- benchmark failure taxonomy views.

FiftyOne/VBS-style tooling from the training video is a conceptual capability reference, not a mandated dependency.

## 7. Sparse-access strategy

No production winner is selected.

Evaluate by workload:

1. sequential PyAV reference;
2. grouped sequential decode;
3. preview/thumbnail/predecoded cache;
4. GOP-aware bounded-seek prototype;
5. optional GPU-assisted decode.

The benchmark must separate:

- repeated operator neighbor browsing;
- one-off uncached late sparse access;
- multi-candidate dense refinement;
- batch ingestion/extraction.

Cache vs bounded seek: `TO_BE_MEASURED BY WORKLOAD`.

Any alternative that fails authoritative ordinal parity is rejected regardless of speed.

## 8. Historical ideas worth testing

| Concept | Source | Decision | Why |
|---|---|---|---|
| timeline/neighbor strip | Panel2026 | ADOPT as experiment | low retrieval-core risk; operator value measurable |
| video player/seek | AIC2024/Panel2026 | ADOPT as experiment | contextual verification capability |
| DINOv3 sidecar | AIC2025 | EXPERIMENT | dedicated instance-oriented representation path |
| TransNetV2 sampling | AIC2025 | EXPERIMENT | tests index-cardinality vs recall tradeoff |
| GPU decode concept | AIC2025/FFmpeg | EXPERIMENT | possible throughput/sparse-access value; no current main implementation |
| generic result serializer/validator | training/historical workflows | ADOPT | decouples retrieval from output transport |

## 9. Ideas not recommended for the current roadmap

### Vector backend expansion

`KEEP FAISS DIRECT` for the current roadmap.

Reason: no demonstrated requirement for another backend, while additional backends increase operational surface area. This is **not** a claim that FAISS is universally faster or that Milvus/Qdrant cannot run locally.

Comparative backend performance: `NOT BENCHMARKED`.

### MongoDB spatial engine

Not recommended without a target-task GT showing spatial predicates are needed. The concern is additional storage/query complexity, not a claim that MongoDB is intrinsically bad.

### WordNet expansion

Not recommended because it introduces semantic/query drift risk relative to the current structured QueryRefiner path. Comparative retrieval impact: `TO_BE_MEASURED`.

### Raw heterogeneous score addition

Not recommended as the default fusion strategy because heterogeneous raw score scales are combined without explicit calibration. Comparative quality impact: `TO_BE_MEASURED`.

### Competition-specific hardcoded submission endpoints

Reject from retrieval core. Preserve only generic result-contract/export concepts.

## 10. Priority order

### P0 — establish evidence

1. expand multi-task GT across the 18 benchmark categories;
2. freeze CurrentSystem baseline provenance/index/model settings;
3. run baseline KIS/OCR/ASR/QA/TRAKE metrics;
4. assign failure taxonomy labels;
5. capture current operator workflow baseline;
6. validate generic result serialization and frame-ID correctness.

No model/score/decoder tuning before this baseline exists.

### P1-A — operator workflow evaluation

Build an isolated prototype/evaluation path for:

- neighbor/timeline browsing;
- keyboard navigation;
- video verification.

Compare TFRR/TTEF, task completion, reformulations and interactions against the current UI baseline.

### P1-B — sparse frame access benchmark

Compare sequential reference, grouped sequential, cache, GOP-aware prototype, and optional GPU-assisted decoding.

Select by workload using parity + latency/resource evidence.

### P1-C — dense KIS A/B

Only after P1-B establishes safe sparse-access baselines.

Compare coarse KIS vs controlled dense refinement using FIR@K/MRR/localization/resource metrics.

### P2 — representation, sampling and temporal experiments

- DINOv3 sidecar instance index;
- TransNetV2/hybrid sampling;
- TRAKE transition-penalty/max-gap study.

All P2 gains remain `TO_BE_MEASURED`.

## 11. Conceptual next architecture

This is a target shape, not a promise that every optional block will ship.

```text
Operator UI
  search grid
  keyboard navigation
  timeline / neighbors [P1-A experiment]
  video verification [P1-A experiment]
        |
        v
FastAPI orchestration
        |
        v
Query intelligence
  QueryRefiner / QueryPlan
        |
        +-----------------------+
        |           |           |
        v           v           v
     Visual        OCR         ASR
   SigLIP2      evidence    evidence
        \           |           /
         \----------+----------/
                    v
               RRF + rules
                    v
             temporal NMS
                    v
       +------------+------------+
       |            |            |
       v            v            v
      KIS           QA          TRAKE
 dense A/B      evidence      TemporalRefiner
 [P1-C]         answerer      + monotonic DP
       |
       v
Result contract / diagnostics / export
       |
       v
Storage / sparse access
  FrameStore + text evidence + FAISS
  sparse strategy TO_BE_SELECTED_BY_BENCHMARK
```

A DINOv3 sidecar, TransNetV2 sampling, tuned TRAKE parameters, or new sparse decoder are **experimental blocks**, not baseline requirements.

## 12. What not to implement yet

Do not yet:

- replace SigLIP2;
- change FAISS backend;
- wire old DenseTemporalRefiner directly into active KIS;
- select GOP-aware seek as the production decoder without parity/benchmark evidence;
- assume cache is always better than seeking;
- add DINOv3 to the main ranking path;
- switch ingestion to TransNetV2;
- tune RRF/reranker weights without GT;
- enable non-zero TRAKE penalties without a parameter study;
- reintroduce competition-specific submission endpoints into core architecture.

## 13. Promotion gate for any experiment

An experiment is eligible for production consideration only when it has:

1. reproducible branch/SHA and configuration;
2. frozen GT version and baseline;
3. correctness parity gates;
4. task-specific quality metrics;
5. latency/resource metrics where applicable;
6. failure taxonomy review;
7. no regression outside its intended task;
8. documented rollback/default-off strategy where risk warrants it.

This roadmap ends at evaluation and experiment selection; implementation priority is decided from resulting evidence.
