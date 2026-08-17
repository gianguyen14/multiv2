# Master Benchmark Plan

## 1. Purpose

This plan defines the evidence needed before changing retrieval architecture. It establishes a frozen baseline, broad multi-task ground truth, failure attribution, system/resource metrics, and controlled ablations.

No architecture change may be declared better on wall-clock speed alone if authoritative frame-ID parity fails.

Unknown performance thresholds remain `TO_BE_ESTABLISHED_FROM_BASELINE`.

## 2. Benchmark dataset design

Build a compact but representative multi-task dataset spanning 18 categories:

| ID | Category | Main evidence |
|---|---|---|
| C01 | Easy semantic KIS | salient visual scene |
| C02 | Hard semantic KIS | subtle action/object and adjacent-frame distractors |
| C03 | OCR exact text | signs, banners, names |
| C04 | plates / numeric codes | compact alphanumeric strings |
| C05 | ASR speech | timestamped spoken fragment |
| C06 | visual distractors | similar scene, wrong location/object |
| C07 | repeated scenes | same-looking recurring segment |
| C08 | landmark/location instance | same location vs generic look-alike |
| C09 | object instance | same logo/package/vehicle/clothing |
| C10 | near duplicate | re-encoded/trimmed/duplicated clip |
| C11 | long video | target late in long source |
| C12 | sparse late-frame access | arbitrary late source ordinal |
| C13 | TRAKE 3-stage | ordered multi-event sequence |
| C14 | tight TRAKE | rapidly consecutive events |
| C15 | wide TRAKE | events separated by large time gaps |
| C16 | QA obvious evidence | direct local evidence |
| C17 | QA multi-frame/evidence | evidence spread across local context |
| C18 | negative/ambiguous | abstention / no valid target |

The final sample size is **not hardcoded**. It should be selected from category coverage and a precision/power requirement after a pilot.

## 3. Ground-truth schema

Use JSONL. Core fields:

```json
{
  "query_id": "GT_C04_001",
  "category_id": "C04",
  "task_type": "kis",
  "query": "Xe tải mang biển số 50H-052.03 chạy trên đường",
  "query_vi": "...",
  "query_en": "...",
  "exact_terms": ["50H-052.03", "50H 052.03", "50H05203"],
  "ground_truth": {
    "video_id": "L22_V001",
    "exact_frame_id": 1500,
    "valid_frame_interval": [1470, 1530],
    "timestamp_range_seconds": [49.0, 51.0]
  },
  "qa_ground_truth": null,
  "trake_ground_truth": null,
  "hard_negatives": [],
  "notes": "human annotation note"
}
```

Primary frame correctness/quality evaluation should use the **annotated valid authoritative-frame interval**. Do not impose a universal ±5-frame tolerance across different frame rates.

An exact representative frame may still be stored for frame-distance analysis.

## 4. KIS metrics

For top-K results:

- **Video Recall@K (VR@K):** target video appears in top K.
- **Frame Interval Recall@K (FIR@K):** at least one top-K result lands in the annotated valid frame interval in the correct video.
- **MRR:** reciprocal rank of the first valid interval hit.
- **Mean Frame Distance (MFD):** distance from a chosen representative exact annotation for hits, reported as a secondary localization diagnostic.

Required K values should include at least the operator-relevant range (e.g. 1/5/20), but acceptance targets are established only after baseline capture.

## 5. QA metrics

Separate retrieval and answer stages:

- Evidence Video Recall@K;
- Evidence Frame Recall@K;
- Evidence-Grounded Answer Rate;
- Abstention precision/recall for unanswerable cases;
- answer-quality metric(s) selected after annotation review.

A normalized **Exact Match Answer Accuracy (EMA)** may be used as a `PROPOSED_INTERNAL_METRIC` for suitable answer types. Official QA scoring semantics are `UNRESOLVED`; internal EMA is not asserted to reproduce official scoring.

## 6. TRAKE metrics

For ordered events:

- **Valid Monotonic Sequence Rate (VMSR)**;
- **Event Hit Recall (EHR)**;
- **Complete Sequence Accuracy (CSA)**;
- **Mean Event Frame Error (MEFE)** or interval-distance equivalent.

Strict event order is a logical correctness requirement for monotonic tasks. Parameter quality is not inferred from that invariant.

## 7. Text retrieval metrics

On OCR/ASR subsets capture:

- exact/normalized term hit rate;
- false-positive substring rate;
- accent/diacritic variation behavior;
- plate/code separator variation behavior;
- speech query top-K hit rate;
- per-channel contribution and failure taxonomy.

## 8. Frame-ID correctness gates

### Reference

The sequential PyAV emitted-frame ordinal is the authoritative reference in the audited CurrentSystem baseline.

### Gates

1. decode ordinal increments exactly `0..N-1` for the reference decode;
2. every alternative sparse-access strategy must return the same authoritative source ordinal as the reference;
3. API/result round-trip must preserve the same `video_id` + source ordinal identity;
4. frame-ID mismatch tolerance = **0**.

Synthetic CFR/VFR/B-frame fixtures are useful for regression coverage, but no claim of universal container correctness should be made from static source inspection alone.

## 9. Sparse decode benchmark

Benchmark distinct workloads; do not preselect a winner.

| Strategy | Workload | Required outputs |
|---|---|---|
| D1 current sequential PyAV | reference contiguous/sparse access | parity reference, p50/p95, frames decoded, CPU |
| D2 grouped sequential decode | many regions/candidates in same video | parity, total decode calls, frames decoded, p50/p95 |
| D3 cache/predecoded lookup | repeated UI neighbor/preview access | hit/miss latency, storage, parity |
| D4 GOP-aware bounded seek prototype | uncached late sparse access | parity, p50/p95, decoded frames, failure modes |
| D5 optional GPU-assisted decode | batch/sequential extraction | parity, throughput, VRAM, CPU/GPU utilization |

For every alternative:

- authoritative ordinal parity;
- requested-frame coverage;
- p50/p95 latency;
- frames decoded;
- CPU/GPU/RAM/VRAM as applicable;
- codec/container metadata;
- cache hit/miss state where relevant.

No `<150ms`, `<200ms`, or other latency target is assumed before the baseline/product requirement is known.

## 10. Sampling benchmark

Compare:

1. timestamp-interval sampling at current default (`1.0s`);
2. a coarser timestamp interval;
3. TransNetV2 scene-midpoint sampling;
4. hybrid scene + timestamp coverage.

Capture:

- vector count/index cardinality;
- disk footprint;
- ingest time per hour of source;
- FIR@K / task-specific recall;
- sampling-miss taxonomy;
- authoritative frame mapping integrity.

TransNetV2 index-size and quality gains are `TO_BE_MEASURED`.

## 11. Dense KIS A/B

Run only after the sparse-access reference benchmark exists.

- **A:** active CurrentSystem KIS baseline.
- **B:** coarse candidate retrieval + controlled dense local refinement.

Capture:

- FIR@1/5/20;
- MRR;
- MFD/localization error;
- p50/p95 query latency;
- frames decoded/query;
- image embeddings/query;
- decode calls/video;
- fallback/error count.

Dense refinement quality gain: `TO_BE_MEASURED`.

Do not directly wire the old per-candidate refiner into production as part of this benchmark.

## 12. TRAKE parameter study

Current active baseline parameters include `transition_penalty=0.0` and `max_gap=None`.

Future study should compare a bounded grid of penalty/gap candidates selected from corpus duration/frame statistics. The exact grid is a **proposed experiment configuration**, not a universal optimum.

Capture CSA, EHR, MEFE/interval error, invalid sequence count, and alignment latency.

## 13. DINOv3 instance study

Keep CurrentSystem SigLIP2 as the semantic-image baseline and add DINOv3 only as an isolated sidecar experiment.

Use instance GT for logos, vehicles, landmarks, clothing/packages, and near duplicates.

Capture Precision/Recall@K and false-positive categories.

DINOv3 quality advantage: `TO_BE_MEASURED`.

## 14. OCR / ASR / pipeline ablations

Run controlled variants such as:

- visual only;
- visual + OCR;
- visual + ASR;
- full visual + OCR + ASR;
- QueryRefiner ON/OFF;
- CandidateReranker ON/OFF;
- temporal NMS ON/OFF where safe.

Each query must run against the same frozen corpus/index generation unless the experiment explicitly studies ingestion/sampling.

## 15. Operator workflow evaluation

First capture the **current UI baseline**, then compare an isolated timeline/neighbor prototype.

Metrics:

- time to first relevant candidate;
- time to exact accepted frame;
- query reformulation count;
- clicks/keystrokes;
- candidate inspections;
- task completion rate;
- operator error/failure category.

No unsupported percentage of operator time is assumed.

## 16. Resource metrics

Where relevant capture:

- p50/p90/p95 latency;
- peak RSS;
- VRAM allocation;
- CPU/GPU utilization;
- disk I/O;
- frames decoded/query;
- image embeddings/query;
- index search time;
- reranker/fusion/temporal times.

## 17. Proposed statistical methodology

This section is a **proposal**, not a pre-registered final design.

- Pilot first to estimate category variance and practical effect sizes.
- Determine final sample size from category coverage and desired precision/power.
- Prefer paired bootstrap confidence intervals for metric differences such as ΔFIR@K and ΔMRR.
- Use paired permutation/randomization tests where metric properties make them appropriate.
- Select final tests based on bounded/rank metric behavior and sample size.

Do not default to a Student t-test simply because it is familiar. Bootstrap resample count and alpha level are implementation-time decisions that should be documented when the benchmark is frozen.

## 18. Acceptance criteria policy

| Gate/metric | Target |
|---|---|
| authoritative frame-ID parity | 0 mismatches |
| required monotonic TRAKE output | valid strict order where task contract requires it |
| offline local execution | PASS required as a future operational gate; current audit does not itself prove runtime success |
| FIR@20 | `TO_BE_ESTABLISHED_FROM_BASELINE` |
| p95 latency | `TO_BE_ESTABLISHED_FROM_BASELINE` |
| operator TTEF target | `TO_BE_ESTABLISHED_FROM_BASELINE_AND_PRODUCT_REQUIREMENT` |

## 19. Execution order

```text
Stage 1 — correctness / provenance
  frame-ID reference, index generation, serialization, environment capture

Stage 2 — P0 baseline retrieval
  KIS + OCR + ASR + QA + TRAKE baseline metrics

Stage 3 — component ablations
  channels, QueryRefiner, CandidateReranker, temporal NMS

Stage 4 — P1-A operator workflow
  current UI baseline -> isolated timeline/neighbor prototype evaluation

Stage 5 — P1-B sparse frame access
  D1 sequential vs D2 grouped vs D3 cache vs D4 GOP-seek vs optional D5 GPU

Stage 6 — P1-C dense KIS
  dense-refinement A/B after sparse-access baseline exists

Stage 7 — P2 experiments
  DINOv3 sidecar, TransNetV2/hybrid sampling, TRAKE penalty/max_gap study
```

## 20. Required benchmark metadata

Every result artifact should record:

- Git SHA / branch;
- index generation ID;
- dataset/GT version;
- model identifiers/revisions;
- device/runtime versions;
- feature flags;
- query IDs/categories;
- wall-clock measurement method;
- random seeds where relevant;
- failures/exclusions with reasons.

This is required before any benchmark number is used to justify implementation work.
