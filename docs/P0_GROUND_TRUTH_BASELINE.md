# P0 Ground-Truth and Baseline Evaluator

This evaluation-only utility implements the first measurable gate from the master evaluation roadmap without changing production retrieval behavior.

## Scope

- validates multi-task JSONL ground truth for KIS, QA, and TRAKE;
- requires category IDs `C01` through `C18` but does not fabricate category coverage;
- uses annotated authoritative-frame intervals rather than a universal `±N` tolerance;
- computes KIS `VR@K`, `FIR@K`, `MRR`, and `MFD`;
- computes QA evidence recall plus an **internal** normalized exact-match diagnostic;
- computes TRAKE event-hit recall, complete-sequence accuracy, monotonicity, and optional mean event-frame error;
- captures prediction latency percentiles only when latency values are supplied;
- writes JSON and Markdown reports with Git/generation/runtime provenance fields.

Official QA scoring semantics remain unresolved; the evaluator's `ema_internal` metric must not be presented as the official scoring rule.

## Input contracts

Ground truth is JSONL with one record per query. See `eval/p0_ground_truth.example.jsonl` for schema-only examples. Those records are intentionally synthetic and **must not** be included in benchmark scores.

Predictions are JSONL with this common shape:

```json
{"query_id":"GT_C01_001","latency_ms":12.3,"results":[{"video_id":"L22_V001","frame_id":1234}]}
```

QA result rows may include `answer`. TRAKE result rows use `frame_ids` for the ordered event sequence.

## Frozen CurrentSystem run

```bash
python -m eval.p0_run_current \
  --ground-truth path/to/ground_truth.jsonl \
  --processed-root path/to/processed \
  --output artifacts/p0_predictions.jsonl \
  --manifest artifacts/p0_run_manifest.json \
  --top-k 20
```

The runner calls the existing `ConfiguredSearch.handle()` path for KIS, QA, and TRAKE and records wall-clock latency without modifying retrieval configuration. Use explicit `--no-query-refine`, `--no-rerank`, or `--no-temporal-refine` only for later ablation runs, not for the frozen baseline unless the baseline definition says so.

## Evaluate predictions

```bash
python -m eval.p0_baseline \
  --ground-truth path/to/ground_truth.jsonl \
  --predictions artifacts/p0_predictions.jsonl \
  --output-json artifacts/p0_baseline.json \
  --output-md artifacts/p0_baseline.md \
  --generation-id gen-... \
  --runtime-label cpu-or-gpu-description
```

No model download or inference occurs inside the evaluator itself.

## Correctness rules

- frame intervals are inclusive and expressed in authoritative zero-based source ordinals;
- `exact_frame_id`, when supplied, must lie inside its valid interval;
- TRAKE intervals must be chronologically ordered;
- complete TRAKE sequences must be strictly monotonic;
- frame-ID mismatch tolerance for decoder parity remains zero;
- missing predictions are reported, not silently discarded.

## P0 status

The repository now has the measurement machinery, but the master 18-category benchmark is **not yet populated** by this branch. The example JSONL is schema documentation only. Real baseline metrics require verified annotated data plus the processed index/model artifacts used by CurrentSystem.
