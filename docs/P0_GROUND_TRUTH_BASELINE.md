# P0 Ground-Truth and Baseline Evaluator

This branch adds evaluation-only machinery for the first measurable gate in the retrieval roadmap. Production retrieval code, model configuration, indexes, and ranking behavior are unchanged.

## Components

- `evaluation/p0_baseline.py` — dependency-free evaluator for precomputed KIS / QA / TRAKE predictions.
- `evaluation/p0_run_current.py` — runs the existing `ConfiguredSearch.handle()` path and normalizes its results into evaluator input JSONL.
- `evaluation/fixtures/p0_smoke_gt.jsonl` and `p0_smoke_predictions.jsonl` — synthetic CI smoke fixtures only; they are **not** real benchmark ground truth.
- `tests/unit/test_p0_baseline_evaluator.py` — evaluator semantics.
- `tests/unit/test_p0_run_current.py` — CurrentSystem routing and output-normalization semantics.
- `.github/workflows/p0-baseline.yml` — Python 3.12 dependency-free correctness/smoke gate.

## Measurement rules

- KIS is evaluated against annotated authoritative-frame intervals; there is no universal `±N` frame tolerance.
- KIS metrics include `VR@1/5/20`, `FIR@1/5/20`, `MRR`, and first-hit frame distance when an exact frame is annotated.
- QA evidence localization is separated from answer scoring.
- QA exact match is an **internal diagnostic metric only**; official external QA scoring semantics remain `UNRESOLVED`.
- TRAKE evaluates event hit recall, complete sequence accuracy, strict monotonicity, and optional event-frame error.
- Latency percentiles are reported only from supplied measured `latency_ms` values.
- Performance thresholds remain `TO_BE_ESTABLISHED_FROM_BASELINE`.
- Synthetic smoke metrics must never be reported as real retrieval quality.

## Ground-truth shape

The evaluator accepts one JSON object per line. Existing smoke fixtures show the exact schema used by this branch.

KIS / QA localization uses:

```json
{
  "video_id": "L22_V001",
  "exact_frame_id": 1234,
  "valid_frame_intervals": [[1220, 1250]]
}
```

QA stores `answerable` and `accepted_answers` inside `ground_truth`. TRAKE stores `event_intervals` and optional `exact_event_frames` inside `ground_truth`.

The master 18-category dataset is **not fabricated by this branch**. `category_id` is preserved for later category coverage reporting, but real annotations must come from verified corpus evidence.

## Run the frozen CurrentSystem

```bash
python evaluation/p0_run_current.py \
  --ground-truth path/to/real_ground_truth.jsonl \
  --processed-root path/to/processed \
  --output artifacts/p0_predictions.jsonl \
  --manifest artifacts/p0_run_manifest.json \
  --top-k 20
```

By default the runner preserves the existing baseline behavior: query refinement ON, candidate reranking ON, and TRAKE temporal refinement ON. The `--no-*` flags exist for later controlled ablations and should not be mixed into the frozen baseline without recording them in the manifest.

The run manifest captures Git SHA, task counts, flags, wall-clock total, and available `ConfiguredSearch.status()` / `readiness()` diagnostics including generation information when exposed by the current runtime.

## Evaluate the predictions

```bash
python evaluation/p0_baseline.py \
  --ground-truth path/to/real_ground_truth.jsonl \
  --predictions artifacts/p0_predictions.jsonl \
  --output artifacts/p0_baseline_results.json \
  --scope REAL_BASELINE
```

Use `SYNTHETIC_CI_SMOKE` only for the committed synthetic fixtures. A report may be called a real baseline only when the GT annotations, processed generation, model cache/runtime, and run manifest are all verified.

## P0 status

The branch provides the measurement harness and CI correctness gate. It does **not** claim that the 18-category master dataset is complete and it does **not** contain real retrieval-quality results yet. Real baseline execution requires the verified local annotation set plus the processed CurrentSystem artifacts.
