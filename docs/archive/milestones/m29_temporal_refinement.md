# M29 — Temporal / KIS / TRAKE Refinement

## 1. Measured Temporal Issues
- Prior to M29, redundant near-duplicate frames from the same video clustered around top ranks, preventing distinct temporal events from appearing in top 20/50/100.
- TRAKE alignment outputs did not format `events` array with `frame_id`, causing evaluator dictionary key mismatches.

## 2. Implemented Refinements
1. **Temporal Non-Maximum Suppression (NMS)**: Suppresses redundant frame detections within 90 frames (3.0s) per video while preserving peak scoring candidates.
2. **TRAKE Sequence Refinement**: Added monotonic path constraints and normalized output schema (`frame_ids` and `events` structure).

## 3. Measured Impact

| Metric | M28 Baseline | M29 Refined | Delta |
| :--- | :---: | :---: | :---: |
| **KIS R@1** | 0.26 | **0.26** | 0.00 |
| **KIS R@5** | 0.37 | **0.37** | 0.00 |
| **KIS R@20** | 0.42 | **0.53** | **+0.11** |
| **KIS R@100** | 0.68 | **0.79** | **+0.11** |
| **TRAKE Ordering Validity** | 0.00 | **1.00** | **+1.00** |

## 4. Decision
- **Status**: ACCEPTED.
