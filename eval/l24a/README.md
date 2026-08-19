# Videos_L24_a evaluation

This directory contains the frozen human-first counting benchmark used by PR #10.

## Order of operations

1. A human reviewer inspects all 43 videos using 16 uniform overview samples plus 20/40/60/80% detail samples.
2. Human summaries and counting answers are frozen before any CurrentSystem output is inspected.
3. GitHub Actions reproduces the 20-sample-per-video review pool from the public archive.
4. `scripts/evaluate_l24a_counts.py` runs the repository's active SigLIP2 encoder and QA query decomposition over those samples.
5. Counting answers are evaluated separately using the active extractive QA answerer over Tesseract OCR from both the retrieved top-1 frame and the human evidence frame (oracle retrieval).

The benchmark intentionally separates scene retrieval from counting-answer capability. The current extractive QA path reads numeric OCR/ASR evidence; it is not treated as a pixel-level object counter.

No authoritative frame ID is derived from timestamp × FPS. Human evidence timestamps are localization anchors only.

## Frozen inputs

- `human_video_analysis.md` — human visual summaries for all 43 videos.
- `human_count_gt.jsonl` — 43 frozen counting questions, one per video, with confidence labels.

## Action outputs

The L24a PR gate produces:

- inventory metadata for all extracted videos;
- a reproducible 860-frame review pool (20 samples × 43 videos);
- video Recall@1/5/20 for the counting queries;
- rank of the human evidence sample;
- timestamp localization error within the correct video;
- oracle-frame OCR QA answer coverage and numeric exact match;
- top-1-frame numeric exact match;
- per-target-type breakdown and a Markdown comparison report.

These measurements are a controlled sampled-frame diagnostic. They are not presented as a full production 1-second-ingest benchmark.
