# Project Control CLI

`projectctl.py` is the operator entrypoint. It orchestrates existing modules and does not implement retrieval, indexing, OCR, ASR, KIS, Q&A, or TRAKE algorithms itself.

```bash
python projectctl.py doctor
python projectctl.py status
python projectctl.py info

python projectctl.py backend
python projectctl.py frontend
python projectctl.py dev

python projectctl.py ingest /path/to/videos
python projectctl.py preprocess /path/to/videos
python projectctl.py ocr /path/to/video-or-directory
python projectctl.py asr /path/to/video-or-directory
python projectctl.py index

python projectctl.py search "query"
python projectctl.py kis "query"
python projectctl.py qa "question"
python projectctl.py trake '["event one", "event two"]'

python projectctl.py evaluate --competition --ground-truth data/competition/ground_truth
python projectctl.py benchmark
python projectctl.py test
python projectctl.py clean --staging
```

Common options are `--json`, `--verbose`, and `--output PATH`. Search output is JSON by default with `--json`; an `.csv` output path writes tabular KIS/Q&A candidates. TRAKE accepts a JSON array, an object containing `events`, a JSON file, or pipe-separated event text.

Runtime configuration:

```bash
export VIDEO_PROCESSED_ROOT=/path/to/processed/videos
export SEARCH_MODEL_DEVICE=cpu
```

`preprocess` resumes M15 ingest by default, publishes the visual index, then runs OCR and ASR independently. Missing OCR is reported and skipped; it does not prevent visual ingestion or ASR. Explicit `ocr` fails with an actionable install message when Tesseract is absent. Faster Whisper model weights must already be cached or be downloadable.

`doctor` is the authoritative readiness summary. It checks FFmpeg/Faster Whisper, Tesseract, English/Vietnamese OCR language packs, competition ground truth, configured production index, and optional browser automation.

The server commands expose the same FastAPI application because the current frontend is served by FastAPI. `frontend` is retained as an operator alias rather than introducing a second development server.
