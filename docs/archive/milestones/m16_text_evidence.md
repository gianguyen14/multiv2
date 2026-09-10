# M16 Text Evidence

M16 adds offline OCR and ASR enrichment to persisted M15 videos. `M16TextPipeline` reads sampled `FrameRecord` images, writes `ocr.json` and `asr.json` atomically beside each video manifest, and reuses both files on resume.

Text records retain raw text and a Unicode-NFC, whitespace-normalized, case-folded search form. OCR remains attached to the authoritative `frame_uid` and zero-based decoder ordinal. ASR time boundaries map to the nearest observed sequentially decoded frame timestamp; frame identity is never calculated from timestamp × FPS.

`TesseractOCRBackend` is CPU-only and invokes a locally installed Tesseract executable. `FasterWhisperASRBackend` uses the existing `faster-whisper` dependency on CPU with int8 computation by default. Both are injected behind small backend interfaces so normal tests remain deterministic and offline. Repeated identical OCR text inside the configured temporal gap is suppressed while later reappearances are retained.

The benchmark command reports OCR time, ASR time, wall clock, and realtime factor separately:

```bash
python eval/m16_text_benchmark.py \
  --video /path/to/video.mp4 \
  --processed-root data/processed/videos \
  --whisper-model small \
  --ocr-languages eng+vie
```

The video must first be ingested by M15. The current development environment includes Faster Whisper but no OCR executable, so real OCR benchmarking requires installing Tesseract with English and Vietnamese language data. Synthetic backend tests validate persistence, normalization, mapping, resume, blank media, and failure isolation without making production accuracy claims.
