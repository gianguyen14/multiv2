# M15 Video Ingestion

M15 is an offline, CPU-first pipeline that inspects a video, sequentially decodes its original frames with PyAV, samples coarse temporal representatives, persists images and exact mappings, reuses SigLIP2 for normalized image embeddings, and publishes a FAISS index with resolvable frame payloads.

## Frame identity

`source_frame_index_zero_based` is the zero-based position assigned while frames are emitted by the decoder in display order. It is the authoritative original-frame identity. **Timestamp × FPS MUST NOT be treated as authoritative source-frame identity.**

Each record also stores a configurable `submission_frame_id`. The `zero_based` policy copies the internal index; `one_based` adds one. Both values remain persisted.

## Timing and VFR

PTS and timestamp are observed decoder values. They may be absent and are never invented. Average and real frame rates, time base, reported frame count, and observed decoded count remain distinct metadata. Coarse sampling selects the nearest observed timestamp to each target, breaks ties toward the earlier decoded frame, and retains that frame's original source index. Irregular or repeated timestamps do not alter identity. A stream with no usable timestamps cannot use temporal sampling and fails explicitly.

## Persistence and resume

The configurable processed root contains one directory per filename-stem video ID with `metadata.json`, `frames.json`, `embeddings.npy`, `manifest.json`, and deterministic source-index image names. Writes use temporary siblings and atomic replacement. Source SHA-256 and output-affecting configuration fingerprints determine whether completed work may resume. Changed sources/configuration and incomplete or failed states are rebuilt.

FAISS generations are rebuilt from compatible completed video artifacts in staging and published only after index/payload validation. Every indexed `frame_uid` resolves to its full `FrameRecord` through `PersistentCandidateResolver`.

## CLI

```bash
python -m backend.app.video.ingest \
  --input /path/to/videos \
  --output data/processed/videos \
  --sample-interval 1.0 \
  --device cpu \
  --batch-size 8 \
  --resume
```

Directory ingestion skips unsupported extensions, isolates corrupt-video failures, and continues unless `--fail-fast` is supplied. Duplicate filename-stem video IDs are rejected.

## Temporal neighborhoods

`get_frame_neighborhood` sequentially recovers nearby original decoded-frame indices. `get_temporal_neighborhood` filters observed timestamps. Dense neighbors are not embedded by default.

## M15.1 hardening

Failures preserve the highest validated checkpoint. On restart, metadata, frame records/images, and embeddings are reconciled with their stage fingerprints and actual persisted contents; a status label alone is never trusted. Metadata can be reused after metadata interruption, metadata plus frames after frame interruption, and all per-video artifacts after embedding interruption.

Invalidation follows the dependency chain. Sampling interval, submission policy, image format, or image quality rebuild frames, embeddings, and the index while reusing metadata. Encoder identity changes rebuild embeddings and the index while reusing frames. Changing `index_type` republishes only the global index while reusing metadata, frame records/images, and embeddings without decoding or encoding. Batch size and device are execution-only and do not invalidate artifacts.

M15.2 verifies that clean and interrupted/resumed ingests produce identical deterministic logical artifacts. It also injects failure at the exact atomic `CURRENT` replacement and proves the previous generation remains active, reloadable, searchable, and available alongside a later successful generation.

Directory ingestion isolates corrupt videos and publishes only compatible validated videos. The global index is rebuilt from every compatible completed manifest under the processed root, so later single-video ingestion does not drop older videos.

Index publication uses immutable `index/generations/<id>` directories. New artifacts are built under `index/.staging/<id>`, validated, renamed into the generations directory, and activated only by an atomic `index/CURRENT` replacement. Previous generations remain available after failed or successful publication. Cleanup is scoped to `.staging` and never removes active or rollback generations.

Atomic artifact writes fsync file contents and parent directories on the supported Linux environment. Full power-loss guarantees remain filesystem and hardware dependent; M15.1 is not a transactional database.

The deterministic multi-video smoke corpus uses locally generated MPEG-4 videos. Standard CI uses a deterministic image-derived encoder; an opt-in `RUN_M15_REAL_MODEL=1` test exercises the existing real CPU SigLIP2 encoder and verifies end-to-end mapping without claiming production semantic quality.

## Limitations

M15 performs coarse visual ingestion only. It assumes a single offline index publisher; concurrent publisher coordination is outside M15.1. It does not provide OCR, ASR, VQA, TRAKE alignment, captioning, dense-frame indexing, remote M14 reranking, or GPU scheduling. Sequential round-trip decoding is the correctness reference because random seek behavior depends on codecs and keyframes.
