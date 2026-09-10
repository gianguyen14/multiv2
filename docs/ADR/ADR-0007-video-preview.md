# ADR-0007 — Seekable Video Preview via HTTP Range

**Status:** Accepted  
**Date:** 2026-09-10  
**Authors:** Engineering team

## Context

Search results expose `timestamp_seconds` per frame. Without a video player, users had to manually locate the relevant moment in a video from external storage. The system needed a way to deliver video content with seek capability.

## Decision

Add `GET /api/video/{video_id}` and `HEAD /api/video/{video_id}` routes with full HTTP Range support (single range, suffix range, 416 on unsatisfiable range). Video is served via `StreamingResponse`.

Source priority:
1. `VIDEO_SOURCE_DIR` — local filesystem directory (bind-mounted in Docker).
2. `VIDEO_SOURCE_URL_TEMPLATE` — remote URL template with `{video_id}` substitution; content is atomically cached in `VIDEO_CACHE_DIR`.
3. If neither is configured, `capabilities.raw_video_preview = false`.

Security: `video_id` is validated against an allowlist derived from the DB payload index; path traversal is rejected.

The frontend opens a `<video>` element, seeks to `timestamp_seconds` after `loadedmetadata`, and uses `EventListener` only (no `onclick`, no `innerHTML`).

## Consequences

- `capabilities.raw_video_preview` in `/health/live` accurately reflects source availability.
- Docker Compose `docker-compose.yml` documents `VIDEO_SOURCE_DIR_HOST` / `VIDEO_SOURCE_DIR_CONTAINER` for host-mounted video directories.
- Videos are served read-only; no upload/write path exists.
