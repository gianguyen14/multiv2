"""Lazy raw-video source resolution + HTTP range serving support.

Raw videos are not stored in the packed DB. This module resolves a ``video_id``
to a local file by, in order:

1. an already-cached file under ``VIDEO_CACHE_DIR`` (default ``data/videos``);
2. a local source directory ``VIDEO_SOURCE_DIR`` containing ``<video_id>.mp4``
   (served in place, never copied);
3. a remote source URL template ``VIDEO_SOURCE_URL_TEMPLATE`` containing the
   ``{video_id}`` placeholder, downloaded lazily into the cache on first request.

It never touches the packed DB, never ingests, and never re-encodes. Downloads
are atomic (temp file + rename) and serialized per video so concurrent requests
do not race. Only http(s) sources are allowed.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import urllib.parse
import urllib.request
from pathlib import Path

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_DOWNLOAD_TIMEOUT_SECONDS = 60
_MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024

_download_locks: dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()


def _lock_for(video_id: str) -> threading.Lock:
    with _download_locks_guard:
        lock = _download_locks.setdefault(video_id, threading.Lock())
        return lock


def cache_dir() -> Path:
    return Path(os.getenv("VIDEO_CACHE_DIR", "data/videos"))


def source_dir() -> Path | None:
    raw = os.getenv("VIDEO_SOURCE_DIR", "").strip()
    return Path(raw) if raw else None


def source_url_template() -> str | None:
    raw = os.getenv("VIDEO_SOURCE_URL_TEMPLATE", "").strip()
    return raw or None


def configured() -> bool:
    return source_dir() is not None or source_url_template() is not None


def video_url_for(video_id: str) -> str | None:
    """Return a playable endpoint only when a raw-video source is configured."""
    return f"/api/video/{video_id}" if configured() else None


def valid_video_id(video_id: str) -> bool:
    return bool(_VIDEO_ID_RE.fullmatch(video_id or ""))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(
            url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > _MAX_VIDEO_BYTES:
                raise OSError("remote video exceeds configured size limit")
            copied = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > _MAX_VIDEO_BYTES:
                    raise OSError("remote video exceeds configured size limit")
                out.write(chunk)
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def resolve_video_path(video_id: str) -> Path:
    """Return a local path for ``video_id``, downloading lazily if needed."""
    if not valid_video_id(video_id):
        raise ValueError(f"invalid video_id: {video_id!r}")

    cached = cache_dir() / f"{video_id}.mp4"
    if cached.is_file():
        return cached

    src = source_dir()
    if src is not None:
        source_root = src.resolve()
        candidate = (source_root / f"{video_id}.mp4").resolve()
        if candidate.is_relative_to(source_root) and candidate.is_file():
            return candidate

    template = source_url_template()
    if template:
        url = template.replace("{video_id}", video_id)
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError(
                "VIDEO_SOURCE_URL_TEMPLATE must be an http(s) URL template"
            )
        with _lock_for(video_id):
            if not cached.is_file():
                _download(url, cached)
        return cached

    raise FileNotFoundError(
        "raw video source is not configured; set VIDEO_SOURCE_DIR or "
        "VIDEO_SOURCE_URL_TEMPLATE"
    )
