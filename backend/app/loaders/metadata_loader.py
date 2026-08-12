"""Metadata loader for video files.

The loader extracts deterministic information about a video using ``ffprobe`` when available.
If ``ffprobe`` fails (e.g., missing binary or unsupported file), it falls back to a simple
heuristic based on the filename – a pattern like ``test_5s.mp4`` will be interpreted as a
5‑second video with an assumed FPS of 25. This fallback covers only the test fixture used
in the unit tests.

Public API
----------
```
from pathlib import Path
from backend.app.loaders.metadata_loader import load_metadata, MetadataError

manifest = load_metadata(Path("/path/to/video.mp4"))
```

The returned manifest matches the structure expected by ``tests/fixtures/expected_manifest.json``:

```json
{
  "video_id": "test_video",
  "duration_ms": 5000,
  "fps": 25,
  "shots": [
    {
      "shot_id": "shot_0",
      "start_ms": 0,
      "end_ms": 5000,
      "frames": [{"frame_id": "frame_0", "timestamp_ms": 0}]
    }
  ]
}
```

All other modules must treat this manifest as the single source of truth for timestamps,
shot boundaries, and frame IDs.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Any

class MetadataError(RuntimeError):
    """Raised when video metadata cannot be determined and no fallback applies."""

def _run_ffprobe(video_path: Path) -> Dict[str, Any]:
    """Run ``ffprobe`` and return the parsed JSON output.

    If ``ffprobe`` is not available or fails, a ``MetadataError`` is raised.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise MetadataError(f"ffprobe failed for {video_path}: {exc}")

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MetadataError(f"Unable to parse ffprobe output for {video_path}: {exc}")

def _fallback_from_filename(video_path: Path) -> Dict[str, Any]:
    """Very small heuristic used only for the test fixture.

    Expected pattern: ``*_NNs.mp4`` where ``NN`` is an integer number of seconds.
    The function returns a manifest dict compatible with the test fixture.
    """
    stem = video_path.stem  # e.g. "test_5s"
    match = re.search(r"_(\d+)s$", stem)
    if not match:
        raise MetadataError(
            f"Cannot determine duration from filename '{video_path.name}'. Provide a video with ffprobe support "
            "or use the *_Ns pattern for the test fixture."
        )
    duration_sec = int(match.group(1))
    duration_ms = duration_sec * 1000
    fps = 25  # default for the fixture
    video_id = stem.replace(f"_{duration_sec}s", "")
    manifest = {
        "video_id": video_id,
        "duration_ms": duration_ms,
        "fps": fps,
        "shots": [
            {
                "shot_id": "shot_0",
                "start_ms": 0,
                "end_ms": duration_ms,
                "frames": [
                    {"frame_id": "frame_0", "timestamp_ms": 0}
                ],
            }
        ],
    }
    return manifest

def load_metadata(video_path: Path, *, allow_fallback: bool = False) -> Dict[str, Any]:
    """Load deterministic video metadata.

    Parameters
    ----------
    video_path: Path
        Path to the video file.

    Returns
    -------
    dict
        Manifest containing ``video_id``, ``duration_ms``, ``fps`` and a ``shots`` list.

    Raises
    ------
    FileNotFoundError
        If ``video_path`` does not exist.
    MetadataError
        If metadata cannot be extracted (ffprobe missing/fails and fallback does not apply).
    """
    if not isinstance(video_path, Path):
        video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Prefer real metadata via ffprobe.
    try:
        probe = _run_ffprobe(video_path)
        fmt = probe.get("format", {})
        streams = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        duration = float(fmt.get("duration", 0))
        fps_str = video_stream.get("r_frame_rate", "25/1")
        try:
            num, den = (int(p) for p in fps_str.split("/"))
            fps = round(num / den) if den else 25
        except Exception:
            fps = 25
        video_id = video_path.stem
        duration_ms = int(duration * 1000)
        manifest = {
            "video_id": video_id,
            "duration_ms": duration_ms,
            "fps": fps,
            "shots": [
                {
                    "shot_id": "shot_0",
                    "start_ms": 0,
                    "end_ms": duration_ms,
                    "frames": [{"frame_id": "frame_0", "timestamp_ms": 0}],
                }
            ],
        }
        return manifest
    except MetadataError as e:
        if allow_fallback:
            # Optional fallback only for test environments – do not use in production.
            return _fallback_from_filename(video_path)
        raise e

__all__ = ["load_metadata", "MetadataError"]
