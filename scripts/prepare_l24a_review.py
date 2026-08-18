#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".mpeg", ".mpg"}


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def ffprobe(path: Path) -> dict[str, Any]:
    proc = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,codec_name,pix_fmt:format=duration,size,bit_rate",
        "-of", "json", str(path),
    ])
    data = json.loads(proc.stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    duration = float(fmt.get("duration") or 0.0)
    return {
        "file": path.name,
        "relative_path": str(path),
        "duration_seconds": duration,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "codec": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "nb_frames": stream.get("nb_frames"),
        "size_bytes": int(fmt.get("size") or 0),
        "bit_rate": int(fmt.get("bit_rate") or 0),
    }


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("/", "_")


def extract_frame(video: Path, timestamp: float, out: Path, width: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(timestamp, 0.0):.3f}", "-i", str(video),
        "-frames:v", "1", "-vf", f"scale={width}:-2:flags=lanczos",
        "-q:v", "4", str(out),
    ], check=True)


def font(size: int = 22):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def make_sheet(frames: list[tuple[Path, float]], out: Path, *, cols: int, cell_w: int, cell_h: int, title: str) -> None:
    rows = math.ceil(len(frames) / cols)
    header_h = 52
    sheet = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h), "black")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, fill="white", font=font(24))
    label_font = font(19)
    for idx, (path, ts) in enumerate(frames):
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((cell_w, cell_h))
            x = (idx % cols) * cell_w
            y = header_h + (idx // cols) * cell_h
            bg = Image.new("RGB", (cell_w, cell_h), "black")
            bg.paste(im, ((cell_w - im.width) // 2, (cell_h - im.height) // 2))
            sheet.paste(bg, (x, y))
            draw.rectangle((x + 4, y + 4, x + 150, y + 32), fill="black")
            draw.text((x + 9, y + 6), f"{idx + 1:02d}  {ts:7.2f}s", fill="white", font=label_font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=86, optimize=True)


def sample_times(duration: float, n: int) -> list[float]:
    if duration <= 0:
        return [0.0] * n
    return [duration * ((i + 0.5) / n) for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    args = ap.parse_args()

    root = args.input_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    temp = out / "_frames"

    videos = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"No video files found under {root}")

    inventory: list[dict[str, Any]] = []
    for i, video in enumerate(videos, start=1):
        meta = ffprobe(video)
        meta["relative_path"] = str(video.relative_to(root))
        meta["ordinal"] = i
        inventory.append(meta)
        duration = float(meta["duration_seconds"])
        stem = f"{i:03d}_{safe_stem(video)}"
        print(f"[{i}/{len(videos)}] {meta['relative_path']} duration={duration:.2f}s", flush=True)

        overview_times = sample_times(duration, 16)
        overview_frames: list[tuple[Path, float]] = []
        for j, ts in enumerate(overview_times):
            frame_path = temp / stem / f"overview_{j:02d}.jpg"
            extract_frame(video, ts, frame_path, 480)
            overview_frames.append((frame_path, ts))
        make_sheet(
            overview_frames,
            out / "overview" / f"{stem}.jpg",
            cols=4,
            cell_w=480,
            cell_h=270,
            title=f"{i:03d}  {meta['relative_path']}  |  {duration:.2f}s",
        )

        detail_times = [duration * p for p in (0.20, 0.40, 0.60, 0.80)] if duration > 0 else [0.0] * 4
        detail_frames: list[tuple[Path, float]] = []
        for j, ts in enumerate(detail_times):
            frame_path = out / "detail_frames" / stem / f"p{(j + 1) * 20:02d}_{ts:.2f}s.jpg"
            extract_frame(video, ts, frame_path, 960)
            detail_frames.append((frame_path, ts))
        make_sheet(
            detail_frames,
            out / "detail" / f"{stem}.jpg",
            cols=2,
            cell_w=960,
            cell_h=540,
            title=f"DETAIL {i:03d}  {meta['relative_path']}  |  {duration:.2f}s",
        )

    with (out / "inventory.json").open("w", encoding="utf-8") as f:
        json.dump({"video_count": len(inventory), "videos": inventory}, f, ensure_ascii=False, indent=2)

    fields = [
        "ordinal", "file", "relative_path", "duration_seconds", "width", "height",
        "codec", "pix_fmt", "avg_frame_rate", "r_frame_rate", "nb_frames", "size_bytes", "bit_rate",
    ]
    with (out / "inventory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(inventory)

    with (out / "README.txt").open("w", encoding="utf-8") as f:
        f.write(
            "Human-first review artifact for Videos_L24_a.\n"
            "overview/: 16 uniformly sampled frames per video in a 4x4 sheet.\n"
            "detail/: 4 larger representative frames (20/40/60/80%).\n"
            "detail_frames/: the 4 larger frames individually.\n"
            "inventory.json/csv: ffprobe metadata.\n"
            "No model-generated interpretation is included in this stage.\n"
        )

    # Keep artifact compact; raw temporary overview frames are not needed after sheets are made.
    if temp.exists():
        import shutil
        shutil.rmtree(temp)

    print(f"Prepared {len(videos)} videos into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
