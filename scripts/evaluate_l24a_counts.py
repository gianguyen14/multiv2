#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.app.embeddings.siglip2 import SigLIP2Encoder
from backend.app.retrieval.qa_query_decomposition import QAQueryDecomposer
from backend.app.retrieval.video_qa import ExtractiveAnswerer
from backend.app.video.text_backends import TesseractOCRBackend


VI_NUMBERS = {
    "không": 0, "một": 1, "mot": 1, "hai": 2, "ba": 3, "bốn": 4, "bon": 4,
    "năm": 5, "nam": 5, "sáu": 6, "sau": 6, "bảy": 7, "bay": 7,
    "tám": 8, "tam": 8, "chín": 9, "chin": 9, "mười": 10, "muoi": 10,
}
EN_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def load_gt(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_frame(path: Path) -> tuple[str, float]:
    # Parent names are e.g. 001_L24_V002; timestamps are the final _123.45s token.
    match_video = re.search(r"(L24_V\d+)$", path.parent.name)
    if not match_video:
        raise ValueError(f"cannot parse video id from {path}")
    match_ts = re.search(r"_([0-9]+(?:\.[0-9]+)?)s\.jpg$", path.name)
    if not match_ts:
        raise ValueError(f"cannot parse timestamp from {path}")
    return match_video.group(1), float(match_ts.group(1))


def discover_frames(review_root: Path) -> list[dict]:
    paths = sorted((review_root / "overview_frames").glob("*/*.jpg"))
    paths += sorted((review_root / "detail_frames").glob("*/*.jpg"))
    rows = []
    seen = set()
    for path in paths:
        video_id, timestamp = parse_frame(path)
        key = (video_id, round(timestamp, 3), str(path.resolve()))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"path": path, "video_id": video_id, "timestamp_seconds": timestamp})
    if not rows:
        raise RuntimeError(f"no individual review frames found under {review_root}")
    return rows


def numeric_answer(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", str(text))
    if match:
        return int(match.group(1))
    lowered = re.sub(r"[^\wÀ-ỹ]+", " ", str(text).lower(), flags=re.UNICODE)
    for token in lowered.split():
        if token in VI_NUMBERS:
            return VI_NUMBERS[token]
        if token in EN_NUMBERS:
            return EN_NUMBERS[token]
    return None


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def aggregate(rows: list[dict]) -> dict:
    total = len(rows)
    output = {
        "queries": total,
        "video_recall_at_1": pct(sum(r["video_rank"] <= 1 for r in rows), total),
        "video_recall_at_5": pct(sum(r["video_rank"] <= 5 for r in rows), total),
        "video_recall_at_20": pct(sum(r["video_rank"] <= 20 for r in rows), total),
        "oracle_ocr_answer_coverage": pct(sum(bool(r["oracle_qa_answer"]) for r in rows), total),
        "oracle_ocr_numeric_exact": pct(sum(r["oracle_numeric_exact"] for r in rows), total),
        "top1_ocr_numeric_exact": pct(sum(r["top1_numeric_exact"] for r in rows), total),
    }
    frame_ranks = [r["target_evidence_frame_rank"] for r in rows]
    time_errors = [r["best_target_video_time_error_seconds"] for r in rows]
    if frame_ranks:
        output["target_evidence_frame_rank_median"] = float(statistics.median(frame_ranks))
    if time_errors:
        output["best_target_video_time_error_median_seconds"] = float(statistics.median(time_errors))
    return output


def markdown_report(result: dict) -> str:
    s = result["summary"]
    lines = [
        "# Videos_L24_a — human vs CurrentSystem sampled-frame count diagnostic",
        "",
        "## Scope and interpretation",
        "",
        "Human answers were frozen before this run. The system-side visual retrieval uses the repository's active `SigLIP2Encoder` on the **20 review samples per video** (16 uniform overview frames plus 4 detail frames). The human evidence frame is intentionally present in this pool. Therefore this is a controlled component diagnostic, not a substitute for a full 1-second production-video ingest benchmark.",
        "",
        "For counting, the QA diagnostic gives the active `ExtractiveAnswerer` OCR from the correct human evidence frame (oracle retrieval) and separately OCR from the visual top-1 frame. This is deliberately favorable to the answerer: failure on oracle evidence demonstrates an answer-generation limitation rather than a retrieval miss.",
        "",
        "No authoritative frame ID is reconstructed from timestamp × FPS. Temporal comparison uses timestamps only.",
        "",
        "## Aggregate",
        "",
        f"- Queries: **{s['queries']}**",
        f"- Video Recall@1: **{s['video_recall_at_1']:.2f}%**",
        f"- Video Recall@5: **{s['video_recall_at_5']:.2f}%**",
        f"- Video Recall@20: **{s['video_recall_at_20']:.2f}%**",
        f"- Median rank of the exact human evidence sample: **{s['target_evidence_frame_rank_median']:.1f}**",
        f"- Median time error of best retrieved sample from the correct video: **{s['best_target_video_time_error_median_seconds']:.2f}s**",
        f"- Oracle-frame OCR QA answer coverage: **{s['oracle_ocr_answer_coverage']:.2f}%**",
        f"- Oracle-frame numeric exact match: **{s['oracle_ocr_numeric_exact']:.2f}%**",
        f"- Top-1-frame numeric exact match: **{s['top1_ocr_numeric_exact']:.2f}%**",
        "",
        "## Per query",
        "",
        "| ID | Video | Human | Video rank | Evidence-frame rank | Correct-video best Δt | Top-1 video | Oracle QA answer | Oracle numeric |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for r in result["rows"]:
        ans = str(r.get("oracle_qa_answer") or "ABSTAIN").replace("|", "\\|")
        lines.append(
            f"| {r['id']} | {r['video_id']} | {r['human_answer']} | {r['video_rank']} | "
            f"{r['target_evidence_frame_rank']} | {r['best_target_video_time_error_seconds']:.2f}s | "
            f"{r['top1_video_id']} | {ans[:80]} | {'PASS' if r['oracle_numeric_exact'] else 'FAIL'} |"
        )
    lines += ["", "## By target type", ""]
    for name, metrics in sorted(result["by_type"].items()):
        lines.append(
            f"- `{name}`: n={metrics['queries']}, R@1={metrics['video_recall_at_1']:.1f}%, "
            f"R@5={metrics['video_recall_at_5']:.1f}%, oracle numeric={metrics['oracle_ocr_numeric_exact']:.1f}%"
        )
    lines += [
        "",
        "## Architectural reading",
        "",
        "The current QA answerer is precision-first and extractive: `HOW_MANY` answers must be supported by textual OCR/ASR evidence. It does not inspect image pixels and count arbitrary visual objects. Consequently, visual retrieval quality and count-answer quality must be read as separate axes.",
        "",
        "The next benchmark tier should use the full production video-ingestion path and the 1-second sampling/index artifacts. This sampled-frame diagnostic is the fast first comparison that identifies whether the main failure is scene retrieval, visual counting, or both.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-root", type=Path, required=True)
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    gt = load_gt(args.ground_truth)
    frames = discover_frames(args.review_root)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    frame_paths = [r["path"] for r in frames]
    decomposer = QAQueryDecomposer()
    retrieval_queries = [decomposer.decompose(r["q"])["retrieval_query"] for r in gt]

    encoder = SigLIP2Encoder(device=args.device, local_files_only=True)
    started = time.perf_counter()
    image_emb = encoder.encode_image(frame_paths, batch_size=args.batch_size)
    text_emb = encoder.encode_text(retrieval_queries, batch_size=args.batch_size)
    similarity = encoder.similarity(text_emb, image_emb)
    embedding_seconds = time.perf_counter() - started

    ocr = TesseractOCRBackend(languages="eng+vie")
    answerer = ExtractiveAnswerer()
    ocr_cache: dict[str, str] = {}

    def ocr_text(path: Path) -> str:
        key = str(path)
        if key not in ocr_cache:
            ocr_cache[key] = ocr.extract([path])[0].get("text", "")
        return ocr_cache[key]

    results = []
    for qi, record in enumerate(gt):
        scores = similarity[qi]
        order = np.argsort(-scores, kind="stable")
        target_video = record["video_id"]
        human_t = float(record["t"])

        video_ranks: dict[str, int] = {}
        for idx in order:
            vid = frames[int(idx)]["video_id"]
            if vid not in video_ranks:
                video_ranks[vid] = len(video_ranks) + 1
        video_rank = video_ranks.get(target_video, len(video_ranks) + 1)

        target_indices = [i for i, f in enumerate(frames) if f["video_id"] == target_video]
        target_evidence_idx = min(target_indices, key=lambda i: abs(frames[i]["timestamp_seconds"] - human_t))
        frame_positions = {int(idx): pos + 1 for pos, idx in enumerate(order)}
        target_frame_rank = frame_positions[target_evidence_idx]
        best_target_idx = min(target_indices, key=lambda i: frame_positions[i])

        top1_idx = int(order[0])
        oracle_path = frames[target_evidence_idx]["path"]
        top1_path = frames[top1_idx]["path"]

        oracle_text = ocr_text(oracle_path)
        top1_text = ocr_text(top1_path)
        oracle_answer = answerer.answer(record["q"], [{"id": str(oracle_path), "text": oracle_text}])
        top1_answer = answerer.answer(record["q"], [{"id": str(top1_path), "text": top1_text}])
        oracle_value = numeric_answer(oracle_answer.get("answer"))
        top1_value = numeric_answer(top1_answer.get("answer"))

        results.append({
            "id": record["id"],
            "video_id": target_video,
            "target_type": record["type"],
            "human_confidence": record.get("confidence"),
            "question": record["q"],
            "retrieval_query": retrieval_queries[qi],
            "human_answer": int(record["a"]),
            "human_timestamp_seconds": human_t,
            "video_rank": int(video_rank),
            "target_evidence_frame_rank": int(target_frame_rank),
            "target_evidence_sample_timestamp_seconds": frames[target_evidence_idx]["timestamp_seconds"],
            "best_target_video_sample_timestamp_seconds": frames[best_target_idx]["timestamp_seconds"],
            "best_target_video_time_error_seconds": abs(frames[best_target_idx]["timestamp_seconds"] - human_t),
            "top1_video_id": frames[top1_idx]["video_id"],
            "top1_timestamp_seconds": frames[top1_idx]["timestamp_seconds"],
            "top1_score": float(scores[top1_idx]),
            "target_evidence_score": float(scores[target_evidence_idx]),
            "oracle_ocr_text": oracle_text[:1000],
            "oracle_qa_answer": oracle_answer.get("answer", ""),
            "oracle_qa_confidence": oracle_answer.get("confidence"),
            "oracle_numeric_value": oracle_value,
            "oracle_numeric_exact": oracle_value == int(record["a"]),
            "top1_ocr_text": top1_text[:1000],
            "top1_qa_answer": top1_answer.get("answer", ""),
            "top1_qa_confidence": top1_answer.get("confidence"),
            "top1_numeric_value": top1_value,
            "top1_numeric_exact": top1_value == int(record["a"]),
        })

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        groups[row["target_type"]].append(row)

    payload = {
        "benchmark": "l24a-human-vs-current-sampled-frame-count-diagnostic-v1",
        "scope": {
            "videos": len({f["video_id"] for f in frames}),
            "sampled_frames": len(frames),
            "samples_per_video_expected": 20,
            "questions": len(gt),
            "retrieval": "active SigLIP2Encoder + QAQueryDecomposer over retained human-review samples",
            "answering": "active ExtractiveAnswerer over Tesseract OCR; oracle human evidence frame and visual top-1 frame",
            "not_full_production_ingest": True,
            "authoritative_frame_ids_reconstructed_from_fps": False,
        },
        "model": encoder.get_model_info(),
        "embedding_seconds": embedding_seconds,
        "summary": aggregate(results),
        "by_type": {name: aggregate(rows) for name, rows in groups.items()},
        "rows": results,
    }
    (out / "system_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "system_comparison.md").write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
