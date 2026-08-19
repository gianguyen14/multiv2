#!/usr/bin/env python3
"""Human-vs-current-component probe for L24a counting questions.

Scope is intentionally narrow and explicit:
- human GT is anchored to one p40 representative frame per video;
- CurrentSystem visual retrieval is probed with the repository's SigLIP2Encoder;
- CurrentSystem HOW_MANY answering is probed with the repository's ExtractiveAnswerer;
- OCR evidence comes from Tesseract on the retrieved representative frame;
- this is NOT a full-video ConfiguredSearch/ingestion quality benchmark.

The purpose is to determine whether a visual counting gap is retrieval-related,
answerer-related, or both before changing production code.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from backend.app.embeddings.siglip2 import SigLIP2Encoder
from backend.app.retrieval.video_qa import ExtractiveAnswerer
from eval.l24a_human_annotations import records as human_records


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def _p40_map(review_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for folder in sorted((review_root / "detail_frames").glob("*")):
        if not folder.is_dir():
            continue
        parts = folder.name.split("_")
        if len(parts) < 3:
            continue
        video_id = "_".join(parts[1:3])
        matches = sorted(folder.glob("p40_*.jpg"))
        if matches:
            out[video_id] = matches[0]
    return out


def _ocr(path: Path) -> str:
    cmd = ["tesseract", str(path), "stdout", "-l", "eng+vie", "--psm", "6"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _numeric(answer: str) -> str | None:
    m = re.search(r"\b\d+(?:[.,]\d+)?\b", answer or "")
    return m.group(0).replace(",", ".") if m else None


def _safe_div(a: int, b: int) -> float:
    return float(a / b) if b else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    gt = list(human_records())
    frames = _p40_map(args.review_root)
    missing = [r["video_id"] for r in gt if r["video_id"] not in frames]
    if missing:
        raise RuntimeError(f"missing p40 representative frames: {missing}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _jsonl(out / "human_gt.jsonl", gt)

    video_ids = [r["video_id"] for r in gt]
    image_paths = [frames[v] for v in video_ids]
    questions = [r["question_vi"] for r in gt]

    encoder = SigLIP2Encoder(device=args.device, local_files_only=True)
    image_vecs = encoder.encode_image(image_paths, batch_size=4, normalize=True)
    text_vecs = encoder.encode_text(questions, batch_size=8, normalize=True)
    sims = text_vecs @ image_vecs.T

    answerer = ExtractiveAnswerer()
    ocr_cache: dict[str, str] = {}

    def ocr_for(video_id: str) -> str:
        if video_id not in ocr_cache:
            ocr_cache[video_id] = _ocr(frames[video_id])
        return ocr_cache[video_id]

    results: list[dict] = []
    for qi, row in enumerate(gt):
        order = np.argsort(-sims[qi], kind="stable")
        ranked_videos = [video_ids[int(i)] for i in order]
        top1 = ranked_videos[0]
        top5 = ranked_videos[:5]
        gt_video = row["video_id"]

        top1_text = ocr_for(top1)
        oracle_text = ocr_for(gt_video)
        top1_answer = answerer.answer(row["question_vi"], [{"id": f"ocr:{top1}", "text": top1_text}])
        oracle_answer = answerer.answer(row["question_vi"], [{"id": f"ocr:{gt_video}", "text": oracle_text}])

        expected = str(row["answer_count"])
        top1_num = _numeric(top1_answer.get("answer", ""))
        oracle_num = _numeric(oracle_answer.get("answer", ""))
        retrieval_top1 = top1 == gt_video
        retrieval_top5 = gt_video in top5
        top1_correct = top1_num == expected
        oracle_correct = oracle_num == expected

        failures = []
        if not retrieval_top1:
            failures.append("VISUAL_RETRIEVAL_TOP1_MISS")
        if not top1_answer.get("answer"):
            failures.append("COUNT_ANSWER_ABSTAIN")
        elif not top1_correct:
            failures.append("COUNT_WRONG_NUMERIC_EXTRACTION")
        if not oracle_answer.get("answer"):
            failures.append("ORACLE_FRAME_COUNT_ANSWER_ABSTAIN")
        elif not oracle_correct:
            failures.append("ORACLE_FRAME_COUNT_WRONG_NUMERIC_EXTRACTION")

        results.append({
            **row,
            "probe_scope": "p40_representative_frame_component_probe",
            "retrieved_video_top1": top1,
            "retrieved_video_top5": top5,
            "retrieval_video_top1_hit": retrieval_top1,
            "retrieval_video_top5_hit": retrieval_top5,
            "top1_similarity": float(sims[qi, order[0]]),
            "gt_similarity": float(sims[qi, video_ids.index(gt_video)]),
            "top1_ocr_text": top1_text,
            "oracle_p40_ocr_text": oracle_text,
            "system_answer_top1": top1_answer.get("answer", ""),
            "system_answer_top1_confidence": float(top1_answer.get("confidence", 0.0)),
            "system_answer_top1_numeric": top1_num,
            "system_answer_top1_correct": top1_correct,
            "system_answer_oracle_p40": oracle_answer.get("answer", ""),
            "system_answer_oracle_p40_confidence": float(oracle_answer.get("confidence", 0.0)),
            "system_answer_oracle_p40_numeric": oracle_num,
            "system_answer_oracle_p40_correct": oracle_correct,
            "failure_categories": failures,
        })

    _jsonl(out / "system_probe_results.jsonl", results)

    n = len(results)
    top1_hits = sum(r["retrieval_video_top1_hit"] for r in results)
    top5_hits = sum(r["retrieval_video_top5_hit"] for r in results)
    ans_top1 = sum(r["system_answer_top1_correct"] for r in results)
    ans_oracle = sum(r["system_answer_oracle_p40_correct"] for r in results)
    abst_top1 = sum(not r["system_answer_top1"] for r in results)
    abst_oracle = sum(not r["system_answer_oracle_p40"] for r in results)
    failures = Counter(x for r in results for x in r["failure_categories"])

    by_target: dict[str, dict] = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0, "answer": 0, "oracle": 0})
    for r in results:
        d = by_target[r["count_target"]]
        d["n"] += 1
        d["top1"] += int(r["retrieval_video_top1_hit"])
        d["top5"] += int(r["retrieval_video_top5_hit"])
        d["answer"] += int(r["system_answer_top1_correct"])
        d["oracle"] += int(r["system_answer_oracle_p40_correct"])

    summary = {
        "probe_scope": "CURRENT_COMPONENT_PROBE_NOT_FULL_CONFIGUREDSEARCH",
        "questions": n,
        "human_gt_high_confidence": sum(r["annotation_confidence"] == "high" for r in results),
        "human_gt_medium_confidence": sum(r["annotation_confidence"] == "medium" for r in results),
        "retrieval_video_top1_hits": top1_hits,
        "retrieval_video_top1_rate": _safe_div(top1_hits, n),
        "retrieval_video_top5_hits": top5_hits,
        "retrieval_video_top5_rate": _safe_div(top5_hits, n),
        "count_answer_top1_correct": ans_top1,
        "count_answer_top1_accuracy": _safe_div(ans_top1, n),
        "count_answer_oracle_p40_correct": ans_oracle,
        "count_answer_oracle_p40_accuracy": _safe_div(ans_oracle, n),
        "count_answer_top1_abstentions": abst_top1,
        "count_answer_oracle_p40_abstentions": abst_oracle,
        "failure_counts": dict(failures),
        "by_count_target": dict(by_target),
        "architecture_interpretation": {
            "how_many_answerer": "extracts numeric text from OCR/ASR evidence; it does not visually count objects in pixels",
            "meaning_of_oracle_p40_metric": "isolates answerer/OCR limitations from representative-frame retrieval",
            "meaning_of_top1_metric": "includes both representative-frame retrieval and answer extraction",
        },
    }
    (out / "system_probe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Videos_L24_a — Human vs CurrentSystem counting probe",
        "",
        "## Scope",
        "",
        "This is a **representative-frame component probe**, not a full CurrentSystem corpus benchmark. Human labels use the p40 review frame. The system side uses the repository SigLIP2 encoder to rank the same 43 p40 frames, then runs Tesseract OCR plus the repository ExtractiveAnswerer. No visual counting model is added.",
        "",
        "## Aggregate results",
        "",
        f"- Questions: **{n}**",
        f"- Human high-confidence labels: **{summary['human_gt_high_confidence']}**; medium-confidence: **{summary['human_gt_medium_confidence']}**",
        f"- Representative-frame video retrieval top-1: **{top1_hits}/{n} ({summary['retrieval_video_top1_rate']:.1%})**",
        f"- Representative-frame video retrieval top-5: **{top5_hits}/{n} ({summary['retrieval_video_top5_rate']:.1%})**",
        f"- End answer exact numeric accuracy from retrieved top-1 frame: **{ans_top1}/{n} ({summary['count_answer_top1_accuracy']:.1%})**",
        f"- Answer accuracy with the human p40 frame forced (oracle frame): **{ans_oracle}/{n} ({summary['count_answer_oracle_p40_accuracy']:.1%})**",
        f"- Top-1 answer abstentions: **{abst_top1}/{n}**",
        f"- Oracle-frame answer abstentions: **{abst_oracle}/{n}**",
        "",
        "## Key architectural finding",
        "",
        "`ExtractiveAnswerer` classifies `bao nhiêu`/`mấy` as `HOW_MANY`, but its answer path extracts numbers from textual OCR/ASR evidence. It does not count visible lions, people, flags, mats, poles, or other objects from image pixels. Therefore oracle-frame counting accuracy is the key isolation metric: if it remains low even when the correct representative frame is supplied, the main gap is answer capability rather than retrieval alone.",
        "",
        "## Per-question comparison",
        "",
        "| ID | GT video | Target | Human | Retrieved top-1 | Top-1 hit | System | Oracle-frame system | Failures |",
        "|---|---|---|---:|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['video_id']} | {r['count_target']} | {r['answer_count']} | {r['retrieved_video_top1']} | "
            f"{'Y' if r['retrieval_video_top1_hit'] else 'N'} | {r['system_answer_top1'] or 'ABSTAIN'} | "
            f"{r['system_answer_oracle_p40'] or 'ABSTAIN'} | {', '.join(r['failure_categories']) or 'PASS'} |"
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Do not treat this as a full-video retrieval score; only one representative p40 frame per video is indexed.",
        "- Do not treat OCR-extracted numbers as visual object counting.",
        "- Medium-confidence human labels must be rechecked against original video before immutable GT promotion.",
        "- Authoritative frame IDs remain unresolved here and are not reconstructed from timestamps.",
    ]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
