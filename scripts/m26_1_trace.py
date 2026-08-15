"""
M26.1 — Pre-change reproduction trace.

Produces a full candidate trace for the ASR-regression query:
    "thiếu niên nghiện smartphone có nguy cơ trầm cảm"

and the five negative Q&A cases.

Run WITHOUT modifying any production code.
"""
import os
import json
from pathlib import Path

# Point at the frozen processed root
os.environ.setdefault("VIDEO_PROCESSED_ROOT", "data/processed-validation/real-sample-20260814")

from backend.app.services.configured_search import ConfiguredSearch
from backend.app.retrieval.video_multimodal import lexical_score

# ── helpers ────────────────────────────────────────────────────────────────────

def rank_of(results, video_id, interval):
    for i, r in enumerate(results):
        if r["video_id"] == video_id and interval[0] <= r["source_frame_index_zero_based"] <= interval[1]:
            return i + 1
    return -1


def init_searcher(ocr="true", asr="true"):
    os.environ["SEARCH_ENABLE_OCR"] = ocr
    os.environ["SEARCH_ENABLE_ASR"] = asr
    s = ConfiguredSearch()
    s._initialize()
    return s


# ──────────────────────────────────────────────────────────────────────────────
# PART 1 — ASR regression trace for "thiếu niên nghiện smartphone có nguy cơ trầm cảm"
# ──────────────────────────────────────────────────────────────────────────────

gt_path = Path("data/validation/three_video_ground_truth/mini_gt.json")
with open(gt_path) as f:
    gt = json.load(f)

TARGET_QUERY = "thiếu niên nghiện smartphone có nguy cơ trầm cảm"
target_item = next(k for k in gt["kis"] if k["query"] == TARGET_QUERY)
target_video = target_item["video_id"]
target_interval = target_item["accepted_frame_interval"]

print("=" * 70)
print("ASR REGRESSION TRACE")
print(f"Query: {TARGET_QUERY}")
print(f"Correct video: {target_video}, interval: {target_interval}")
print("=" * 70)

# Visual-only baseline
searcher_vis = init_searcher(ocr="false", asr="false")
res_vis = searcher_vis.search(TARGET_QUERY, top_k=10)
rank_vis = rank_of(res_vis, target_video, target_interval)
print(f"\nVISUAL-ONLY correct rank: {rank_vis}")

# Multimodal (ASR enabled)
searcher_asr = init_searcher(ocr="false", asr="true")
res_asr = searcher_asr.search(TARGET_QUERY, top_k=10)
rank_asr = rank_of(res_asr, target_video, target_interval)
print(f"VISUAL+ASR  correct rank: {rank_asr}")
print(f"REGRESSION: {'YES' if rank_asr > rank_vis else 'NO'}")

# ── Detailed per-candidate breakdown ──────────────────────────────────────────
print("\n--- Full candidate trace (visual+ASR, top 10) ---")
print(f"{'Rank':<5} {'video_id':<12} {'frame':>7} {'vis_raw':>9} {'asr_raw':>9} {'asr_norm':>9} "
      f"{'vis_norm':>9} {'intent':>8} {'asr_w':>6} {'fused':>9}")

# We need raw (un-normalized) scores.  Re-run search with internal scores.
os.environ["SEARCH_ENABLE_OCR"] = "false"
os.environ["SEARCH_ENABLE_ASR"] = "true"
s = ConfiguredSearch()
s._initialize()

raw_results = []
vector = s._encoder.encode_text([TARGET_QUERY])[0]
for hit in s._bundle.index.search(vector, 50):
    payload = s._bundle.resolver.resolve(hit["frame_id"])
    asr_texts = [item.normalized_text for item in s._asr
                 if item.video_id == payload["video_id"]
                 and item.start_frame is not None
                 and item.start_frame <= payload["source_frame_index_zero_based"] <= (item.end_frame or item.start_frame)]
    asr_joined = " ".join(asr_texts)
    asr_raw = lexical_score(TARGET_QUERY, asr_joined)
    raw_results.append({
        "video_id": payload["video_id"],
        "frame": payload["source_frame_index_zero_based"],
        "vis_raw": float(hit["score"]),
        "asr_raw": asr_raw,
        "asr_texts": asr_texts,
        "payload": payload,
    })

# Normalise visual scores across all candidates (mimic minmax but bounded by search top_k)
vis_scores = [r["vis_raw"] for r in raw_results]
asr_scores = [r["asr_raw"] for r in raw_results]
vis_min, vis_max = min(vis_scores), max(vis_scores)
asr_min, asr_max = min(asr_scores), max(asr_scores)

is_asr_query = any(w in TARGET_QUERY.lower() for w in ["nói", "phát biểu", "kể", "speaker"])
asr_w = 0.5 if is_asr_query else 0.05

for r in raw_results:
    vis_norm = (r["vis_raw"] - vis_min) / (vis_max - vis_min) if vis_max > vis_min else 0.0
    asr_norm = (r["asr_raw"] - asr_min) / (asr_max - asr_min) if asr_max > asr_min else 0.0
    r["vis_norm"] = vis_norm
    r["asr_norm"] = asr_norm
    # Current multiplicative formula
    r["fused"] = r["vis_raw"] * (1.0 + r["asr_raw"] * asr_w)

sorted_all = sorted(raw_results, key=lambda x: -x["fused"])

for rank, r in enumerate(sorted_all[:10], 1):
    is_correct = (r["video_id"] == target_video
                  and target_interval[0] <= r["frame"] <= target_interval[1])
    marker = " <-- CORRECT" if is_correct else ""
    print(f"{rank:<5} {r['video_id']:<12} {r['frame']:>7} {r['vis_raw']:>9.4f} {r['asr_raw']:>9.4f} "
          f"{r['asr_norm']:>9.4f} {r['vis_norm']:>9.4f} {'ASR' if is_asr_query else 'VISUAL':>8} "
          f"{asr_w:>6.2f} {r['fused']:>9.4f}{marker}")
    if r["asr_raw"] > 0:
        print(f"       ASR text(s): {r['asr_texts']}")

print("\n--- Why regression happens ---")
correct = next((r for r in sorted_all if r["video_id"] == target_video
                and target_interval[0] <= r["frame"] <= target_interval[1]), None)
false_pos = sorted_all[0]
if correct and false_pos is not correct:
    print(f"Correct  candidate: vis={correct['vis_raw']:.4f}, asr_raw={correct['asr_raw']:.4f}, fused={correct['fused']:.4f}")
    print(f"FalsePos candidate: vis={false_pos['vis_raw']:.4f}, asr_raw={false_pos['asr_raw']:.4f}, fused={false_pos['fused']:.4f}")
    print(f"False positive has HIGHER ASR lexical score despite being semantically irrelevant.")
    print(f"  False positive ASR: {false_pos['asr_texts']}")
    print(f"  Correct     ASR:    {correct['asr_texts']}")

# ──────────────────────────────────────────────────────────────────────────────
# PART 2 — Q&A negative cases root cause
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Q&A NEGATIVE CASES — ROOT CAUSE TRACE")
print("=" * 70)

from backend.app.retrieval.video_qa import ExtractiveAnswerer
from backend.app.retrieval.qa_query_decomposition import QAQueryDecomposer

negative_questions = [
    "Tổng thống Mỹ nói gì?",
    "Vụ tai nạn giao thông xảy ra ở đâu?",
    "Ai đang hát trên sân khấu?",
    "Giá vàng hôm nay là bao nhiêu?",
    "Người phụ nữ mặc áo màu gì?",
]

os.environ["SEARCH_ENABLE_OCR"] = "true"
os.environ["SEARCH_ENABLE_ASR"] = "true"
searcher_full = ConfiguredSearch()
searcher_full._initialize()
answerer = ExtractiveAnswerer()
decomposer = QAQueryDecomposer()

for nq in negative_questions:
    print(f"\nQuestion: {nq}")
    decomp = decomposer.decompose(nq)
    rows = searcher_full.search(decomp["retrieval_query"], top_k=10)

    pooled = []
    for row in rows[:10]:
        for ocr in searcher_full._ocr:
            if (ocr.video_id == row["video_id"]
                    and ocr.source_frame_index_zero_based == row["source_frame_index_zero_based"]):
                pooled.append({"id": ocr.frame_uid, "text": ocr.raw_text, "src": "ocr"})
        for asr in searcher_full._asr:
            if (asr.video_id == row["video_id"] and asr.start_frame is not None
                    and asr.start_frame <= row["source_frame_index_zero_based"] <= (asr.end_frame or asr.start_frame)):
                pooled.append({"id": asr.segment_id, "text": asr.raw_text, "src": "asr"})

    seen = set()
    unique = []
    for ev in pooled:
        if ev["text"] not in seen:
            seen.add(ev["text"])
            unique.append(ev)

    qa_res = answerer.answer(nq, unique)
    print(f"  Answer produced: '{qa_res['answer']}' (conf={qa_res['confidence']:.2f})")
    print(f"  Abstained: {qa_res['answer'] == ''}")
    if qa_res["answer"]:
        ev_id = qa_res["evidence_sources"][0] if qa_res["evidence_sources"] else "?"
        ev_text = next((e["text"] for e in unique if e["id"] == ev_id), "?")
        print(f"  Evidence used:   '{ev_text[:120]}'")
        print(f"  WHY WRONG: Regex pattern accepted irrelevant text as valid answer.")
