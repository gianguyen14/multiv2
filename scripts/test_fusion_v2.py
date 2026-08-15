import os
import sys
from pathlib import Path

os.environ["VIDEO_PROCESSED_ROOT"] = "data/processed-validation/three-video-final"
sys.path.insert(0, str(Path.cwd()))

from backend.app.services.configured_search import ConfiguredSearch
from backend.app.retrieval.video_multimodal import lexical_score

def test_single_query(query, target_vid, interval):
    searcher = ConfiguredSearch()
    searcher._initialize()
    
    # 1. Visual search top 100
    vec = searcher._encoder.encode_text([query])[0]
    hits = searcher._bundle.index.search(vec, 100)
    
    candidates = {}
    for h in hits:
        payload = searcher._bundle.resolver.resolve(h["frame_id"])
        fid = payload["source_frame_index_zero_based"]
        candidates[(payload["video_id"], fid)] = {
            "video_id": payload["video_id"],
            "frame_id": payload["submission_frame_id"],
            "source_frame_index_zero_based": fid,
            "visual_score": float(h["score"]),
            "ocr_score": 0.0,
            "asr_score": 0.0
        }
        
    # 2. Add OCR matches
    for o in searcher._ocr:
        s = lexical_score(query, o.normalized_text)
        if s > 0.15:
            key = (o.video_id, o.source_frame_index_zero_based)
            if key in candidates:
                candidates[key]["ocr_score"] = max(candidates[key]["ocr_score"], s)
            else:
                # Find payload
                candidates[key] = {
                    "video_id": o.video_id,
                    "frame_id": o.source_frame_index_zero_based,
                    "source_frame_index_zero_based": o.source_frame_index_zero_based,
                    "visual_score": 0.0, # baseline
                    "ocr_score": s,
                    "asr_score": 0.0
                }
                
    # 3. Add ASR matches
    for a in searcher._asr:
        s = lexical_score(query, a.normalized_text)
        if s > 0.15 and a.start_frame is not None:
            # Add representative frames in segment
            key = (a.video_id, a.start_frame)
            if key in candidates:
                candidates[key]["asr_score"] = max(candidates[key]["asr_score"], s)
            else:
                candidates[key] = {
                    "video_id": a.video_id,
                    "frame_id": a.start_frame,
                    "source_frame_index_zero_based": a.start_frame,
                    "visual_score": 0.0,
                    "ocr_score": 0.0,
                    "asr_score": s
                }
                
    # Normalize visual
    cand_list = list(candidates.values())
    v_scores = [c["visual_score"] for c in cand_list]
    v_min, v_max = min(v_scores), max(v_scores)
    v_rng = max(v_max - v_min, 1e-9)
    
    for c in cand_list:
        v_norm = (c["visual_score"] - v_min) / v_rng if c["visual_score"] > 0 else 0.0
        c["fused"] = v_norm + 1.2 * c["ocr_score"] + 1.2 * c["asr_score"]
        
    ranked = sorted(cand_list, key=lambda x: -x["fused"])
    for i, r in enumerate(ranked[:10]):
        hit = (r["video_id"] == target_vid and interval[0] <= r["source_frame_index_zero_based"] <= interval[1])
        print(f"Rank {i+1}: vid={r['video_id']}, fid={r['source_frame_index_zero_based']}, fused={r['fused']:.3f}, vis={r['visual_score']:.3f}, ocr={r['ocr_score']:.2f}, asr={r['asr_score']:.2f}, HIT={hit}")

print("=== Testing OCR Query ===")
test_single_query("Bảng điện tử hiển thị Vietnam Airlines Nhóm 1 Nhóm 2 tại sân bay", "L22_V003", [7000, 7300])

print("\n=== Testing ASR Query ===")
test_single_query("nhiệt độ đạt 40 độ C", "L22_V001", [5400, 6600])

print("\n=== Testing Visual Query ===")
test_single_query("cảnh quay từ trên cao của một pháo đài hình ngôi sao được bao quanh bởi hào nước", "L22_V001", [6750, 7050])
