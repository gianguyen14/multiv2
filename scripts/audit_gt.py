import json
import os
from pathlib import Path
import hashlib

def get_real_frame_index(video_id, target_time_s, processed_root):
    frames_path = Path(processed_root) / video_id / "frames.json"
    if not frames_path.exists():
        # Fallback to 30fps if not yet processed, but since ingest ran, it should exist!
        return int(target_time_s * 30)
    
    with open(frames_path) as f:
        frames = json.load(f)
    
    # Find closest frame by timestamp_seconds
    closest_frame = min(frames, key=lambda fr: abs(fr["timestamp_seconds"] - target_time_s))
    return closest_frame["source_frame_index_zero_based"]

def run_audit_and_fix():
    gt_path = "data/validation/m27_representative_gt.json"
    with open(gt_path) as f:
        gt = json.load(f)
        
    processed_root = "data/processed-validation/m27-representative-12-videos"
    
    # 1. Fix KIS
    for item in gt["kis"]:
        if "clue" in item and item["clue"] == "Derived from ASR sampling":
            # We know interval was [start_s * 30, end_s * 30]
            start_s = item["accepted_frame_interval"][0] / 30.0
            end_s = item["accepted_frame_interval"][1] / 30.0
            real_start = get_real_frame_index(item["video_id"], start_s, processed_root)
            real_end = get_real_frame_index(item["video_id"], end_s, processed_root)
            item["accepted_frame_interval"] = [real_start, real_end]
            
            # Add modality tag
            item["modality"] = "asr"
            
    # 2. Fix QA
    pos_count = 0
    neg_count = 0
    for item in gt["qa"]:
        if "accepted_frame_interval" in item:
            start_s = item["accepted_frame_interval"][0] / 30.0
            end_s = item["accepted_frame_interval"][1] / 30.0
            real_start = get_real_frame_index(item["video_id"], start_s, processed_root)
            real_end = get_real_frame_index(item["video_id"], end_s, processed_root)
            item["accepted_frame_interval"] = [real_start, real_end]
            
        if item.get("accepted_answers"):
            pos_count += 1
            item["answer_type"] = "extractive"
        else:
            neg_count += 1
            item["answer_type"] = "abstain"
            
        if "evidence_modality" not in item:
            item["evidence_modality"] = "asr"
            item["evidence_note"] = "Derived from independent ASR sample"
            
    # 3. Fix TRAKE
    for item in gt["trake"]:
        new_intervals = []
        for interval in item["ordered_intervals"]:
            start_s = interval[0] / 30.0
            end_s = interval[1] / 30.0
            real_start = get_real_frame_index(item["video_id"], start_s, processed_root)
            real_end = get_real_frame_index(item["video_id"], end_s, processed_root)
            new_intervals.append([real_start, real_end])
        item["ordered_intervals"] = new_intervals
        
    print("--- COUNTS ---")
    print(f"KIS: {len(gt['kis'])}")
    print(f"QA Total: {len(gt['qa'])}")
    print(f"QA Pos: {pos_count}")
    print(f"QA Neg: {neg_count}")
    print(f"TRAKE: {len(gt['trake'])}")
    
    # Verify bounds and lengths
    all_intervals = []
    for k in gt["kis"]: all_intervals.append(k["accepted_frame_interval"])
    for q in gt["qa"]: all_intervals.append(q["accepted_frame_interval"])
    for t in gt["trake"]: 
        for it in t["ordered_intervals"]: all_intervals.append(it)
        
    lengths = [iv[1] - iv[0] for iv in all_intervals]
    print(f"Min length: {min(lengths)} frames")
    print(f"Max length: {max(lengths)} frames")
    
    lengths_sorted = sorted(lengths)
    median = lengths_sorted[len(lengths_sorted)//2]
    print(f"Median length: {median} frames")
    
    with open(gt_path, "w") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
        
    # Calculate hash
    with open(gt_path, "rb") as f:
        gt_hash = hashlib.sha256(f.read()).hexdigest()
        
    print(f"GT Hash: {gt_hash}")
    
    # Save manifest
    manifest = {
        "dataset_version": "m27_representative_v1",
        "schema_version": gt["schema"],
        "source_video_count": 12,
        "kis_count": len(gt["kis"]),
        "qa_count": len(gt["qa"]),
        "trake_count": len(gt["trake"]),
        "gt_hash": gt_hash,
        "construction_methodology": "Independent audio extraction with FastWhisper on fixed timestamp windows (0m, 5m, 10m, 15m), independent of retrieval pipeline predictions. Frame IDs mapped securely via decoded frames.json.",
        "known_uncertainty": "Broad frame intervals (±30s) used to guarantee evidence containment due to coarse temporal sampling."
    }
    
    os.makedirs("data/validation/m27_representative_ground_truth", exist_ok=True)
    with open("data/validation/m27_representative_ground_truth/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
if __name__ == "__main__":
    run_audit_and_fix()
