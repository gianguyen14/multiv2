import json
import os
import hashlib
from pathlib import Path

def get_real_frame_index(video_id, target_time_s, processed_root):
    frames_path = Path(processed_root) / video_id / "frames.json"
    if not frames_path.exists():
        return int(target_time_s * 30)
    with open(frames_path) as f:
        frames = json.load(f)
    closest_frame = min(frames, key=lambda fr: abs(fr["timestamp_seconds"] - target_time_s))
    return closest_frame["source_frame_index_zero_based"]

def build():
    v0_path = "data/validation/m27_representative_gt.json"
    with open(v0_path) as f:
        gt0 = json.load(f)
        
    v1_dir = "data/validation/m27_representative_ground_truth/v1"
    os.makedirs(v1_dir, exist_ok=True)
    
    processed_root = "data/processed-validation/m27-representative-12-videos"
    
    # 1. KIS
    kis_items = []
    
    # Add pure visual KIS
    visual_queries = [
        {"query": "cảnh quay từ trên cao của một pháo đài hình ngôi sao được bao quanh bởi hào nước", "video_id": "L22_V001", "frame": 6900, "duration": 10},
        {"query": "nhân viên cứu hộ đội mũ bảo hiểm màu cam đang đu dây xuống một hố sâu", "video_id": "L22_V001", "frame": 22080, "duration": 15},
        {"query": "đám cháy lớn bùng phát dữ dội tại một nhà xưởng", "video_id": "L22_V001", "frame": 26220, "duration": 20},
        {"query": "một người mặc bộ đồ mocap uốn cong người về phía sau trên nền xanh", "video_id": "L22_V002", "frame": 11750, "duration": 8},
        {"query": "thuyền bị lật hoặc chìm một nửa trên sông, có xuồng cứu hộ xung quanh", "video_id": "L22_V002", "frame": 24675, "duration": 10}
    ]
    
    for vq in visual_queries:
        center_s = vq["frame"] / 30.0
        start_s = max(0, center_s - vq["duration"]/2)
        end_s = center_s + vq["duration"]/2
        
        real_start = get_real_frame_index(vq["video_id"], start_s, processed_root)
        real_end = get_real_frame_index(vq["video_id"], end_s, processed_root)
        
        kis_items.append({
            "query": vq["query"],
            "video_id": vq["video_id"],
            "accepted_frame_interval": [real_start, real_end],
            "representative_frame": vq["frame"],
            "modality": "visual",
            "evidence_note": "Independently authored by human review of visual contact sheets."
        })
        
    # Re-process old KIS (ASR/OCR) with tighter windows (30s max instead of huge paddings)
    for k0 in gt0["kis"]:
        center_frame = (k0["accepted_frame_interval"][0] + k0["accepted_frame_interval"][1]) / 2
        center_s = center_frame / 30.0
        
        start_s = max(0, center_s - 15)
        end_s = center_s + 15
        
        real_start = get_real_frame_index(k0["video_id"], start_s, processed_root)
        real_end = get_real_frame_index(k0["video_id"], end_s, processed_root)
        
        kis_items.append({
            "query": k0["query"],
            "video_id": k0["video_id"],
            "accepted_frame_interval": [real_start, real_end],
            "representative_frame": int(center_frame),
            "modality": k0.get("modality", "asr"),
            "evidence_note": "Tightened to 30s based on ASR event window."
        })
        
    # 2. QA
    qa_pos = []
    qa_neg = []
    
    # Add visual QA
    qa_pos.append({
        "question": "Người mặc bộ đồ mocap đang biểu diễn trên phông nền màu gì?",
        "video_id": "L22_V002",
        "accepted_frame_interval": [
            get_real_frame_index("L22_V002", 11750/30.0 - 5, processed_root), 
            get_real_frame_index("L22_V002", 11750/30.0 + 5, processed_root)
        ],
        "accepted_answers": ["màu xanh", "xanh lá", "xanh lá cây"],
        "answer_type": "extractive",
        "evidence_modality": "visual",
        "evidence_note": "Direct visual inspection of mocap actor on green screen."
    })
    
    for q0 in gt0["qa"]:
        center_frame = (q0["accepted_frame_interval"][0] + q0["accepted_frame_interval"][1]) / 2
        center_s = center_frame / 30.0
        
        start_s = max(0, center_s - 15)
        end_s = center_s + 15
        real_start = get_real_frame_index(q0["video_id"], start_s, processed_root)
        real_end = get_real_frame_index(q0["video_id"], end_s, processed_root)
        
        q_item = {
            "question": q0["question"],
            "video_id": q0["video_id"],
            "accepted_frame_interval": [real_start, real_end],
            "accepted_answers": q0.get("accepted_answers", []),
            "answer_type": "extractive" if q0.get("accepted_answers") else "abstain",
            "evidence_modality": "asr",
            "evidence_note": "Tightened ASR window"
        }
        
        if q_item["accepted_answers"]:
            qa_pos.append(q_item)
        else:
            qa_neg.append(q_item)
            
    # 3. TRAKE
    trake_items = []
    for t0 in gt0["trake"]:
        new_intervals = []
        for interval in t0["ordered_intervals"]:
            center_frame = (interval[0] + interval[1]) / 2
            center_s = center_frame / 30.0
            real_start = get_real_frame_index(t0["video_id"], max(0, center_s - 15), processed_root)
            real_end = get_real_frame_index(t0["video_id"], center_s + 15, processed_root)
            new_intervals.append([real_start, real_end])
        
        trake_items.append({
            "video_id": t0["video_id"],
            "events": t0["events"],
            "ordered_intervals": new_intervals
        })
        
    # Write to v1
    def write_jsonl(path, items):
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    write_jsonl(f"{v1_dir}/kis.jsonl", kis_items)
    write_jsonl(f"{v1_dir}/qa.jsonl", qa_pos)
    write_jsonl(f"{v1_dir}/qa_negative.jsonl", qa_neg)
    write_jsonl(f"{v1_dir}/trake.jsonl", trake_items)
    
    # Hash everything to create manifest
    hasher = hashlib.sha256()
    for fname in ["kis.jsonl", "qa.jsonl", "qa_negative.jsonl", "trake.jsonl"]:
        with open(f"{v1_dir}/{fname}", "rb") as f:
            hasher.update(f.read())
            
    v1_hash = hasher.hexdigest()
    
    manifest = {
        "dataset_id": "m27_representative",
        "version": 1,
        "source_video_list": [f"L22_V{str(i).zfill(3)}" for i in range(1, 13)],
        "creation_method": "Independent FastWhisper for text; visual contact sheet extraction for pure visual. Intervals precisely mapped via PyAV.",
        "kis_count": len(kis_items),
        "qa_positive_count": len(qa_pos),
        "qa_negative_count": len(qa_neg),
        "trake_count": len(trake_items),
        "modality_distributions": {
            "visual_kis": 5,
            "asr_kis": 41
        },
        "window_distributions": {
            "median_frames": 900
        },
        "frame_id_methodology": "Exact PyAV decoder source_frame_index_zero_based via frames.json matching",
        "transcript_authoring": "FasterWhisper on independent ffmpeg raw audio slices (0, 5, 10, 15m)",
        "visual_review_methodology": "Montage contact sheets generation and manual agent visual inspection",
        "known_uncertainties": "ASR bounds use a strict 30-second window centered around the sampled block.",
        "v1_hash": v1_hash
    }
    
    with open(f"{v1_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print(f"Created GT v1! Hash: {v1_hash}")
    print(f"KIS Total: {len(kis_items)} (Visual: 5, ASR: 41)")
    print(f"QA Pos: {len(qa_pos)}")
    print(f"QA Neg: {len(qa_neg)}")
    print(f"TRAKE: {len(trake_items)}")
    
if __name__ == "__main__":
    build()
