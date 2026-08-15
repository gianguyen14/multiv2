import os
import json
from pathlib import Path
from eval_mini_gt import ConfiguredSearch

gt_path = Path("data/validation/three_video_ground_truth/mini_gt.json")
with open(gt_path) as f:
    gt = json.load(f)
    
os.environ["VIDEO_PROCESSED_ROOT"] = "data/processed-validation/real-sample-20260814"
os.environ["SEARCH_ENABLE_OCR"] = "true"
os.environ["SEARCH_ENABLE_ASR"] = "true"
searcher = ConfiguredSearch()
searcher._initialize()

for item in gt["qa"]:
    q = item["question"]
    vid = item["video_id"]
    start, end = item["accepted_frame_interval"]
    
    ocr = []
    for o in searcher._ocr:
        if o.video_id == vid and start <= o.source_frame_index_zero_based <= end:
            ocr.append(o.raw_text)
            
    asr = []
    for a in searcher._asr:
        if a.video_id == vid and a.start_frame is not None and a.start_frame <= end and (a.end_frame or a.start_frame) >= start:
            asr.append(a.raw_text)
            
    print("====================")
    print("Q:", q)
    print("OCR:", ocr)
    print("ASR:", asr)
