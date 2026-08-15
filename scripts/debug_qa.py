import os
import json
from pathlib import Path
from eval_mini_gt import ConfiguredSearch
from backend.app.retrieval.qa_query_decomposition import QAQueryDecomposer

gt_path = Path("data/validation/three_video_ground_truth/mini_gt.json")
with open(gt_path) as f:
    gt = json.load(f)
    
os.environ["VIDEO_PROCESSED_ROOT"] = "data/processed-validation/real-sample-20260814"
os.environ["SEARCH_ENABLE_OCR"] = "true"
os.environ["SEARCH_ENABLE_ASR"] = "true"
searcher = ConfiguredSearch()
searcher._initialize()
decomposer = QAQueryDecomposer()

for q in gt["qa"]:
    print(f"\nQ: {q['question']}")
    decomp = decomposer.decompose(q["question"])
    res = searcher.handle({"query_type": "qa", "query": q["question"], "top_k": 3})
    for i, r in enumerate(res):
        print(f"  Rank {i+1}: Video={r['video_id']}, Frame={r['source_frame_index_zero_based']}")
        print(f"    Answer: {r.get('answer')}")
        
        evidence = []
        for ocr in searcher._ocr:
            if ocr.video_id == r["video_id"] and ocr.source_frame_index_zero_based == r["source_frame_index_zero_based"]:
                evidence.append({"id": ocr.frame_uid, "text": ocr.raw_text})
        for asr in searcher._asr:
            if asr.video_id == r["video_id"] and asr.start_frame is not None and asr.start_frame <= r["source_frame_index_zero_based"] <= (asr.end_frame or asr.start_frame):
                evidence.append({"id": asr.segment_id, "text": asr.raw_text})
        print(f"    Evidence: {[e['text'] for e in evidence]}")
