import os
os.environ["VIDEO_PROCESSED_ROOT"] = "data/processed-validation/m27-representative-12-videos"

import json
from backend.app.services.configured_search import ConfiguredSearch

def evaluate(visual_w, ocr_w, asr_w):
    os.environ["VISUAL_WEIGHT"] = str(visual_w)
    os.environ["OCR_WEIGHT"] = str(ocr_w)
    os.environ["ASR_WEIGHT"] = str(asr_w)
    
    with open("data/validation/m27_representative_gt.json") as f:
        gt = json.load(f)
        
    searcher = ConfiguredSearch()
    
    kis_hits_1 = 0
    kis_hits_5 = 0
    kis_total = len(gt["kis"])
    
    qa_correct = 0
    qa_false_ans = 0
    qa_abstain = 0
    qa_total_pos = len([q for q in gt["qa"] if q["accepted_answers"]])
    qa_total_neg = len([q for q in gt["qa"] if not q["accepted_answers"]])
    
    print(f"--- Modality Ablation: Visual={visual_w}, OCR={ocr_w}, ASR={asr_w} ---")
    
    for item in gt["kis"]:
        query = item["query"]
        target_video = item["video_id"]
        interval = item["accepted_frame_interval"]
        
        try:
            results = searcher.search(query, top_k=5)
            
            # Check rank 1
            if len(results) > 0:
                r1 = results[0]
                if r1["video_id"] == target_video and interval[0] <= r1["frame_id"] <= interval[1]:
                    kis_hits_1 += 1
            
            # Check rank 5
            for r in results:
                if r["video_id"] == target_video and interval[0] <= r["frame_id"] <= interval[1]:
                    kis_hits_5 += 1
                    break
        except Exception as e:
            # Maybe text indexing is not done yet
            if "OCR" in str(e) or "ASR" in str(e) or "Payload" in str(e):
                continue
            
    for item in gt["qa"]:
        question = item["question"]
        target_video = item["video_id"]
        accepted = item["accepted_answers"]
        
        try:
            res = searcher.handle({"query_type": "qa", "query": question, "top_k": 5})
            if res and len(res) > 0 and "answer" in res[0]:
                ans_text = res[0]["answer"]
            else:
                ans_text = None
            
            if len(accepted) > 0: # Positive case
                if ans_text and ans_text in accepted:
                    qa_correct += 1
            else: # Negative case
                if ans_text is None:
                    qa_abstain += 1
                else:
                    qa_false_ans += 1
        except Exception as e:
            pass

    print(f"KIS R@1: {kis_hits_1}/{kis_total} ({(kis_hits_1/kis_total)*100:.1f}%)")
    print(f"KIS R@5: {kis_hits_5}/{kis_total} ({(kis_hits_5/kis_total)*100:.1f}%)")
    if qa_total_pos > 0:
        print(f"QA Pos Acc: {qa_correct}/{qa_total_pos} ({(qa_correct/qa_total_pos)*100:.1f}%)")
    if qa_total_neg > 0:
        print(f"QA Neg Abstain: {qa_abstain}/{qa_total_neg} ({(qa_abstain/qa_total_neg)*100:.1f}%)")
        print(f"QA Neg False Ans: {qa_false_ans}/{qa_total_neg} ({(qa_false_ans/qa_total_neg)*100:.1f}%)")
    
evaluate(1.0, 0.0, 0.0)
