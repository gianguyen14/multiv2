import json
from pathlib import Path
from backend.app.services.configured_search import ConfiguredSearch
from backend.app.retrieval.qa_query_decomposition import QAQueryDecomposer

def run_eval():
    gt_path = Path("data/validation/three_video_ground_truth/mini_gt.json")
    with open(gt_path) as f:
        gt = json.load(f)
        
    searcher = ConfiguredSearch()
    decomposer = QAQueryDecomposer()
    
    results = {
        "kis": [],
        "qa": [],
        "trake": []
    }
    
    for k in gt["kis"]:
        res = searcher.search(k["query"], top_k=20)
        found = False
        best_rank = -1
        for i, r in enumerate(res):
            if r["video_id"] == k["video_id"] and k["accepted_frame_interval"][0] <= r["source_frame_index_zero_based"] <= k["accepted_frame_interval"][1]:
                found = True
                best_rank = i + 1
                break
        results["kis"].append({
            "query": k["query"],
            "found": found,
            "rank": best_rank,
            "r1": best_rank == 1,
            "r5": 1 <= best_rank <= 5,
            "r20": 1 <= best_rank <= 20
        })

    for q in gt["qa"]:
        decomp = decomposer.decompose(q["question"])
        # Use handle directly for QA to test the full pipeline including answerer
        res = searcher.handle({"query_type": "qa", "query": q["question"], "top_k": 20})
        
        found = False
        best_rank = -1
        correct_answer = False
        
        if res and res[0].get("answer"):
            ans_lower = res[0]["answer"].lower()
            if any(acc.lower() in ans_lower or ans_lower in acc.lower() for acc in q["accepted_answers"]):
                correct_answer = True
                
        for i, r in enumerate(res):
            if r["video_id"] == q["video_id"] and q["accepted_frame_interval"][0] <= r["source_frame_index_zero_based"] <= q["accepted_frame_interval"][1]:
                found = True
                best_rank = i + 1
                break
                
        results["qa"].append({
            "question": q["question"],
            "decomposed": decomp["retrieval_query"],
            "found_localization": found,
            "rank": best_rank,
            "correct_answer": correct_answer,
            "answer": res[0].get("answer") if res else ""
        })
        
    # Negative QA tests for abstention
    negative_questions = [
        "Tổng thống Mỹ nói gì?",
        "Vụ tai nạn giao thông xảy ra ở đâu?",
        "Ai đang hát trên sân khấu?",
        "Giá vàng hôm nay là bao nhiêu?",
        "Người phụ nữ mặc áo màu gì?"
    ]
    
    results["qa_negative"] = []
    for nq in negative_questions:
        decomp = decomposer.decompose(nq)
        res = searcher.handle({"query_type": "qa", "query": nq, "top_k": 20})
        ans = res[0].get("answer", "") if res else ""
        results["qa_negative"].append({
            "question": nq,
            "abstained": ans == "",
            "answer": ans
        })
        
    for t in gt["trake"]:
        res = searcher.handle({"query_type": "trake", "events": t["events"], "top_k": 5})
        
        # Check if top 1 matches video
        found_video = res and len(res) > 0 and res[0].get("video_id") == t["video_id"]
        
        results["trake"].append({
            "events": t["events"],
            "found_video": found_video,
            "top_score": res[0].get("score") if res and len(res) > 0 else 0
        })
        
    kis_r1 = sum(1 for k in results["kis"] if k["r1"]) / len(results["kis"])
    kis_r5 = sum(1 for k in results["kis"] if k["r5"]) / len(results["kis"])
    qa_loc = sum(1 for q in results["qa"] if q["found_localization"]) / len(results["qa"])
    qa_ans = sum(1 for q in results["qa"] if q["correct_answer"]) / len(results["qa"])
    qa_abstention = sum(1 for q in results["qa_negative"] if q["abstained"]) / len(results["qa_negative"])
    qa_false_ans = 1.0 - qa_abstention
    trake_vid = sum(1 for t in results["trake"] if t["found_video"]) / len(results["trake"])
    
    baseline = {
        "metrics": {
            "kis_r1": kis_r1,
            "kis_r5": kis_r5,
            "qa_localization": qa_loc,
            "qa_final_answer": qa_ans,
            "qa_abstention_rate": qa_abstention,
            "qa_false_answer_rate": qa_false_ans,
            "trake_video_match": trake_vid
        },
        "details": results
    }
    
    out_path = Path("eval/baselines/real_three_video.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)
        
    print(json.dumps(baseline["metrics"], indent=2))

if __name__ == "__main__":
    run_eval()
