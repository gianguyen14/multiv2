import os
import json
from pathlib import Path
from eval_mini_gt import ConfiguredSearch

def run_traces():
    gt_path = Path("data/validation/three_video_ground_truth/mini_gt.json")
    with open(gt_path) as f:
        gt = json.load(f)
        
    os.environ["VIDEO_PROCESSED_ROOT"] = "data/processed-validation/real-sample-20260814"
    
    # Run with all modalities to get scores
    os.environ["SEARCH_ENABLE_OCR"] = "true"
    os.environ["SEARCH_ENABLE_ASR"] = "true"
    searcher_all = ConfiguredSearch()
    searcher_all._initialize()
    
    # Just use the _bundle, _ocr, _asr from searcher_all to avoid reloading
    
    # For each KIS query
    for item in gt["kis"]:
        query = item["query"]
        accepted_intervals = [[item["video_id"], item["accepted_frame_interval"][0], item["accepted_frame_interval"][1]]]

        
        # We need to trace candidates. We can just call searcher_all.search and get the components
        res_all = searcher_all.search(query, top_k=20)
        
        # Re-sort for visual only
        res_vis = sorted(res_all, key=lambda x: -x["visual_score"])
        
        # Check ranks
        rank_vis = -1
        for i, cand in enumerate(res_vis):
            if any(cand["video_id"] == a[0] and a[1] <= cand["source_frame_index_zero_based"] <= a[2] for a in accepted_intervals):
                rank_vis = i + 1
                break
                
        rank_all = -1
        for i, cand in enumerate(res_all):
            if any(cand["video_id"] == a[0] and a[1] <= cand["source_frame_index_zero_based"] <= a[2] for a in accepted_intervals):
                rank_all = i + 1
                break
                
        if rank_all > rank_vis: # Regressed
            print("======================================================================")
            print(f"QUERY: {query}")
            print(f"REGRESSION: Rank dropped from {rank_vis} (Visual) to {rank_all} (Multimodal)")
            print("\nCORRECT CANDIDATE (Rank {}):".format(rank_all))
            correct = next(c for c in res_all if any(c["video_id"] == a[0] and a[1] <= c["source_frame_index_zero_based"] <= a[2] for a in accepted_intervals))
            print(f"  Video: {correct['video_id']}, Frame: {correct['source_frame_index_zero_based']}")
            print(f"  Visual: {correct['visual_score']:.4f}")
            print(f"  OCR: {correct['ocr_score']:.4f}")
            print(f"  ASR: {correct['asr_score']:.4f}")
            print(f"  Fused: {correct['score']:.4f}")
            
            # Print the text responsible
            ocr_texts = [o.normalized_text for o in searcher_all._ocr if o.video_id == correct['video_id'] and o.source_frame_index_zero_based == correct['source_frame_index_zero_based']]
            asr_texts = [a.normalized_text for a in searcher_all._asr if a.video_id == correct['video_id'] and a.start_frame is not None and a.start_frame <= correct['source_frame_index_zero_based'] <= (a.end_frame or a.start_frame)]
            print(f"  OCR Texts: {ocr_texts}")
            print(f"  ASR Texts: {asr_texts}")
            
            print("\nFALSE POSITIVE THAT OVERTOOK (Rank 1):")
            fp = res_all[0]
            print(f"  Video: {fp['video_id']}, Frame: {fp['source_frame_index_zero_based']}")
            print(f"  Visual: {fp['visual_score']:.4f}")
            print(f"  OCR: {fp['ocr_score']:.4f}")
            print(f"  ASR: {fp['asr_score']:.4f}")
            print(f"  Fused: {fp['score']:.4f}")
            
            fp_ocr_texts = [o.normalized_text for o in searcher_all._ocr if o.video_id == fp['video_id'] and o.source_frame_index_zero_based == fp['source_frame_index_zero_based']]
            fp_asr_texts = [a.normalized_text for a in searcher_all._asr if a.video_id == fp['video_id'] and a.start_frame is not None and a.start_frame <= fp['source_frame_index_zero_based'] <= (a.end_frame or a.start_frame)]
            print(f"  OCR Texts: {fp_ocr_texts}")
            print(f"  ASR Texts: {fp_asr_texts}")

if __name__ == "__main__":
    run_traces()
