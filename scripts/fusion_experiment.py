import os
import json
from pathlib import Path
from eval_mini_gt import ConfiguredSearch
from backend.app.video.text_evidence import normalize_text

gt_path = Path("data/validation/three_video_ground_truth/mini_gt.json")
with open(gt_path) as f:
    gt = json.load(f)
    
os.environ["VIDEO_PROCESSED_ROOT"] = "data/processed-validation/real-sample-20260814"
os.environ["SEARCH_ENABLE_OCR"] = "true"
os.environ["SEARCH_ENABLE_ASR"] = "true"
searcher = ConfiguredSearch()
searcher._initialize()

def lexical_score_v2(query, text):
    stopwords = {"có", "là", "và", "của", "trong", "ở", "một", "những", "cho", "để", "với", "không", "đến", "các", "thì", "mà", "như"}
    query_terms = set(normalize_text(query).split()) - stopwords
    text_terms = set(normalize_text(text).split())
    if not query_terms:
        return 0.0
    overlap = query_terms & text_terms
    if not overlap:
        return 0.0
    return len(overlap) / len(query_terms)

def strategy_d_visual_first_rerank(query, candidates):
    ranked = sorted(candidates, key=lambda x: -x["visual_score"])
    top_n = 20
    top_cands = ranked[:top_n]
    bottom_cands = ranked[top_n:]
    
    is_ocr = any(w in query.lower() for w in ["chữ", "ghi", "biển báo", "text", "bệnh nhân"])
    is_asr = any(w in query.lower() for w in ["nói", "phát biểu", "kể", "speaker"])
    
    ocr_w = 0.5 if is_ocr else 0.1
    asr_w = 0.5 if is_asr else 0.1
    
    for c in top_cands:
        ocr_s = lexical_score_v2(query, c["ocr_text"])
        asr_s = lexical_score_v2(query, c["asr_text"])
        c["score"] = c["visual_score"] * (1.0 + ocr_s * ocr_w + asr_s * asr_w)
        
    reranked_top = sorted(top_cands, key=lambda x: -x["score"])
    return reranked_top + bottom_cands

for item in gt["kis"]:
    query = item["query"]
    accepted_intervals = [[item["video_id"], item["accepted_frame_interval"][0], item["accepted_frame_interval"][1]]]
    
    vector = searcher._encoder.encode_text([query])[0]
    results = []
    for hit in searcher._bundle.index.search(vector, 100):
        payload = searcher._bundle.resolver.resolve(hit["frame_id"])
        ocr_texts = [o.normalized_text for o in searcher._ocr if o.video_id == payload["video_id"] and o.source_frame_index_zero_based == payload["source_frame_index_zero_based"]]
        asr_texts = [a.normalized_text for a in searcher._asr if a.video_id == payload["video_id"] and a.start_frame is not None and a.start_frame <= payload["source_frame_index_zero_based"] <= (a.end_frame or a.start_frame)]
        
        row = {
            "video_id": payload["video_id"],
            "source_frame_index_zero_based": payload["source_frame_index_zero_based"],
            "visual_score": hit["score"],
            "ocr_text": " ".join(ocr_texts),
            "asr_text": " ".join(asr_texts),
        }
        results.append(row)
        
    vis_ranked = sorted(results, key=lambda x: -x["visual_score"])
    vis_rank = -1
    for i, c in enumerate(vis_ranked):
        if any(c["video_id"] == a[0] and a[1] <= c["source_frame_index_zero_based"] <= a[2] for a in accepted_intervals):
            vis_rank = i + 1
            break
            
    strat_ranked = strategy_d_visual_first_rerank(query, results)
    strat_rank = -1
    for i, c in enumerate(strat_ranked):
        if any(c["video_id"] == a[0] and a[1] <= c["source_frame_index_zero_based"] <= a[2] for a in accepted_intervals):
            strat_rank = i + 1
            break
            
    print(f"QUERY: {query} -> VIS: {vis_rank}, STRAT: {strat_rank}")
