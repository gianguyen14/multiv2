import pytest

def evaluate_kis_result(prediction, accepted_intervals):
    # prediction: {"video_id": str, "frame_id": int}
    # accepted_intervals: list of (video_id, start_frame, end_frame)
    if not prediction:
        return False
    vid = prediction.get("video_id")
    fid = prediction.get("frame_id")
    for (acc_vid, start_f, end_f) in accepted_intervals:
        if vid == acc_vid and start_f <= fid <= end_f:
            return True
    return False

def evaluate_qa_result(prediction, accepted_video, accepted_interval, accepted_answers):
    # prediction: {"video_id": str, "frame_id": int, "answer": str}
    if not prediction:
        return {"localization": False, "answer": False, "abstained": True}
    
    ans = prediction.get("answer", "")
    vid = prediction.get("video_id")
    fid = prediction.get("frame_id")
    
    loc_hit = (vid == accepted_video and accepted_interval[0] <= fid <= accepted_interval[1]) if (accepted_video and accepted_interval) else False
    
    if not accepted_answers: # Negative question
        abstained = (ans == "" or ans is None)
        return {"localization": loc_hit, "answer": abstained, "abstained": abstained}
    
    if not ans:
        return {"localization": loc_hit, "answer": False, "abstained": True}
        
    ans_lower = ans.lower().strip()
    ans_hit = any(acc.lower() in ans_lower or ans_lower in acc.lower() for acc in accepted_answers)
    return {"localization": loc_hit, "answer": ans_hit, "abstained": False}

def evaluate_trake_result(prediction, target_video, ordered_intervals):
    # prediction: list of {"video_id": str, "frame_id": int}
    if not prediction or len(prediction) != len(ordered_intervals):
        return {"video_match": False, "hits": 0, "fraction": 0.0, "monotonic": False}
    
    first_vid = prediction[0].get("video_id")
    if first_vid != target_video:
        return {"video_match": False, "hits": 0, "fraction": 0.0, "monotonic": False}
    
    for p in prediction:
        if p.get("video_id") != target_video:
            return {"video_match": False, "hits": 0, "fraction": 0.0, "monotonic": False}
            
    hits = 0
    frames = []
    for p, (start_f, end_f) in zip(prediction, ordered_intervals):
        fid = p.get("frame_id")
        frames.append(fid)
        if start_f <= fid <= end_f:
            hits += 1
            
    monotonic = all(frames[i] < frames[i+1] for i in range(len(frames)-1))
    fraction = hits / len(ordered_intervals)
    return {"video_match": True, "hits": hits, "fraction": fraction, "monotonic": monotonic}

def test_kis_evaluator():
    intervals = [("L22_V001", 100, 200), ("L22_V001", 500, 600)]
    assert evaluate_kis_result({"video_id": "L22_V001", "frame_id": 150}, intervals) is True
    assert evaluate_kis_result({"video_id": "L22_V001", "frame_id": 100}, intervals) is True # boundary
    assert evaluate_kis_result({"video_id": "L22_V001", "frame_id": 200}, intervals) is True # boundary
    assert evaluate_kis_result({"video_id": "L22_V001", "frame_id": 550}, intervals) is True # multiple interval
    assert evaluate_kis_result({"video_id": "L22_V001", "frame_id": 250}, intervals) is False # miss
    assert evaluate_kis_result({"video_id": "L22_V002", "frame_id": 150}, intervals) is False # wrong video

def test_qa_evaluator():
    # Correct all
    r = evaluate_qa_result({"video_id": "L22_V001", "frame_id": 150, "answer": "40 độ C"}, "L22_V001", [100, 200], ["40", "40 độ C"])
    assert r["localization"] is True and r["answer"] is True and r["abstained"] is False
    
    # Alias / substring
    r = evaluate_qa_result({"video_id": "L22_V001", "frame_id": 150, "answer": "nhiệt độ đạt 40 độ"}, "L22_V001", [100, 200], ["40 độ", "40"])
    assert r["answer"] is True
    
    # Wrong answer
    r = evaluate_qa_result({"video_id": "L22_V001", "frame_id": 150, "answer": "50 độ"}, "L22_V001", [100, 200], ["40", "40 độ C"])
    assert r["localization"] is True and r["answer"] is False
    
    # Wrong frame
    r = evaluate_qa_result({"video_id": "L22_V001", "frame_id": 300, "answer": "40 độ C"}, "L22_V001", [100, 200], ["40", "40 độ C"])
    assert r["localization"] is False and r["answer"] is True
    
    # Negative question abstain
    r = evaluate_qa_result({"video_id": "L22_V001", "frame_id": 150, "answer": ""}, None, None, [])
    assert r["answer"] is True and r["abstained"] is True
    
    # Negative question false answer
    r = evaluate_qa_result({"video_id": "L22_V001", "frame_id": 150, "answer": "Paris"}, None, None, [])
    assert r["answer"] is False and r["abstained"] is False

def test_trake_evaluator():
    intervals = [[100, 200], [500, 600], [800, 900]]
    
    # Full hit and monotonic
    pred = [{"video_id": "L22_V001", "frame_id": 150}, {"video_id": "L22_V001", "frame_id": 550}, {"video_id": "L22_V001", "frame_id": 850}]
    r = evaluate_trake_result(pred, "L22_V001", intervals)
    assert r["video_match"] is True and r["hits"] == 3 and r["fraction"] == 1.0 and r["monotonic"] is True
    
    # Partial hit
    pred_part = [{"video_id": "L22_V001", "frame_id": 150}, {"video_id": "L22_V001", "frame_id": 400}, {"video_id": "L22_V001", "frame_id": 850}]
    r = evaluate_trake_result(pred_part, "L22_V001", intervals)
    assert r["hits"] == 2 and r["fraction"] == pytest.approx(2/3)
    
    # Wrong video
    pred_wvid = [{"video_id": "L22_V002", "frame_id": 150}, {"video_id": "L22_V002", "frame_id": 550}, {"video_id": "L22_V002", "frame_id": 850}]
    r = evaluate_trake_result(pred_wvid, "L22_V001", intervals)
    assert r["video_match"] is False
    
    # Non-monotonic order
    pred_non_mono = [{"video_id": "L22_V001", "frame_id": 850}, {"video_id": "L22_V001", "frame_id": 550}, {"video_id": "L22_V001", "frame_id": 150}]
    r = evaluate_trake_result(pred_non_mono, "L22_V001", intervals)
    assert r["monotonic"] is False
