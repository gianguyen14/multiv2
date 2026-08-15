import json

with open("data/processed-validation/three-video-final/L22_V001/asr.json") as f:
    asr = json.load(f)
for seg in asr:
    txt = seg.get("raw_text", "")
    if any(w in txt.lower() for w in ["độ", "nhiệt", "cảnh", "rùa", "pháo đài", "cháy"]):
        print(f"[{seg.get('start_frame')}-{seg.get('end_frame')}] ({seg.get('start_seconds')}s-{seg.get('end_seconds')}s): {txt}")
