import json

with open("data/processed-validation/three-video-final/L22_V002/asr.json") as f:
    asr2 = json.load(f)
print("=== L22_V002 ASR ===")
for seg in asr2:
    txt = seg.get("raw_text", "")
    if any(w in txt.lower() for w in ["trầm cảm", "smartphone", "thuyên tắc", "bác sĩ"]):
        print(f"[{seg.get('start_frame')}-{seg.get('end_frame')}] ({seg.get('start_seconds')}s-{seg.get('end_seconds')}s): {txt}")

with open("data/processed-validation/three-video-final/L22_V003/asr.json") as f:
    asr3 = json.load(f)
print("\n=== L22_V003 ASR ===")
for seg in asr3:
    txt = seg.get("raw_text", "")
    if any(w in txt.lower() for w in ["bolivia", "cháy rừng", "vietnam airlines", "explore", "bầu cử"]):
        print(f"[{seg.get('start_frame')}-{seg.get('end_frame')}] ({seg.get('start_seconds')}s-{seg.get('end_seconds')}s): {txt}")
