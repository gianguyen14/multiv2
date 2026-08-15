import json

with open("data/processed-validation/three-video-final/L22_V001/asr.json") as f:
    asr = json.load(f)
print("=== L22_V001 ASR with '40' or 'độ' or 'nhiệt' ===")
for seg in asr:
    txt = seg.get("raw_text", "")
    if any(w in txt for w in ["40", "độ", "nhiệt", "Cảnh", "rùa"]):
        print(f"[{seg.get('start_frame')}-{seg.get('end_frame')}] ({seg.get('start_seconds')}s): {txt}")

with open("data/processed-validation/three-video-final/L22_V001/ocr.json") as f:
    ocr = json.load(f)
print("\n=== L22_V001 OCR with '40' or 'độ' or 'Cảnh' ===")
for item in ocr:
    txt = item.get("raw_text", "")
    if any(w in txt for w in ["40", "độ", "Cảnh", "Nguyễn"]):
        print(f"Frame {item.get('source_frame_index_zero_based')} ({item.get('timestamp_seconds')}s): {txt}")

