import json
from collections import Counter
from pathlib import Path

root = Path("data/processed-validation/full-3-videos")
for video_id in ["L22_V001", "L22_V002", "L22_V003"]:
    ocr = json.loads((root / video_id / "ocr.json").read_text())
    asr = json.loads((root / video_id / "asr.json").read_text())
    
    ocr_texts = [d['normalized_text'] for d in ocr if len(d['normalized_text']) > 5]
    asr_texts = [d['normalized_text'] for d in asr if len(d['normalized_text']) > 5]
    
    print(f"--- {video_id} ---")
    print("Top OCR:", Counter(ocr_texts).most_common(5))
    print("Sample ASR:", asr_texts[:5])
