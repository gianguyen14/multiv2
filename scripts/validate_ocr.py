import json
import sys
from pathlib import Path

def validate(video_id):
    path = Path(f"data/processed-validation/ocr-asr-3-videos/{video_id}/ocr.json")
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"{video_id}: JSON error: {e}")
        return
    
    print(f"--- {video_id} ---")
    print(f"Total records: {len(data)}")
    
    non_empty = [d for d in data if d.get('normalized_text', '').strip()]
    print(f"Non-empty text: {len(non_empty)}")
    
    # Check valid frame links and boxes
    invalid_links = [d for d in data if d.get('frame_id') is None]
    invalid_boxes = [d for d in data if not isinstance(d.get('boxes'), list)]
    
    print(f"Invalid frame links: {len(invalid_links)}")
    print(f"Invalid boxes: {len(invalid_boxes)}")
    
    # Sample
    if non_empty:
        print("Sample text:", repr(non_empty[0]['normalized_text']))
        
for v in ["L22_V001", "L22_V002", "L22_V003"]:
    validate(v)
