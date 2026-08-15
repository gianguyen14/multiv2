import json
from pathlib import Path
import faiss

root = Path("data/processed-validation/full-3-videos")
index_dir = root / "index"
current = index_dir / "CURRENT"
if current.exists():
    try:
        data = json.loads(current.read_text())
        gen = data["generation_id"]
    except json.JSONDecodeError:
        gen = current.read_text().strip()
        
    print(f"CURRENT gen: {gen}")
    index_path = index_dir / "generations" / gen / "frames.faiss"
    payload_path = index_dir / "generations" / gen / "payloads.json"
    print(f"FAISS exists: {index_path.exists()}")
    
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        print(f"FAISS vectors: {index.ntotal}")
        payloads = json.loads(payload_path.read_text())
        print(f"Payloads: {len(payloads)}")
        assert index.ntotal == len(payloads), "Vector/Payload count mismatch"
else:
    print("CURRENT missing")
