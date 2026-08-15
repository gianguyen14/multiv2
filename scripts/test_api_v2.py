import subprocess
import time
import requests
import json
import os

os.environ["VIDEO_PROCESSED_ROOT"] = "data/processed-validation/full-3-videos"
server = subprocess.Popen(["./.venv/bin/python", "-m", "uvicorn", "backend.app.main:app", "--port", "8000"])

try:
    time.sleep(10)
    
    print("Search:")
    print(requests.post("http://localhost:8000/api/search", json={"query": "cháy rừng", "query_type": "kis"}).json())
    
    print("TRAKE:")
    print(requests.post("http://localhost:8000/api/search", json={"events": ["cháy rừng bốc lên", "người dân dập lửa"], "query_type": "trake"}).json())
    
    print("Invalid Input:")
    res = requests.post("http://localhost:8000/api/search", json={"query": "cháy rừng", "top_k": -1})
    print(f"Status: {res.status_code}")
    print(res.json())
    
finally:
    server.terminate()
    server.wait()
