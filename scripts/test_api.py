import subprocess
import time
import requests
import json
import os

os.environ["VIDEO_PROCESSED_ROOT"] = "data/processed-validation/full-3-videos"
server = subprocess.Popen(["./.venv/bin/python", "-m", "uvicorn", "backend.app.main:app", "--port", "8000"])

try:
    time.sleep(10) # wait for startup
    
    print("Health:")
    print(requests.get("http://localhost:8000/health/live").json())
    print(requests.get("http://localhost:8000/health/ready").json())
    
    print("Search:")
    print(requests.post("http://localhost:8000/api/v1/search", json={"query": "cháy rừng"}).json())
    
    print("TRAKE:")
    print(requests.post("http://localhost:8000/api/v1/trake", json={"events": ["cháy rừng bốc lên", "người dân dập lửa"]}).json())
    
    print("Invalid Input:")
    res = requests.post("http://localhost:8000/api/v1/search", json={"query": "cháy rừng", "top_k": -1})
    print(f"Status: {res.status_code}")
    print(res.json())
    
    print("Path traversal frame:")
    res = requests.get("http://localhost:8000/api/frames/L22_V001/../../../etc/passwd")
    print(f"Status: {res.status_code}")
    
finally:
    server.terminate()
    server.wait()
