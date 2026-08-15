import json
import subprocess
import glob
from pathlib import Path
import os

videos = sorted(glob.glob("data/test-videos/*.mp4"))

print("video_id | duration | resolution | description")
for v in videos:
    vid = Path(v).stem
    
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height", "-of", "default=noprint_wrappers=1:nokey=1", v]
    out = subprocess.check_output(cmd, text=True).strip().split('\n')
    if len(out) >= 3:
        width, height, duration = out[0], out[1], out[2]
        print(f"{vid} | {float(duration):.1f}s | {width}x{height} | ...")
    else:
        print(f"{vid} | error | error | ...")

