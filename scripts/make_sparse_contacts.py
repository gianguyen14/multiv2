import subprocess
import os
import glob
from pathlib import Path

os.makedirs("contact_sheets_sparse", exist_ok=True)
videos = sorted(glob.glob("data/test-videos/*.mp4"))

for v in videos:
    vid = Path(v).stem
    out_path = f"contact_sheets_sparse/{vid}.jpg"
    if os.path.exists(out_path):
        continue
    # Extract 1 frame every 60 seconds and tile them into a grid
    cmd = [
        "ffmpeg", "-y", "-i", v,
        "-vf", "fps=1/60,scale=320:-1,tile=6x6",
        "-frames:v", "1", "-q:v", "2",
        out_path
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"Created {out_path}")

