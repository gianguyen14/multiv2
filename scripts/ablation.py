import os
import json
from pathlib import Path
from eval_mini_gt import run_eval

def run_ablation():
    gt_path = Path("data/validation/three_video_ground_truth/mini_gt.json")
    with open(gt_path) as f:
        gt = json.load(f)
        
    configs = [
        {"name": "visual_only", "ocr": "false", "asr": "false"},
        {"name": "visual_ocr", "ocr": "true", "asr": "false"},
        {"name": "visual_asr", "ocr": "false", "asr": "true"},
        {"name": "visual_ocr_asr", "ocr": "true", "asr": "true"}
    ]
    
    import subprocess
    import sys
    
    results = {}
    
    for c in configs:
        env = os.environ.copy()
        env["SEARCH_ENABLE_OCR"] = c["ocr"]
        env["SEARCH_ENABLE_ASR"] = c["asr"]
        print(f"Running ablation: {c['name']}")
        
        proc = subprocess.run([sys.executable, "eval_mini_gt.py"], env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr)
        
        # Parse stdout to get metrics dict
        try:
            # Find the JSON block
            out = proc.stdout
            idx = out.rfind("{")
            if idx != -1:
                metrics = json.loads(out[idx:])
                results[c["name"]] = metrics
        except Exception as e:
            print("Failed to parse", e)
            
    with open("eval/baselines/ablation_three_video.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_ablation()
