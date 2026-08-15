import os, subprocess, json
from pathlib import Path
from backend.app.video.text_backends import FasterWhisperASRBackend

def main():
    videos = sorted([v for v in os.listdir('data/test-videos-m27') if v.endswith('.mp4')])
    asr = FasterWhisperASRBackend()
    results = {}
    
    os.makedirs("artifacts/tmp_audio2", exist_ok=True)
    
    # We will sample 3 distinct 30-second windows per video (e.g. at 5m, 10m, 15m)
    windows = [300, 600, 900]
    
    for v in videos:
        print(f"Sampling {v}...")
        v_path = Path('data/test-videos-m27') / v
        results[v] = {}
        
        for w in windows:
            audio_path = Path(f"artifacts/tmp_audio2/{v}_{w}.wav")
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(w), "-i", str(v_path), 
                    "-t", "30", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
                    str(audio_path)
                ], check=True, capture_output=True)
                
                segments = asr.transcribe(audio_path)
                text = " ".join([s['text'] for s in segments]).strip()
                results[v][f"window_{w}"] = text
                print(f"  {w}s: {text[:100]}...")
            except Exception as e:
                print(f"  Failed at {w}s: {e}")
            
    with open("artifacts/video_transcripts_sampled_windows.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
