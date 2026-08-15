import os, subprocess, json
from pathlib import Path
from backend.app.video.text_backends import FasterWhisperASRBackend

def main():
    videos = sorted([v for v in os.listdir('data/test-videos') if v.endswith('.mp4')])[3:18] # V004-V018
    asr = FasterWhisperASRBackend()
    results = {}
    
    os.makedirs("artifacts/tmp_audio", exist_ok=True)
    
    for v in videos:
        print(f"Processing {v}...")
        v_path = Path('data/test-videos') / v
        audio_path = Path(f"artifacts/tmp_audio/{v}.wav")
        
        # Extract first 60 seconds of audio
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(v_path), 
                "-t", "120", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
                str(audio_path)
            ], check=True, capture_output=True)
            
            segments = asr.transcribe(audio_path)
            text = " ".join([s['text'] for s in segments]).strip()
            results[v] = text
            print(f"  Snippet: {text[:150]}...")
        except Exception as e:
            print(f"  Failed: {e}")
            
    with open("artifacts/video_transcripts_sample.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
