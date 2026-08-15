import json

videos = ["L22_V001", "L22_V002", "L22_V003"]
times = [300, 600, 900]

for vid in videos:
    with open(f"data/processed-validation/m27-representative-12-videos/{vid}/frames.json") as f:
        frames = json.load(f)
    print(f"=== {vid} ===")
    for t in times:
        # find closest frame to timestamp
        closest = min(frames, key=lambda fr: abs(fr["timestamp_seconds"] - t))
        idx = frames.index(closest)
        print(f"Target: {t}s")
        for i in range(max(0, idx-1), min(len(frames), idx+2)):
            fr = frames[i]
            print(f"  Frame {fr['source_frame_index_zero_based']}: pts={fr['pts']}, time={fr['timestamp_seconds']}")
        print()
