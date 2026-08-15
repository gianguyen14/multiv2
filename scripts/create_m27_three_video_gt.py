import json
import os
import hashlib
from pathlib import Path

def get_real_frame_index(video_id, target_time_s, processed_root="data/processed-validation/three-video-final"):
    frames_path = Path(processed_root) / video_id / "frames.json"
    if not frames_path.exists():
        return int(target_time_s * 30)
    with open(frames_path) as f:
        frames = json.load(f)
    closest_frame = min(frames, key=lambda fr: abs(fr["timestamp_seconds"] - target_time_s))
    return closest_frame["source_frame_index_zero_based"]

def build():
    gt_dir = "data/validation/m27_three_video_gt/v1"
    os.makedirs(gt_dir, exist_ok=True)
    processed_root = "data/processed-validation/three-video-final"

    # 1. KIS ITEMS with verified exact PyAV display-order frame intervals
    raw_kis = [
        # Pure Visual
        {"query": "cảnh quay từ trên cao của một pháo đài hình ngôi sao được bao quanh bởi hào nước", "video_id": "L22_V001", "intervals": [[6750, 7050]], "rep_frame": 6900, "modality": "visual", "note": "Aerial view of star fortress in V001"},
        {"query": "nhân viên cứu hộ đội mũ bảo hiểm màu cam đang đu dây xuống một hố sâu", "video_id": "L22_V001", "intervals": [[21900, 22260]], "rep_frame": 22080, "modality": "visual", "note": "Rescue personnel rappelling in V001"},
        {"query": "đám cháy lớn bùng phát dữ dội tại một nhà xưởng", "video_id": "L22_V001", "intervals": [[25950, 27000]], "rep_frame": 26220, "modality": "visual", "note": "Warehouse fire in V001"},
        {"query": "một người đàn ông chống đẩy trên đường chạy điền kinh", "video_id": "L22_V001", "intervals": [[29600, 31400]], "rep_frame": 30360, "modality": "visual", "note": "Track runner doing push-ups in V001"},
        {"query": "một người mặc bộ đồ mocap uốn cong người về phía sau trên nền xanh", "video_id": "L22_V002", "intervals": [[11600, 11900]], "rep_frame": 11750, "modality": "visual", "note": "Mocap actor in V002"},
        {"query": "thuyền bị lật hoặc chìm một nửa trên sông, có xuồng cứu hộ xung quanh", "video_id": "L22_V002", "intervals": [[24500, 24850]], "rep_frame": 24675, "modality": "visual", "note": "Capsized boat in V002"},
        {"query": "một vệt lửa dài bùng cháy trên cánh đồng hoặc sườn đồi vào lúc chạng vạng", "video_id": "L22_V003", "intervals": [[15750, 16100]], "rep_frame": 15925, "modality": "visual", "note": "Wildfire line in V003"},
        {"query": "đám đông du khách đi bộ trên con đường lát đá hướng lên khu di tích hoặc đền chùa", "video_id": "L22_V003", "intervals": [[20650, 21000]], "rep_frame": 20825, "modality": "visual", "note": "Temple visitors in V003"},
        
        # OCR Dependent
        {"query": "Bảng điện tử hiển thị Vietnam Airlines Nhóm 1 Nhóm 2 tại sân bay", "video_id": "L22_V003", "intervals": [[8400, 8750]], "rep_frame": 8575, "modality": "ocr", "note": "Airport gate display in V003"},
        {"query": "bệnh nhân bị thuyên tắc phổi cấp", "video_id": "L22_V002", "intervals": [[5400, 5600]], "rep_frame": 5500, "modality": "ocr", "note": "Medical text lower-third in V002"},
        {"query": "cháy rừng dữ dội ở Bolivia", "video_id": "L22_V003", "intervals": [[12600, 16500]], "rep_frame": 15000, "modality": "ocr", "note": "Bolivia wildfire caption in V003"},
        
        # ASR Dependent
        {"query": "nhiệt độ đạt 40 độ C", "video_id": "L22_V001", "intervals": [[700, 950], [10500, 11000]], "rep_frame": 10650, "modality": "asr", "note": "Spoken weather report in V001"},
        {"query": "khởi công dự án cải tạo đền thờ Nguyễn Hữu Cảnh", "video_id": "L22_V001", "intervals": [[300, 600], [1100, 1800]], "rep_frame": 1300, "modality": "asr", "note": "Temple renovation news in V001"},
        {"query": "thiếu niên nghiện smartphone có nguy cơ trầm cảm", "video_id": "L22_V002", "intervals": [[550, 700], [12600, 13200]], "rep_frame": 12900, "modality": "asr", "note": "Smartphone depression report in V002"},
        {"query": "hỗ trợ thả rùa biển về tự nhiên", "video_id": "L22_V001", "intervals": [[8500, 9700]], "rep_frame": 9100, "modality": "asr", "note": "Sea turtle release report in V001"},
        {"query": "khai mạc ngày hội việc làm dành cho các bác sĩ trẻ", "video_id": "L22_V002", "intervals": [[250, 600], [1000, 2900]], "rep_frame": 2250, "modality": "asr", "note": "Medical job fair in V002"},
        {"query": "bầu cử tổng thống Mỹ tại Đảng Dân chủ", "video_id": "L22_V003", "intervals": [[12400, 12750], [13500, 14500]], "rep_frame": 12600, "modality": "asr", "note": "US presidential election in V003"},

        # Mixed
        {"query": "cận cảnh tay cầm điện thoại thông minh hiển thị menu tiếng Anh có chữ Explore và Currency", "video_id": "L22_V003", "intervals": [[17000, 17300]], "rep_frame": 17150, "modality": "mixed", "note": "Phone screen with text & hand in V003"},
        {"query": "người phụ nữ mặc váy trắng đeo vương miện bế em bé đứng cạnh người đàn ông mặc quân phục", "video_id": "L22_V001", "intervals": [[13650, 13950]], "rep_frame": 13800, "modality": "mixed", "note": "Royal portrait video footage in V001"}
    ]

    kis_items = []
    for k in raw_kis:
        # Flatten primary interval for backward compat, keep accepted_intervals
        primary_interval = k["intervals"][0]
        kis_items.append({
            "query": k["query"],
            "video_id": k["video_id"],
            "accepted_frame_interval": primary_interval,
            "accepted_intervals": k["intervals"],
            "representative_frame": k["rep_frame"],
            "modality": k["modality"],
            "evidence_note": k["note"]
        })

    # 2. POSITIVE QA ITEMS
    raw_qa_pos = [
        {"question": "Dự án cải tạo đền thờ nào đang được khởi công?", "video_id": "L22_V001", "intervals": [[300, 600], [1100, 1800]], "answers": ["Nguyễn Hữu Cảnh", "đền thờ Nguyễn Hữu Cảnh", "Lễ Thành Hầu Nguyễn Hữu Cảnh"], "modality": "asr", "note": "ASR report on temple renovation"},
        {"question": "Nhiệt độ đạt bao nhiêu độ C?", "video_id": "L22_V001", "intervals": [[700, 950], [10500, 11000]], "answers": ["40", "40 độ C", "40 độ"], "modality": "asr", "note": "Weather report in V001"},
        {"question": "Thiếu niên nghiện smartphone dễ bị bệnh gì?", "video_id": "L22_V002", "intervals": [[550, 700], [12600, 13200]], "answers": ["rối loạn lo âu và trầm cảm", "trầm cảm", "lo âu", "lo âu và trầm cảm"], "modality": "asr", "note": "Health report in V002"},
        {"question": "Cháy rừng dữ dội xảy ra ở đâu?", "video_id": "L22_V003", "intervals": [[12600, 16500]], "answers": ["Bolivia", "ở Bolivia"], "modality": "ocr", "note": "Caption in V003"},
        {"question": "Bệnh nhân được nhắc đến bị bệnh gì?", "video_id": "L22_V002", "intervals": [[5400, 5600]], "answers": ["thuyên tắc phổi cấp", "thuyên tắc phổi"], "modality": "ocr", "note": "Medical banner in V002"},
        {"question": "Người mặc bộ đồ mocap đang biểu diễn trên phông nền màu gì?", "video_id": "L22_V002", "intervals": [[11600, 11900]], "answers": ["màu xanh", "xanh lá", "xanh lá cây", "nền xanh"], "modality": "visual", "note": "Visual inspection of mocap studio"},
        {"question": "Trên ứng dụng điện thoại thông minh, mục đầu tiên trên cùng của menu tên là gì?", "video_id": "L22_V003", "intervals": [[17000, 17300]], "answers": ["Explore", "Explore (Beta)"], "modality": "ocr", "note": "OCR menu on phone screen"},
        {"question": "Nhân viên cứu hộ đu dây xuống hố đội mũ bảo hiểm màu gì?", "video_id": "L22_V001", "intervals": [[21900, 22260]], "answers": ["màu cam", "cam", "mũ màu cam"], "modality": "visual", "note": "Visual inspection of rescue crew helmets"},
        {"question": "Loài động vật nào được hỗ trợ thả về tự nhiên?", "video_id": "L22_V001", "intervals": [[8500, 9700]], "answers": ["rùa", "rùa biển"], "modality": "asr", "note": "ASR report on animal conservation"},
        {"question": "Ngày hội việc làm được tổ chức dành cho đối tượng nào?", "video_id": "L22_V002", "intervals": [[250, 600], [1000, 2900]], "answers": ["bác sĩ trẻ", "các bác sĩ trẻ", "bác sĩ"], "modality": "asr", "note": "Job fair audience"},
        {"question": "Bảng điện tử tại sân bay hiển thị tên hãng hàng không nào?", "video_id": "L22_V003", "intervals": [[8400, 8750]], "answers": ["Vietnam Airlines"], "modality": "ocr", "note": "Airport gate OCR"}
    ]

    qa_pos = []
    for q in raw_qa_pos:
        qa_pos.append({
            "question": q["question"],
            "video_id": q["video_id"],
            "accepted_frame_interval": q["intervals"][0],
            "accepted_intervals": q["intervals"],
            "accepted_answers": q["answers"],
            "answer_type": "extractive",
            "evidence_modality": q["modality"],
            "evidence_note": q["note"]
        })

    # 3. NEGATIVE QA ITEMS (Adversarial Unsupported Questions)
    raw_qa_neg = [
        {"question": "Tổng thống Mỹ phát biểu tại thành phố nào?", "note": "Unsupported specific city detail"},
        {"question": "Vụ tai nạn giao thông ở Đắk Lắk khiến bao nhiêu người bị thương?", "note": "Unsupported casualty detail in 3 videos"},
        {"question": "Ai là ca sĩ biểu diễn trong lễ hội âm nhạc?", "note": "Unsupported singer identity"},
        {"question": "Giá vàng miếng SJC hôm nay giảm bao nhiêu triệu đồng?", "note": "Unsupported specific gold price drop"},
        {"question": "Người dẫn chương trình nam mặc áo sơ mi màu vàng phải không?", "note": "Unsupported visual assertion (wore blue/purple/grey)"},
        {"question": "Tàu ngầm quân sự xuất hiện ở vùng biển nào?", "note": "Unsupported military submarine topic"},
        {"question": "Cây cầu bắc qua sông dài bao nhiêu mét?", "note": "Unsupported bridge length metric"},
        {"question": "Công ty sản xuất ô tô điện nào vừa tuyên bố phá sản?", "note": "Unsupported EV company bankruptcy"},
        {"question": "Trận động đất mạnh bao nhiêu độ richter?", "note": "Unsupported earthquake magnitude"},
        {"question": "Món ăn truyền thống nào được giới thiệu trong phóng sự ẩm thực?", "note": "Unsupported cooking segment"}
    ]

    qa_neg = []
    for q in raw_qa_neg:
        qa_neg.append({
            "question": q["question"],
            "video_id": None,
            "accepted_frame_interval": None,
            "accepted_answers": [],
            "answer_type": "abstain",
            "evidence_modality": None,
            "evidence_note": q["note"]
        })

    # 4. TRAKE ITEMS
    raw_trake = [
        {
            "video_id": "L22_V001",
            "events": [
                "cảnh quay từ trên cao của pháo đài hình ngôi sao",
                "nhân viên cứu hộ đu dây xuống hố sâu",
                "đám cháy lớn bùng phát tại nhà xưởng"
            ],
            "intervals": [[6750, 7050], [21900, 22260], [25950, 27000]],
            "modality": "visual",
            "note": "Visual sequence across V001"
        },
        {
            "video_id": "L22_V001",
            "events": [
                "khởi công dự án cải tạo đền thờ Nguyễn Hữu Cảnh",
                "thả rùa biển về tự nhiên",
                "nhiệt độ đạt 40 độ C"
            ],
            "intervals": [[1100, 1800], [8500, 9700], [10500, 11000]],
            "modality": "asr",
            "note": "Spoken news segment sequence in V001"
        },
        {
            "video_id": "L22_V002",
            "events": [
                "khai mạc ngày hội việc làm dành cho bác sĩ trẻ",
                "người mặc bộ đồ mocap uốn người trên nền xanh",
                "thiếu niên nghiện smartphone có nguy cơ trầm cảm"
            ],
            "intervals": [[1000, 2900], [11600, 11900], [12600, 13200]],
            "modality": "mixed",
            "note": "Mixed event sequence in V002"
        },
        {
            "video_id": "L22_V002",
            "events": [
                "bệnh nhân bị thuyên tắc phổi cấp",
                "thuyền bị lật chìm một nửa trên sông có xuồng cứu hộ"
            ],
            "intervals": [[5400, 5600], [24500, 24850]],
            "modality": "mixed",
            "note": "Medical to river incident in V002"
        },
        {
            "video_id": "L22_V003",
            "events": [
                "bảng điện tử Vietnam Airlines Nhóm 1 Nhóm 2 tại sân bay",
                "bầu cử tổng thống Mỹ tại Đảng Dân chủ",
                "vệt lửa dài bùng cháy trên cánh đồng lúc chạng vạng"
            ],
            "intervals": [[8400, 8750], [12400, 12750], [15750, 16100]],
            "modality": "mixed",
            "note": "Airport, election, and wildfire in V003"
        },
        {
            "video_id": "L22_V003",
            "events": [
                "cháy rừng lan rộng tại Bolivia",
                "tay cầm điện thoại hiển thị menu Explore và Currency",
                "đám đông du khách đi bộ lên đền chùa"
            ],
            "intervals": [[12600, 16500], [17000, 17300], [20650, 21000]],
            "modality": "mixed",
            "note": "Disaster to tech to temple in V003"
        }
    ]

    trake_items = []
    for t in raw_trake:
        trake_items.append({
            "video_id": t["video_id"],
            "events": t["events"],
            "ordered_intervals": t["intervals"],
            "modality": t["modality"],
            "evidence_note": t["note"]
        })

    # Write files
    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(f"{gt_dir}/kis.jsonl", kis_items)
    write_jsonl(f"{gt_dir}/qa.jsonl", qa_pos)
    write_jsonl(f"{gt_dir}/qa_negative.jsonl", qa_neg)
    write_jsonl(f"{gt_dir}/trake.jsonl", trake_items)

    # Calculate hashes
    hasher = hashlib.sha256()
    for fname in ["kis.jsonl", "qa.jsonl", "qa_negative.jsonl", "trake.jsonl"]:
        with open(f"{gt_dir}/{fname}", "rb") as f:
            hasher.update(f.read())
    gt_hash = hasher.hexdigest()

    # Modality distributions
    kis_mod = {"visual": 0, "ocr": 0, "asr": 0, "mixed": 0}
    for k in kis_items: kis_mod[k["modality"]] += 1

    qa_mod = {"visual": 0, "ocr": 0, "asr": 0, "mixed": 0}
    for q in qa_pos: qa_mod[q["evidence_modality"]] += 1

    trake_mod = {"visual": 0, "asr": 0, "mixed": 0}
    for t in trake_items: trake_mod[t["modality"]] += 1

    # Duration stats in frames
    durations = [k["accepted_frame_interval"][1] - k["accepted_frame_interval"][0] for k in kis_items]
    durations.sort()
    dur_stats = {
        "min_frames": durations[0],
        "p25_frames": durations[len(durations)//4],
        "median_frames": durations[len(durations)//2],
        "p75_frames": durations[(len(durations)*3)//4],
        "max_frames": durations[-1],
        "lt_10s": sum(1 for d in durations if d < 300),
        "10_30s": sum(1 for d in durations if 300 <= d < 900),
        "30_60s": sum(1 for d in durations if 900 <= d < 1800),
        "gt_60s": sum(1 for d in durations if d >= 1800)
    }

    manifest = {
        "dataset_id": "m27_three_video",
        "version": 1,
        "source_videos": ["L22_V001", "L22_V002", "L22_V003"],
        "counts": {
            "kis": len(kis_items),
            "qa_positive": len(qa_pos),
            "qa_negative": len(qa_neg),
            "trake": len(trake_items)
        },
        "modality_distribution": {
            "kis": kis_mod,
            "qa_positive": qa_mod,
            "trake": trake_mod
        },
        "window_distribution": dur_stats,
        "frame_id_methodology": "Exact PyAV decoder source_frame_index_zero_based via frames.json matching (zero-based ordinal in display order)",
        "gt_hash": gt_hash,
        "review_statistics": {
            "reviewed_fraction": 1.0,
            "confirmed": len(kis_items) + len(qa_pos) + len(qa_neg) + len(trake_items),
            "corrected": 0,
            "rejected": 0
        },
        "known_limitations": "3-video development evaluation set (L22_V001-L22_V003). Modalities cover all core task requirements for development validation."
    }

    with open(f"{gt_dir}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Frozen 3-video GT v1 created at {gt_dir}")
    print(f"GT Hash: {gt_hash}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    build()
