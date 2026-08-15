import json

def get_interval(window_s):
    # given a window start in seconds (e.g., 300, 600, 900), and knowing the snippet is 30s long,
    # assume ~30fps for the source video. We'll use a broad interval to accommodate uncertainty.
    # We will expand it by +-30s.
    start_s = max(0, window_s - 30)
    end_s = window_s + 60
    return [int(start_s * 30), int(end_s * 30)]

kis = []
qa = []
trake = []

# Existing M26.1 queries (incorporating them as baseline check)
kis.extend([
    {"query": "khởi công dự án cải tạo đền thờ Nguyễn Hữu Cảnh", "video_id": "L22_V001", "accepted_frame_interval": [10000, 16000], "clue": "Baseline"},
    {"query": "thiếu niên nghiện smartphone có nguy cơ trầm cảm", "video_id": "L22_V002", "accepted_frame_interval": [18000, 19000], "clue": "Baseline"},
    {"query": "bệnh nhân bị thuyên tắc phổi cấp", "video_id": "L22_V002", "accepted_frame_interval": [13000, 15000], "clue": "Baseline"},
    {"query": "cháy rừng dữ dội ở Bolivia", "video_id": "L22_V003", "accepted_frame_interval": [1000, 4000], "clue": "Baseline"},
    {"query": "nhiệt độ đạt 40 độ C", "video_id": "L22_V001", "accepted_frame_interval": [5000, 7000], "clue": "Baseline"}
])

qa.extend([
    {"question": "Dự án cải tạo đền thờ nào đang được khởi công?", "video_id": "L22_V001", "accepted_answers": ["Nguyễn Hữu Cảnh", "đền thờ Nguyễn Hữu Cảnh"], "accepted_frame_interval": [10000, 16000]},
    {"question": "Thiếu niên nghiện smartphone dễ bị bệnh gì?", "video_id": "L22_V002", "accepted_answers": ["rối loạn lo âu và trầm cảm", "trầm cảm"], "accepted_frame_interval": [18000, 19000]},
    {"question": "Cháy rừng dữ dội xảy ra ở đâu?", "video_id": "L22_V003", "accepted_answers": ["Bolivia"], "accepted_frame_interval": [1000, 4000]},
    {"question": "Nhiệt độ đạt bao nhiêu độ C?", "video_id": "L22_V001", "accepted_answers": ["40", "40 độ C"], "accepted_frame_interval": [5000, 7000]}
])

trake.extend([
    {
      "events": [
        "khởi công dự án cải tạo đền thờ Nguyễn Hữu Cảnh",
        "nhiệt độ đạt 40 độ C"
      ],
      "video_id": "L22_V001",
      "ordered_intervals": [[10000, 16000], [5000, 7000]]
    }
])


# NEW VIDEOS (V004 - V012)
new_events = {
    "L22_V004": [
        (0, "thi công cầu nam lý và cầu tăng long"),
        (0, "cô gái người nước ngoài rơi lầu tử vong ở Thủ Đức"),
        (600, "trồng rêu trên mái nhà để làm giảm nhiệt độ"),
        (900, "đoạn tuyến cao tốc này do nhà nước đại diện chủ sở hữu")
    ],
    "L22_V005": [
        (0, "khép kính 2 loạn còn lại của Văn Đại 2"),
        (0, "Bảo Trà Bi dự báo sóng biển cao 2 mét"),
        (300, "giác cố tạm thời khu vực sạc lỡ"),
        (600, "trò chơi video, cosplay từ 36 quốc gia trang tài")
    ],
    "L22_V006": [
        (0, "bán hàng lũ động bình ổng thị trường trong 30 ngày"),
        (0, "tuyển sinh đại học tăng mạnh nhất ở khoa học giáo dục đào tạo giáo viên"),
        (300, "nồng dân phấn khởi khi tăng lệ nhuận từ lúa gạo"),
        (600, "lễ hội truyền thống Saborun tại Kyrgyzstan")
    ],
    "L22_V007": [
        (0, "hôm chui Nguyễn Văn Linh, Nguyễn Hủ Thò"),
        (0, "công trình trường học mới"),
        (300, "Phẫu thuật tạo hình đoàn thanh môn ghép khí quản"),
        (600, "yên nghĩ của khoảng 8.500 người, cố tổng thống, danh nhân"),
        (900, "xe tải bị đức rời và dính vào thùng xe đầu kéo")
    ],
    "L22_V008": [
        (0, "treo cầu tổ quốc, nghỉ lễ quốc khánh 2.9"),
        (0, "thi tốt nghiệp trung học phổ thông 2025 gồm 4 môn"),
        (300, "giá dầu DSN giảm 737 đồng 1 lít"),
        (900, "văn đáng của bà Nguyễn Thị Kim Hoa bốt cháy")
    ],
    "L22_V009": [
        (0, "hầm chui dài hơn 400m tại phan thúc xuyện Trần Quốc Hoàng"),
        (0, "đường nối trường quốc hoan cùng hoa sẵn sàng thông xe ngày 10 tháng 8"),
        (600, "đường hầm kiên cố về khoảng 220 mét đào về phía ngân hàng")
    ],
    "L22_V010": [
        (0, "Sạch lỡ tại Sơn La, một homestay bị buổi lấp"),
        (0, "Mấy mây rơi ở Brazil, 61 người thiệt mạng"),
        (300, "xách kiên lậu con sẽ không dùng"),
        (600, "nhà hàng Plosi thay đổi cộng thức đường phố truyền thống")
    ],
    "L22_V011": [
        (0, "xe dựng cầu rạch đĩa kết nối giao thôn khu vực quần 7 với huyện nhà bè"),
        (300, "Cổng ưu Nguyễn Tấn Thanh hoàn thành"),
        (600, "lấp học này giúp mọi người bất bị côn lập"),
        (900, "thành lập bảo tàng về trung tâm dân quá đôi thị")
    ],
    "L22_V012": [
        (0, "kế hoạch ứng phó phòng bệnh sợi tại Thành phố Chí Minh"),
        (0, "Bá ô tô Tông Liên Hoàng trong mưa cao tốt nội bài lau cai"),
        (300, "nhà dân, sân dù cao hơn mặt đường, kê lên cài gỗ"),
        (600, "tập yoga có thể đem lại lợi ích sức khỏe lâu dài")
    ]
}

new_qa = [
    {"q": "Hai cây cầu nào đang được yêu cầu tăng tốc thi công?", "a": ["nam lý và tăng long", "cầu nam lý và cầu tăng long"], "v": "L22_V004", "t": 0},
    {"q": "Bão nào dự báo tạo sóng biển cao 2 mét?", "a": ["Trà Bi", "Bảo Trà Bi", "Prapiroon"], "v": "L22_V005", "t": 0},
    {"q": "Lĩnh vực nào tuyển sinh đại học tăng mạnh nhất?", "a": ["khoa học giáo dục đào tạo giáo viên", "đào tạo giáo viên"], "v": "L22_V006", "t": 0},
    {"q": "Hầm chui Nguyễn Văn Linh giao với đường nào?", "a": ["Nguyễn Hữu Thọ", "Nguyễn Hủ Thò"], "v": "L22_V007", "t": 0},
    {"q": "Kỳ thi tốt nghiệp THPT 2025 sẽ có bao nhiêu môn?", "a": ["4", "4 môn"], "v": "L22_V008", "t": 0},
    {"q": "Đường hầm kiên cố dài bao nhiêu mét được phát hiện?", "a": ["220", "220 mét"], "v": "L22_V009", "t": 600},
    {"q": "Có bao nhiêu người thiệt mạng trong vụ rơi máy bay ở Brazil?", "a": ["61", "61 người"], "v": "L22_V010", "t": 0},
    {"q": "Cầu Rạch Đĩa kết nối quận 7 với huyện nào?", "a": ["nhà bè", "huyện nhà bè"], "v": "L22_V011", "t": 0},
    {"q": "Sở Y tế thành phố lên kế hoạch ứng phó phòng bệnh gì?", "a": ["sợi", "bệnh sợi", "sởi"], "v": "L22_V012", "t": 0}
]

neg_qa = [
    {"q": "Dự án cải tạo đền thờ Nguyễn Trãi khi nào hoàn thành?", "a": [], "v": "L22_V001", "t": 0},
    {"q": "Thiếu niên nghiện smartphone dễ bị đau dạ dày không?", "a": [], "v": "L22_V002", "t": 0},
    {"q": "Cháy rừng ở Brazil làm bao nhiêu người chết?", "a": [], "v": "L22_V003", "t": 0},
    {"q": "Bão Trà Bi làm bao nhiêu người mất tích?", "a": [], "v": "L22_V005", "t": 0},
    {"q": "Cầu nam lý đã hoàn thành vào tháng mấy?", "a": [], "v": "L22_V004", "t": 0},
    {"q": "Kỳ thi tốt nghiệp 2025 có bắt buộc môn Lịch sử không?", "a": [], "v": "L22_V008", "t": 0},
    {"q": "Hầm chui Trần Quốc Hoàn dài 1000m phải không?", "a": [], "v": "L22_V009", "t": 0},
    {"q": "Xe tải đứt rời cabin ở quận mấy?", "a": [], "v": "L22_V007", "t": 900},
    {"q": "Đại học khoa học tự nhiên có tăng tuyển sinh không?", "a": [], "v": "L22_V006", "t": 0},
    {"q": "Vụ tông xe liên hoàn ở Nội Bài xảy ra vào ngày nào?", "a": [], "v": "L22_V012", "t": 0}
]

for v, events in new_events.items():
    ev_list = []
    int_list = []
    for t, e in events:
        interval = get_interval(t) if t > 0 else [0, 6000]
        kis.append({
            "query": e,
            "video_id": v,
            "accepted_frame_interval": interval,
            "clue": "Derived from ASR sampling"
        })
        ev_list.append(e)
        int_list.append(interval)
    
    if len(ev_list) >= 2:
        trake.append({
            "events": ev_list[:2],
            "video_id": v,
            "ordered_intervals": int_list[:2]
        })

for qa_item in new_qa:
    t = qa_item["t"]
    qa.append({
        "question": qa_item["q"],
        "video_id": qa_item["v"],
        "accepted_answers": qa_item["a"],
        "accepted_frame_interval": get_interval(t) if t > 0 else [0, 6000]
    })

for qa_item in neg_qa:
    t = qa_item["t"]
    qa.append({
        "question": qa_item["q"],
        "video_id": qa_item["v"],
        "accepted_answers": qa_item["a"],
        "accepted_frame_interval": get_interval(t) if t > 0 else [0, 6000]
    })

output = {
    "schema": "internal_m27_representative_v1",
    "kis": kis,
    "qa": qa,
    "trake": trake
}

with open("data/validation/m27_representative_gt.json", "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Created GT with {len(kis)} KIS, {len(qa)} QA, {len(trake)} TRAKE")
