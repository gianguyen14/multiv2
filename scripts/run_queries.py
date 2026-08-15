import subprocess
import os

os.environ["VIDEO_PROCESSED_ROOT"] = "data/processed-validation/full-3-videos"
executable = "./.venv/bin/python"

kis_queries = [
    "khởi công dự án cải tạo đền thờ",
    "thiếu niên nghiện smartphone",
    "bác sĩ đang khám bệnh cho người dân",
    "rừng bị cháy dữ dội",
    "nhiệt độ lên tới 40 độ"
]

qa_queries = [
    "Dự án cải tạo đền thờ nào đang được khởi công?",
    "Thiếu niên nghiện smartphone dễ bị bệnh gì?",
    "Nhiệt độ đạt bao nhiêu độ C?",
    "Bác sĩ khám bệnh ở đâu?",
    "Cháy rừng xảy ra ở nước nào?"
]

trake_queries = [
    '["cháy rừng bốc lên dữ dội", "người dân đang dập lửa", "thiệt hại sau vụ cháy"]',
    '["bác sĩ chuẩn bị dụng cụ", "bác sĩ khám cho bệnh nhân", "bệnh nhân xuất viện"]'
]

print("=== KIS ===")
for q in kis_queries:
    print(f"Query: {q}")
    result = subprocess.run([executable, "projectctl.py", "kis", q, "--top-k", "1"], capture_output=True, text=True)
    print(result.stdout)

print("=== QA ===")
for q in qa_queries:
    print(f"Query: {q}")
    result = subprocess.run([executable, "projectctl.py", "qa", q, "--top-k", "1"], capture_output=True, text=True)
    print(result.stdout)

print("=== TRAKE ===")
for q in trake_queries:
    print(f"Query: {q}")
    result = subprocess.run([executable, "projectctl.py", "trake", q], capture_output=True, text=True)
    print(result.stdout)
