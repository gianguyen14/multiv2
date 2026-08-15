<div align="center">

# 🎬 Unified AIC Retrieval

### Hệ thống truy hồi video đa phương thức cho Ho Chi Minh City AI Challenge 2026

**Văn bản → Frame · Video Q&A · TRAKE · Tìm bằng ảnh · OCR · ASR · Tinh chỉnh theo thời gian**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-Recommended-2496ED?logo=docker&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](#)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF)](#)
[![SigLIP2](https://img.shields.io/badge/SigLIP2-768D-FF6F00)](#)
[![Status](https://img.shields.io/badge/Release-1.1.0--rc2_prevalidation-orange)](#)

**[English](README.md) · Tiếng Việt**

*Một hệ thống retrieval ưu tiên chạy local/offline, được xây dựng để tìm đúng video, đúng frame và đúng chuỗi sự kiện theo thời gian.*

</div>

---

## ✨ Hệ thống làm được gì?

| Chế độ | Đầu vào | Đầu ra | Tín hiệu chính |
|---|---|---|---|
| 🔎 **Textual KIS** | Mô tả bằng ngôn ngữ tự nhiên | Danh sách `video_id`, `frame_id` đã xếp hạng | SigLIP2 + OCR + ASR + fusion |
| 💬 **Video Q&A** | Câu hỏi về nội dung video | Frame bằng chứng + đường dẫn trả lời | Visual + OCR + ASR |
| 🧭 **TRAKE** | Chuỗi sự kiện có thứ tự | Một video + các keyframe theo đúng thứ tự | Coarse retrieval + temporal refinement + DP alignment |
| 🖼️ **Image Search** | Ảnh truy vấn | Các frame giống về mặt thị giác | SigLIP2 image embeddings |
| 🔤 **OCR / ASR** | Frame + âm thanh | Bằng chứng văn bản có thể tìm kiếm | Tesseract + Faster Whisper |

Lớp retrieval hỗ trợ truy vấn tiếng Việt và tiếng Anh, evidence-aware reranking, temporal deduplication và QueryRefiner local tùy chọn với deterministic fallback.

---

## 🧠 Kiến trúc tổng quan

```text
                               ┌─────────────────────┐
                               │      USER QUERY     │
                               │ text / image / QA   │
                               │ ordered TRAKE events│
                               └──────────┬──────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │   Query Intelligence  │
                              │ VI/EN · lexical · LLM │
                              └───────────┬───────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
             ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
             │   SigLIP2   │       │     OCR     │       │     ASR     │
             │ visual/text │       │  Tesseract  │       │   Whisper   │
             └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
                    │                     │                     │
                    └─────────────────────┼─────────────────────┘
                                          ▼
                                ┌───────────────────┐
                                │ Candidate Fusion  │
                                │ RRF + reranking   │
                                └─────────┬─────────┘
                                          │
                        ┌─────────────────┼─────────────────┐
                        │                 │                 │
                        ▼                 ▼                 ▼
                      KIS               Q&A              TRAKE
                                                          │
                                                          ▼
                                                Dense TemporalRefiner
                                                          │
                                                          ▼
                                               Ordered DP alignment
```

### Frame ID là định danh chuẩn

`frame_id` trả về là **chỉ số zero-based theo thứ tự frame mà PyAV giải mã tuần tự ở display order**.

```python
for frame_id, frame in enumerate(container.decode(stream)):
    ...
```

Hệ thống **không** tái tạo frame ID chuẩn bằng `timestamp × FPS`. Điều này giúp ingestion, retrieval, evaluation và temporal refinement cùng dùng một hệ quy chiếu frame thống nhất.

---

# 🚀 Bắt đầu nhanh với Docker

Docker là runtime được khuyến nghị trên **Linux** và **Windows 10/11**.

Image chứa các dependency Linux cốt lõi dùng bởi ứng dụng, gồm Python 3.12, FFmpeg, Tesseract, Git, GCC và G++.

## Yêu cầu

### Linux

- Docker Engine
- Docker Compose v2 (`docker compose`)

### Windows 10/11

- Docker Desktop
- Bật WSL2 backend
- Git for Windows hoặc Git bên trong WSL2

Trên Windows, cách đơn giản nhất là giữ video, model cache và processed artifacts bên trong thư mục repository.

## 1. Clone repository

```bash
git clone git@github.com:gianguyen14/multiv2.git
cd multiv2
```

> Repository đang private, vì vậy cần cấu hình SSH GitHub trước trên máy chạy.

## 2. Tạo thư mục dữ liệu local

Linux / WSL2:

```bash
mkdir -p data/test-videos data/processed models
```

PowerShell:

```powershell
New-Item -ItemType Directory -Force data/test-videos, data/processed, models
```

Đặt video nguồn tại:

```text
data/test-videos/
```

Docker mount mặc định:

```text
Host                       Container
------------------------------------------------
./data/test-videos   -->   /data/videos       read-only
./data/processed     -->   /data/processed    read/write
./models             -->   /models            read/write
```

## 3. Build image

```bash
docker compose build
```

## 4. Kiểm tra runtime

```bash
docker compose --profile tools run --rm worker env --check
docker compose --profile tools run --rm worker doctor
```

## 5. Chuẩn bị model

Chuẩn bị model visual + ASR mặc định:

```bash
docker compose --profile tools run --rm worker models --prepare
```

Kiểm tra model hiện có:

```bash
docker compose --profile tools run --rm worker models
```

QueryRefiner local là tùy chọn:

```bash
docker compose --profile tools run --rm worker models --prepare --query-refiner
```

Nếu QueryRefiner chưa có model, search vẫn có thể dùng deterministic fallback.

## 6. Preprocess / index video

```bash
docker compose --profile tools run --rm worker preprocess /data/videos
```

Lệnh này chạy ingestion pipeline theo cấu hình và tạo searchable artifacts bên trong processed directory đã mount. Các phần việc tương thích có thể được resume.

Kiểm tra trạng thái:

```bash
docker compose --profile tools run --rm worker status
```

## 7. Khởi động ứng dụng

```bash
docker compose up -d backend
```

Kiểm tra container:

```bash
docker compose ps
```

Theo dõi log:

```bash
docker compose logs -f backend
```

Mở giao diện web:

```text
http://127.0.0.1:8000
```

Health endpoints:

```text
http://127.0.0.1:8000/health/live
http://127.0.0.1:8000/health/ready
http://127.0.0.1:8000/health
```

Linux / WSL2:

```bash
curl http://127.0.0.1:8000/health/ready
```

PowerShell:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health/ready
```

---

# 🎮 Cách sử dụng

Có ba cách chính để dùng project:

1. **Web UI** — phù hợp nhất cho thao tác tương tác.
2. **`projectctl.py` CLI** — phù hợp cho development, experiment, batch và validation.
3. **FastAPI** — phù hợp khi tích hợp với ứng dụng hoặc frontend khác.

## Cách A — Web UI

```bash
docker compose up -d backend
```

Sau đó mở:

```text
http://127.0.0.1:8000
```

Frontend được FastAPI phục vụ trực tiếp nên workflow Docker thông thường không cần chạy thêm frontend server riêng.

---

## Cách B — CLI với `projectctl.py`

Trong Docker, dùng service `worker`:

```bash
docker compose --profile tools run --rm worker --help
```

### 🔎 Textual KIS

Tìm frame phù hợp với mô tả:

```bash
docker compose --profile tools run --rm worker \
  kis "một người phụ nữ mặc áo dài" --top-k 20
```

Ví dụ truy vấn:

```text
"a red car crossing an intersection"
"người đàn ông đang đứng trước màn hình lớn"
"biển số xe 79H-6072"
```

### 💬 Video Q&A

Truy hồi bằng chứng cho câu hỏi:

```bash
docker compose --profile tools run --rm worker \
  qa "Nhiệt độ hiển thị trên màn hình là bao nhiêu?" --top-k 20
```

Q&A kết hợp bằng chứng visual, OCR và ASR có sẵn trước khi xử lý bước trả lời tiếp theo.

### 🧭 TRAKE

TRAKE tìm **chuỗi sự kiện có thứ tự trong cùng một video**.

Cú pháp phân tách bằng `|`:

```bash
docker compose --profile tools run --rm worker \
  trake "người đứng yên | bắt đầu chạy | nhảy lên | tiếp đất" --top-k 30
```

Cú pháp JSON:

```bash
docker compose --profile tools run --rm worker \
  trake '["đứng", "chạy đà", "nhảy", "tiếp đất"]' --top-k 30
```

TRAKE có thể chạy dense temporal refinement quanh các vùng coarse candidate rồi ép thứ tự sự kiện theo chiều thời gian.

Tắt dense temporal refinement để chẩn đoán:

```bash
docker compose --profile tools run --rm worker \
  trake "event one | event two" --no-temporal-refine
```

### 🖼️ Tìm frame bằng ảnh

Nếu ảnh truy vấn có sẵn trong container:

```bash
docker compose --profile tools run --rm worker \
  image-search /data/videos/query.jpg --top-k 20
```

Hoặc dùng HTTP image endpoint từ host.

### 🧠 Xem QueryPlan

```bash
docker compose --profile tools run --rm worker \
  query-plan "biển số xe 79H-6072" --task kis --json
```

Lệnh này hữu ích để kiểm tra query expansion, lexical terms và đường chạy QueryRefiner local.

### 🩺 Chẩn đoán

```bash
docker compose --profile tools run --rm worker doctor
docker compose --profile tools run --rm worker status
docker compose --profile tools run --rm worker info
docker compose --profile tools run --rm worker smoke
```

### 📦 Kiểm tra dataset

```bash
docker compose --profile tools run --rm worker \
  dataset verify /data/videos
```

Workflow validation đại diện trong repository:

```bash
docker compose --profile tools run --rm worker validate-dataset
```

### 📊 Evaluation

```bash
docker compose --profile tools run --rm worker \
  evaluate --competition --ground-truth /path/to/ground_truth
```

Competition scorer có trong repository là **internal provisional metric**, không phải tuyên bố về official competition scoring.

---

## Cách C — HTTP API

Service hiện tại expose retrieval API từ `backend.app.main`.

### KIS request

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "một người phụ nữ mặc áo dài",
    "query_type": "kis",
    "top_k": 20
  }'
```

### Q&A request

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Nhiệt độ hiển thị là bao nhiêu?",
    "query_type": "qa",
    "top_k": 20
  }'
```

### TRAKE request

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "trake",
    "events": ["đứng", "chạy", "nhảy", "tiếp đất"],
    "top_k": 30,
    "temporal_refine": true,
    "query_refine": true,
    "rerank": true
  }'
```

### Image search request

```bash
curl -X POST "http://127.0.0.1:8000/api/search/image?top_k=20" \
  -F "file=@query.jpg"
```

Các định dạng ảnh hỗ trợ: JPEG, PNG và WebP. API giới hạn upload ảnh ở 15 MB.

### Debug QueryPlan qua API

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "biển số xe 79H-6072",
    "query_type": "kis",
    "top_k": 20,
    "debug_query_plan": true
  }'
```

---

# 🧩 Workflow end-to-end điển hình

```text
1. Đặt video vào data/test-videos/
           │
           ▼
2. docker compose build
           │
           ▼
3. models --prepare
           │
           ▼
4. doctor
           │
           ▼
5. preprocess /data/videos
           │
           ▼
6. status
           │
           ▼
7. docker compose up -d backend
           │
           ▼
8. Mở Web UI / chạy KIS / Q&A / TRAKE / image search
           │
           ▼
9. evaluate / benchmark / tune
```

---

# 🐳 Cấu hình Docker

## Đổi vị trí data/model

Compose hỗ trợ:

```text
VIDEOS_DIR
PROCESSED_DIR
MODELS_DIR
HF_HUB_OFFLINE
TRANSFORMERS_OFFLINE
```

Ví dụ Linux / WSL2:

```bash
VIDEOS_DIR=/mnt/videos \
PROCESSED_DIR=/mnt/aic-processed \
MODELS_DIR=/mnt/aic-models \
docker compose up -d backend
```

Trên Windows, nên ưu tiên path tương đối trong repository trừ khi Docker Desktop đã có quyền truy cập ổ đĩa/path bên ngoài.

## Chế độ offline

Sau khi model weight cần thiết đã được cache vào thư mục model đã mount:

Linux / WSL2:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 docker compose up -d backend
```

PowerShell:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
docker compose up -d backend
```

Kiểm tra model offline:

```bash
docker compose --profile tools run --rm worker models --verify-offline
```

## Dừng / build lại

```bash
docker compose down
```

Sau khi source thay đổi:

```bash
docker compose up --build -d backend
```

---

# ⚡ NVIDIA GPU / CUDA

Repository có CUDA Compose override:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.cuda.yml \
  up --build -d backend
```

Yêu cầu host:

- **Linux:** NVIDIA driver tương thích và Docker GPU/container runtime.
- **Windows:** NVIDIA driver hỗ trợ GPU trong WSL2 và Docker Desktop dùng WSL2 backend.

Nên xác nhận Docker nhìn thấy GPU trước khi chạy CUDA override.

> ⚠️ **Ghi chú OCR cho RC2:** CUDA override vẫn có routing OCR GPU ở mức thử nghiệm. PaddleOCR GPU **chưa** thuộc runtime RC2 đã được chấp nhận. Để tái lập RC2, dùng Tesseract OCR cho đến khi CUDA/Paddle được validate riêng.

---

# 🔬 Thiết kế retrieval

### Sparse toàn cục, dense cục bộ

FAISS index lưu lâu dài vẫn ở dạng sparse. Với TRAKE, dense decoding/embedding chỉ chạy trong vùng thời gian giới hạn quanh các coarse hit có triển vọng.

### Ưu tiên bằng chứng

OCR và ASR có thể nâng hạng candidate mà visual similarity bỏ sót, ví dụ chữ, số, biển báo và nội dung lời nói.

### Thứ tự xác định được

Fusion và reranking được thiết kế để giữ thứ tự ổn định và deterministic tie-breaking khi có thể.

### Optional intelligence theo kiểu fail-open

Các thành phần query refinement tùy chọn có thể rơi về deterministic parsing thay vì làm toàn bộ retrieval path bị gián đoạn.

---

# 🛠️ Phát triển không dùng Docker

```bash
python -m pip install -e .
python projectctl.py env --check
python projectctl.py doctor
pytest
```

Chạy local server:

```bash
python projectctl.py dev
```

Mở:

```text
http://127.0.0.1:8000
```

Help của revision đang checkout là tham chiếu CLI chính xác nhất:

```bash
python projectctl.py --help
```

---

# 📁 Cấu trúc repository

```text
backend/       ứng dụng chính và retrieval pipeline
frontend/      operator UI được FastAPI phục vụ
eval/          công cụ evaluation và benchmark
scripts/       diagnostics, validation, dataset và experiment helpers
tests/         unit test và integration test
docs/          tài liệu kiến trúc, triển khai và engineering
projectctl.py  operator CLI / entry point của project
```

## Tài liệu

- [Project Control CLI](docs/projectctl.md)
- [Architecture](ARCHITECTURE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Engineering / Agent Rules](AGENTS.md)

---

# 🎯 Trạng thái hiện tại

**`1.1.0-rc2` prevalidation source**

Release candidate đang được validate trên môi trường NVIDIA GPU mục tiêu trước khi promote. Các tuyên bố về hiệu năng nên dựa trên dữ liệu competition đại diện và kết quả validation đã được ghi lại.

<div align="center">

### Xây dựng cho chất lượng retrieval, tính đúng theo thời gian và khả năng tái lập experiment.

**KIS · Q&A · TRAKE · OCR · ASR · SigLIP2 · FAISS · FastAPI**

</div>
