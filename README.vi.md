<div align="center">

# 🎬 Unified Video Retrieval

### Truy hồi video đa phương thức, ưu tiên chạy cục bộ

**Văn bản → Khung hình · Video Q&A · TRAKE · OCR · ASR** (Tìm kiếm bằng ảnh: chỉ có trên nhánh thử nghiệm)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Recommended-2496ED?logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF)
![SigLIP2](https://img.shields.io/badge/SigLIP2-768D-FF6F00)
![Status](https://img.shields.io/badge/Release-1.1.0--rc2_prevalidation-orange)

**[English](README.md) · Tiếng Việt**

*Một hệ thống truy hồi thân thiện với môi trường cục bộ/ngoại tuyến, được xây dựng để tìm đúng video, đúng khung hình và đúng chuỗi theo thời gian.*

</div>

---

## ✨ Hệ thống làm được gì?

| Chế độ | Đầu vào | Kết quả | Tín hiệu chính |
|---|---|---|---|
| 🔎 **Textual KIS** | Mô tả bằng ngôn ngữ tự nhiên | `video_id`, `frame_id` được xếp hạng | Hình ảnh Qwen3-VL + hợp nhất OCR + ASR |
| 💬 **Video Q&A** | Câu hỏi về nội dung video | Khung hình bằng chứng + `answer` (trích xuất hoặc LLM từ xa) | Bằng chứng hình ảnh + OCR + ASR |
| 🧭 **TRAKE** | Các sự kiện ngữ nghĩa có thứ tự | Một video + các khung hình chính theo thứ tự | Căn chỉnh sự kiện đơn điệu trong cùng một video |
| 🖼️ **Image Search** | Ảnh truy vấn | Các khung hình tương đồng về hình ảnh | **KHÔNG KHẢ DỤNG TRÊN `main` HIỆN TẠI** (nhánh thử nghiệm) |
| 🔤 **OCR / ASR** | Khung hình + âm thanh | Bằng chứng văn bản có thể tìm kiếm | Spool được tính trước khi ingest ngoại tuyến |

Backend tìm kiếm dùng trong production là **Qwen3-VL-Embedding-2B** (1024-d). SigLIP2 — backend mặc định trước đây — hiện là **LEGACY** và chỉ được sử dụng khi được bật một cách tường minh. Hệ thống hỗ trợ truy vấn tiếng Việt và tiếng Anh. **Image Search không khả dụng trên backend `main` hiện tại** — tính năng này chỉ tồn tại trên một nhánh thử nghiệm, còn chờ xác minh GPU, chưa được hợp nhất hoặc kích hoạt.

---

## 🧠 Kiến trúc tổng quan

```text
                               ┌─────────────────────┐
                               │ TRUY VẤN NGƯỜI DÙNG│
                               │ text / image / Q&A  │
                               │ sự kiện TRAKE có thứ│
                               │ tự                  │
                               └──────────┬──────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │ Qwen3-VL-Embedding    │
                              │ encode_query (1024-d) │
                              └───────────┬───────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
             ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
             │   FAISS     │       │     OCR     │       │     ASR     │
             │ IndexFlatIP │       │  spool JSON │       │  spool JSON │
             └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
                    │                     │                     │
                    └─────────────────────┼─────────────────────┘
                                          ▼
                                ┌───────────────────┐
                                │ Hợp nhất ứng viên │
                                │ 0.70v+0.18o+0.12a │
                                └─────────┬─────────┘
                                          │
                        ┌─────────────────┼─────────────────┐
                        │                 │                 │
                        ▼                 ▼                 ▼
                      KIS               Q&A              TRAKE
                                          │
                                     tổng hợp câu trả lời
                                     (mặc định trích xuất /
                                     `remote_llm` tùy chọn)
```

> **Backend production:** Qwen3-VL-Embedding-2B trên FAISS `IndexFlatIP` (1024-d), chỉ đọc cơ sở dữ liệu packed. SigLIP2 (`ConfiguredSearch` / RRF / Dense `TemporalRefiner`) là **legacy** và chỉ chạy khi `SEARCH_BACKEND=siglip2`.
> **Câu trả lời Video Q&A** đến từ bằng chứng OCR/ASR (trích xuất) hoặc bước `remote_llm` tùy chọn; xem `docs/QA_SYNTHESIS.md`.
> **Image Search** **không khả dụng trên `main` hiện tại** — chỉ có trên nhánh thử nghiệm và còn chờ xác minh GPU.

### `frame_id` là định danh chuẩn

Mỗi `frame_id` trả về là **chỉ số bắt đầu từ 0 được tạo ra khi PyAV giải mã tuần tự theo thứ tự hiển thị**.

```python
for frame_id, frame in enumerate(container.decode(stream)):
    ...
```

Hệ thống **không** tái tạo `frame_id` chuẩn bằng công thức `timestamp × FPS`. Quy ước này giữ cho ingest, truy hồi, đánh giá và tinh chỉnh theo thời gian cùng căn chỉnh với video nguồn.

---

## 📚 Tài liệu kỹ thuật

Thư mục `docs/` là tài liệu tham chiếu kỹ thuật chính thức:

| Chủ đề | Tài liệu |
|---|---|
| Kiến trúc hệ thống (pipeline Qwen, DB, route) | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Ma trận khả năng backend tìm kiếm (Qwen và SigLIP2) | [`docs/SEARCH_BACKENDS.md`](docs/SEARCH_BACKENDS.md) |
| Schema DB v1 / DB v2 và định danh khung hình | [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) |
| Tài liệu REST API (toàn bộ route, schema) | [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) |
| Tổng hợp câu trả lời Q&A (trích xuất + `remote_llm`) | [`docs/QA_SYNTHESIS.md`](docs/QA_SYNTHESIS.md) |
| Triển khai production (Docker, env, chạy hệ thống) | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| Tham chiếu đầy đủ các biến môi trường | [`docs/RUNTIME_CONFIG.md`](docs/RUNTIME_CONFIG.md) |
| Quickstart 5 bước | [`docs/QUICKSTART.md`](docs/QUICKSTART.md) |
| Architecture Decision Records | [`docs/ADR/`](docs/ADR/) |

Các nhật ký milestone trước đây và báo cáo freeze được lưu trong [`docs/archive/`](docs/archive/) và được giữ lại trong lịch sử Git.

---

# 🚀 Bắt đầu nhanh với Docker

Docker là cách chạy được khuyến nghị trên **Linux** và **Windows 10/11**.

Image Docker chứa các dependency Linux cốt lõi mà ứng dụng sử dụng, gồm Python 3.12, FFmpeg, Tesseract, Git, GCC và G++.

## Yêu cầu

### Linux

- Docker Engine
- Docker Compose v2 (`docker compose`)

### Windows 10/11

- Docker Desktop
- Bật WSL2 backend
- Git for Windows hoặc Git bên trong WSL2

Trên Windows, cách đơn giản nhất là đặt video, bộ nhớ đệm model và dữ liệu đã xử lý ngay trong thư mục của repository.

## 1. Clone repository

```bash
git clone git@github.com:gianguyen14/multiv2.git
cd multiv2
```

> Repository hiện ở chế độ private, vì vậy máy chạy cần được cấu hình quyền truy cập GitHub qua SSH trước.

## 2. Tạo các thư mục dữ liệu

Linux / WSL2:

```bash
mkdir -p data/test-videos data/processed models
```

PowerShell:

```powershell
New-Item -ItemType Directory -Force data/test-videos, data/processed, models
```

Đặt video nguồn vào:

```text
data/test-videos/
```

Các đường dẫn được Docker mount mặc định:

```text
Host                       Container
------------------------------------------------
./data/test-videos   -->   /data/videos       read-only
./data/processed     -->   /data/processed    read/write
./models             -->   /models            read/write
```

## 3. Build Docker image

```bash
docker compose build
```

## 4. Kiểm tra môi trường chạy

```bash
docker compose --profile tools run --rm worker env --check
docker compose --profile tools run --rm worker doctor
```

`doctor` là lệnh kiểm tra mức sẵn sàng của hệ thống trước khi xử lý dữ liệu hoặc chạy truy vấn.

## 5. Chuẩn bị model

Chuẩn bị các model hình ảnh và ASR mặc định:

```bash
docker compose --profile tools run --rm worker models --prepare
```

Kiểm tra các model hiện có:

```bash
docker compose --profile tools run --rm worker models
```

QueryRefiner cục bộ là thành phần tùy chọn:

```bash
docker compose --profile tools run --rm worker models --prepare --query-refiner
```

Nếu model của QueryRefiner chưa có sẵn, hệ thống vẫn có thể dùng đường xử lý dự phòng xác định.

## 6. Tiền xử lý và lập chỉ mục video

```bash
docker compose --profile tools run --rm worker preprocess /data/videos
```

Lệnh này chạy pipeline nhập dữ liệu theo cấu hình hiện tại và xuất các tệp phục vụ tìm kiếm vào thư mục dữ liệu đã xử lý. Những phần đã hoàn thành và còn tương thích có thể được tiếp tục thay vì chạy lại từ đầu.

Kiểm tra trạng thái sau khi xử lý:

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

Các endpoint kiểm tra tình trạng hệ thống:

```text
http://127.0.0.1:8000/health/live
http://127.0.0.1:8000/health/ready
http://127.0.0.1:8000/health
```

## Backend production Qwen3-VL (mặc định)

Backend tìm kiếm production là **Qwen3-VL-Embedding-2B** trên DB packed hiện có gồm 47.430 vector x 1024-d (FAISS generation + spool OCR/ASR), được chọn bằng `SEARCH_BACKEND=qwen3_vl` (mặc định). DB được mount và đọc trực tiếp — **không** ingest lại, re-encode, sao chép hoặc ghi lại. Đường SigLIP2 (`SEARCH_BACKEND=siglip2`) chỉ còn dưới dạng legacy khi bật rõ ràng và sẽ từ chối truy vấn một index được tạo bởi Qwen.

```bash
# .env
SEARCH_BACKEND=qwen3_vl
AIC_DATA_DIR=/home/hermes/aic/data          # thư mục host chứa aic-db-v1/
AIC_MODELS_DIR=/home/hermes/aic/models      # thư mục host chứa Qwen3-VL-Embedding-2B/
VIDEO_PROCESSED_ROOT=/data/aic-db-v1/runtime
QWEN3_VL_MODEL_DIR=/models/Qwen3-VL-Embedding-2B
```

Các khả năng gồm KIS, QA (câu trả lời dựa trên bằng chứng OCR/ASR) và truy vấn văn bản TRAKE (một video, chuỗi frame tăng dần). Image Search và thumbnail frame phụ thuộc backend/dữ liệu đang hoạt động. Packed DB không chứa JPEG nên thumbnail thiếu sẽ hiển thị placeholder. Có thể xem trước video gốc khi cấu hình nguồn video như bên dưới; video không được tự tải xuống trong đường chạy mặc định.

### Xem trước video gốc tùy chọn

Kết quả chứa `video_url` và `timestamp_seconds`. Khi operator nhấn **Play video**, FastAPI phục vụ MP4 tương ứng với hỗ trợ byte-range và frontend sẽ seek tới timestamp của kết quả sau khi metadata được tải. Cấu hình một nguồn sau đây mà không cần ingest hoặc ghi lại DB:

```bash
# File cục bộ, phục vụ trực tiếp:
VIDEO_SOURCE_DIR=/path/to/videos
# Hoặc tải lười theo video ID khi nhấn lần đầu:
VIDEO_SOURCE_URL_TEMPLATE=https://host/videos/{video_id}.mp4
# Cache cho video tải lười:
VIDEO_CACHE_DIR=/path/to/cache/videos
```

API từ chối video ID không an toàn, tải file theo cách atomic và áp dụng giới hạn 2 GiB cho mỗi file tải từ xa. Nếu không cấu hình nguồn video, tìm kiếm vẫn hoạt động nhưng nút Play sẽ bị ẩn.

`SEARCH_ENCODER` vẫn được chấp nhận như alias legacy cho `SEARCH_BACKEND`.

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

Có ba cách chính để sử dụng hệ thống:

1. **Web UI** — thuận tiện nhất khi tìm kiếm và kiểm tra kết quả trực tiếp.
2. **`projectctl.py` CLI** — phù hợp cho phát triển, chạy thử nghiệm, xử lý hàng loạt và kiểm thử.
3. **FastAPI** — phù hợp khi tích hợp với ứng dụng hoặc giao diện bên ngoài.

## Cách A — Web UI

Khởi động backend:

```bash
docker compose up -d backend
```

Sau đó mở:

```text
http://127.0.0.1:8000
```

Frontend được FastAPI phục vụ trực tiếp, vì vậy khi chạy bằng Docker thông thường không cần khởi động thêm một frontend server riêng.

---

## Cách B — CLI với `projectctl.py`

Trong Docker, các lệnh quản trị được chạy qua service `worker`:

```bash
docker compose --profile tools run --rm worker --help
```

### 🔎 Textual KIS

Tìm các khung hình phù hợp với một mô tả:

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

Truy hồi các khung hình và bằng chứng liên quan đến câu hỏi:

```bash
docker compose --profile tools run --rm worker \
  qa "Nhiệt độ hiển thị trên màn hình là bao nhiêu?" --top-k 20
```

Luồng Q&A kết hợp những bằng chứng hình ảnh, OCR và ASR có sẵn trước khi chuyển sang bước xử lý câu trả lời tiếp theo.

### 🧭 TRAKE

TRAKE dùng để tìm **một chuỗi sự kiện có thứ tự trong cùng một video**.

Cú pháp phân tách sự kiện bằng `|`:

```bash
docker compose --profile tools run --rm worker \
  trake "người đứng yên | bắt đầu chạy | nhảy lên | tiếp đất" --top-k 30
```

Hoặc truyền danh sách sự kiện bằng JSON:

```bash
docker compose --profile tools run --rm worker \
  trake '["đứng", "chạy đà", "nhảy", "tiếp đất"]' --top-k 30
```

Sau khi tìm được vùng thời gian có triển vọng, TRAKE có thể giải mã và tìm kiếm dày hơn trong vùng đó, rồi dùng căn chỉnh theo thứ tự để bảo đảm các sự kiện xuất hiện đúng trình tự thời gian.

Tắt bước tinh chỉnh dày để chẩn đoán:

```bash
docker compose --profile tools run --rm worker \
  trake "event one | event two" --no-temporal-refine
```

### 🖼️ Tìm khung hình bằng ảnh

Nếu ảnh truy vấn có sẵn bên trong container:

```bash
docker compose --profile tools run --rm worker \
  image-search /data/videos/query.jpg --top-k 20
```

Bạn cũng có thể gửi ảnh trực tiếp qua HTTP API từ máy host như ví dụ ở phần dưới.

### 🧠 Xem cách hệ thống phân tích truy vấn

```bash
docker compose --profile tools run --rm worker \
  query-plan "biển số xe 79H-6072" --task kis --json
```

Lệnh này giúp kiểm tra các biến thể truy vấn, từ khóa được trích xuất và đường xử lý QueryRefiner cục bộ.

### 🩺 Chẩn đoán hệ thống

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

Chạy quy trình kiểm tra dataset đại diện của repository:

```bash
docker compose --profile tools run --rm worker validate-dataset
```

---

## Cách C — HTTP API

Backend cung cấp API truy hồi từ `backend.app.main`.

### Truy vấn KIS

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "một người phụ nữ mặc áo dài",
    "query_type": "kis",
    "top_k": 20
  }'
```

### Truy vấn Q&A

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Nhiệt độ hiển thị là bao nhiêu?",
    "query_type": "qa",
    "top_k": 20
  }'
```

### Truy vấn TRAKE

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

### Tìm kiếm bằng ảnh

```bash
curl -X POST "http://127.0.0.1:8000/api/search/image?top_k=20" \
  -F "file=@query.jpg"
```

API hỗ trợ ảnh JPEG, PNG và WebP, với kích thước tối đa 15 MB mỗi lần gửi.

### Xem QueryPlan qua API

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

# 🧩 Quy trình sử dụng điển hình từ đầu đến cuối

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
8. Mở Web UI hoặc chạy KIS / Q&A / TRAKE / tìm bằng ảnh
           │
           ▼
9. benchmark / tinh chỉnh
```

---

# 🐳 Cấu hình Docker

## Thay đổi vị trí dữ liệu và model

Docker Compose hỗ trợ các biến môi trường sau:

```text
VIDEOS_DIR
PROCESSED_DIR
MODELS_DIR
HF_HUB_OFFLINE
TRANSFORMERS_OFFLINE
```

Ví dụ trên Linux / WSL2:

```bash
VIDEOS_DIR=/mnt/videos \
PROCESSED_DIR=/mnt/retrieval-processed \
MODELS_DIR=/mnt/retrieval-models \
docker compose up -d backend
```

Trên Windows, nên ưu tiên đường dẫn tương đối trong repository, trừ khi Docker Desktop đã được cấp quyền truy cập vào ổ đĩa hoặc thư mục bên ngoài.

## Chế độ ngoại tuyến

Sau khi các model cần thiết đã được tải vào thư mục model được mount:

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

Kiểm tra khả năng chạy model khi ngoại tuyến:

```bash
docker compose --profile tools run --rm worker models --verify-offline
```

## Dừng hoặc build lại

Dừng các service:

```bash
docker compose down
```

Sau khi thay đổi mã nguồn:

```bash
docker compose up --build -d backend
```

---

# ⚡ NVIDIA GPU / CUDA

Repository có sẵn file Docker Compose dành cho CUDA:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.cuda.yml \
  up --build -d backend
```

Yêu cầu trên máy host:

- **Linux:** NVIDIA driver tương thích và môi trường Docker có thể truy cập GPU.
- **Windows:** NVIDIA driver hỗ trợ GPU trong WSL2 và Docker Desktop sử dụng WSL2 backend.

Nên xác nhận Docker nhìn thấy GPU trước khi bật cấu hình CUDA.

> ⚠️ **Lưu ý OCR cho RC2:** file CUDA Compose hiện vẫn chứa cấu hình thử nghiệm cho OCR chạy bằng GPU. PaddleOCR GPU **chưa** nằm trong đường chạy RC2 đã được chấp nhận. Để giữ khả năng tái lập ở RC2, hãy dùng Tesseract cho OCR cho đến khi phần tích hợp CUDA/Paddle được kiểm thử riêng.

---

# 🔬 Nguyên tắc thiết kế truy hồi

### Lập chỉ mục thưa toàn cục, xử lý dày cục bộ

Chỉ mục FAISS lưu lâu dài sử dụng các khung hình được lấy mẫu. Với TRAKE, việc giải mã và tạo embedding dày hơn chỉ diễn ra trong những vùng thời gian giới hạn quanh các kết quả thô có triển vọng. Cách này giúp tăng độ chính xác theo thời gian mà không phải lập chỉ mục dày cho toàn bộ video.

### Ưu tiên bằng chứng, không chỉ dựa vào độ giống hình ảnh

OCR và ASR có thể đưa những kết quả quan trọng lên cao hơn ngay cả khi độ tương đồng hình ảnh chưa đủ mạnh, đặc biệt với chữ, số, biển báo hoặc nội dung lời nói.

### Xếp hạng ổn định và có thể tái lập

Các bước hợp nhất và xếp hạng lại được thiết kế để giữ thứ tự kết quả ổn định, đồng thời dùng quy tắc phá hòa xác định khi có thể.

### Thành phần AI tùy chọn không làm gián đoạn hệ thống

Những thành phần như QueryRefiner có thể được bật để cải thiện việc hiểu truy vấn. Nếu chúng không khả dụng, hệ thống ưu tiên quay về đường xử lý xác định thay vì làm toàn bộ chức năng truy hồi ngừng hoạt động.

---

# 🛠️ Phát triển không dùng Docker

```bash
python -m pip install -e .
python projectctl.py env --check
python projectctl.py doctor
pytest
```

Chạy server cục bộ:

```bash
python projectctl.py dev
```

Mở:

```text
http://127.0.0.1:8000
```

Để xem danh sách lệnh đúng với phiên bản mã nguồn hiện tại, dùng:

```bash
python projectctl.py --help
```

---

# 📁 Cấu trúc repository

```text
backend/       ứng dụng chính và pipeline truy hồi
frontend/      giao diện vận hành được FastAPI phục vụ trực tiếp
eval/          công cụ đánh giá và benchmark
scripts/       script chẩn đoán, kiểm thử, xử lý dataset và thử nghiệm
tests/         unit test và integration test
docs/          tài liệu kiến trúc, triển khai và ghi chú kỹ thuật
projectctl.py  CLI quản trị và điểm vào chính của project
```

## Tài liệu liên quan

- [Project Control CLI](docs/projectctl.md)
- [Architecture](ARCHITECTURE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Engineering / Agent Rules](AGENTS.md)

---

# 🎯 Trạng thái hiện tại

**`1.1.0-rc3` — mã nguồn đang ở giai đoạn kiểm thử trước khi phát hành**

Bản release candidate đang được xác minh trên môi trường NVIDIA GPU mục tiêu trước khi được đưa lên phát hành. Mọi tuyên bố về hiệu năng phải dựa trên các dataset đại diện và kết quả xác minh đã được ghi nhận.

<div align="center">

### Tập trung vào chất lượng truy hồi, tính đúng theo thời gian và khả năng tái lập kết quả.

**KIS · Q&A · TRAKE · OCR · ASR · SigLIP2 · FAISS · FastAPI**

</div>
