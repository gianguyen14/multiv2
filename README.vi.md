<div align="center">

# 🎬 Unified Video Retrieval

### Hệ thống truy hồi video đa phương thức, ưu tiên vận hành cục bộ

**Văn bản → Khung hình · Video Q&A · TRAKE · Tìm bằng ảnh · OCR · ASR · Tinh chỉnh theo thời gian**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Recommended-2496ED?logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF)
![SigLIP2](https://img.shields.io/badge/SigLIP2-768D-FF6F00)
![Status](https://img.shields.io/badge/Release-1.1.0--rc2_prevalidation-orange)

**[English](README.md) · Tiếng Việt**

*Hệ thống truy hồi (retrieval) ưu tiên khả năng chạy cục bộ và ngoại tuyến, tập trung vào việc tìm đúng video, đúng khung hình và đúng chuỗi sự kiện theo thời gian.*

</div>

---

## ✨ Hệ thống làm được gì?

| Chế độ | Đầu vào | Kết quả | Tín hiệu chính |
|---|---|---|---|
| 🔎 **Textual KIS** | Mô tả bằng ngôn ngữ tự nhiên | Danh sách `video_id`, `frame_id` được xếp hạng | SigLIP2 + OCR + ASR + hợp nhất kết quả |
| 💬 **Video Q&A** | Câu hỏi về nội dung video | Khung hình bằng chứng phục vụ bước trả lời | Hình ảnh + OCR + ASR |
| 🧭 **TRAKE** | Chuỗi sự kiện có thứ tự | Một video + các khung hình đại diện theo đúng trình tự | Truy hồi thô + tinh chỉnh theo thời gian + DP alignment |
| 🖼️ **Image Search** | Ảnh truy vấn | Các khung hình có nội dung hình ảnh tương đồng | SigLIP2 image embeddings |
| 🔤 **OCR / ASR** | Khung hình + âm thanh | Văn bản có thể tìm kiếm | Tesseract + Faster Whisper |

Lớp truy hồi hỗ trợ truy vấn tiếng Việt và tiếng Anh, xếp hạng lại dựa trên bằng chứng, loại bớt kết quả trùng lặp theo thời gian và QueryRefiner cục bộ tùy chọn. Khi QueryRefiner không khả dụng, hệ thống có thể quay về bộ phân tích truy vấn xác định để tiếp tục hoạt động.

---

## 🧠 Kiến trúc tổng quan

```text
                            ┌──────────────────────┐
                            │   TRUY VẤN NGƯỜI DÙNG│
                            │ text / image / Q&A   │
                            │ chuỗi sự kiện TRAKE  │
                            └──────────┬───────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │   Phân tích truy vấn   │
                           │ VI/EN · từ khóa · LLM │
                           └───────────┬────────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 │                     │                     │
                 ▼                     ▼                     ▼
          ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
          │   SigLIP2   │       │     OCR     │       │     ASR     │
          │ ảnh / text  │       │  Tesseract  │       │   Whisper   │
          └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
                 │                     │                     │
                 └─────────────────────┼─────────────────────┘
                                       ▼
                             ┌───────────────────┐
                             │ Hợp nhất ứng viên │
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
                                            Căn chỉnh thứ tự bằng DP
```

### `frame_id` là định danh chuẩn

Mỗi `frame_id` trả về là **chỉ số bắt đầu từ 0 theo đúng thứ tự khung hình mà PyAV giải mã tuần tự để hiển thị**.

```python
for frame_id, frame in enumerate(container.decode(stream)):
    ...
```

Hệ thống **không** suy ra `frame_id` chuẩn bằng công thức `timestamp × FPS`. Quy ước này giúp các bước nhập dữ liệu, truy hồi, đánh giá và tinh chỉnh theo thời gian cùng dùng một hệ quy chiếu khung hình thống nhất.

---

# 🚀 Bắt đầu nhanh với Docker

Docker là cách chạy được khuyến nghị trên **Linux** và **Windows 10/11**.

Docker image đã tích hợp các thành phần hệ thống cần thiết cho ứng dụng, gồm Python 3.12, FFmpeg, Tesseract, Git, GCC và G++.

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

**`1.1.0-rc2` — mã nguồn đang ở giai đoạn kiểm thử trước khi phát hành**

RC2 hiện đang được kiểm thử trên môi trường NVIDIA GPU mục tiêu trước khi phát hành. Mọi tuyên bố về hiệu năng nên dựa trên dữ liệu đại diện và kết quả kiểm thử đã được ghi nhận.

<div align="center">

### Tập trung vào chất lượng truy hồi, tính đúng theo thời gian và khả năng tái lập kết quả.

**KIS · Q&A · TRAKE · OCR · ASR · SigLIP2 · FAISS · FastAPI**

</div>
