# AIC 2026 — Full Network/LAN Usage Guide

Hướng dẫn này chạy hệ thống trên một server Linux và truy cập từ laptop/máy khác trong cùng LAN.

Các cổng sử dụng:

- Backend/API: `0.0.0.0:8000`
- Frontend console: `0.0.0.0:3000`
- Truy cập từ máy khác bằng IP LAN của server, không dùng `0.0.0.0` trong URL.

## 1. Command thật của repository

Đã kiểm tra `projectctl.py` hiện tại. Các command liên quan là:

```text
python3 projectctl.py backend --host HOST --port PORT
python3 projectctl.py frontend --host HOST --port PORT
python3 projectctl.py dev --host HOST --port PORT
```

`backend`, `frontend`, và `dev` đều gọi Uvicorn cho `backend.app.main:app`. Repository hiện tại không có command `start`, `serve`, `stop`, hoặc `restart`, và `dev` không khởi động cả frontend lẫn backend.

`run_dev.py` hiện hard-code cả frontend và backend ở `127.0.0.1`; không dùng file này để expose LAN.

Kiểm tra lại bất cứ lúc nào:

```bash
python3 projectctl.py --help
python3 projectctl.py backend --help
python3 projectctl.py frontend --help
python3 projectctl.py dev --help
```

## 2. Clone hoặc cập nhật source

Trên server:

```bash
git clone https://github.com/gianguyen14/multiv2.git /home/hermes/aic
cd /home/hermes/aic
```

Nếu source đã tồn tại:

```bash
cd /home/hermes/aic
git fetch origin
git checkout main
git pull --ff-only origin main
```

Không chạy `git reset --hard` trên server có thay đổi cục bộ.

## 3. Kiểm tra hardware và profile finals

Chạy trước khi chọn dtype/model runtime:

```bash
cd /home/hermes/aic
bash ops/qualify_finals_hardware.sh
```

Quy tắc chọn profile:

1. `CUDA`: chỉ chọn sau khi GPU NVIDIA, CUDA, VRAM, embedding compatibility, no-OOM và full smoke đã PASS.
2. `FAST_CPU_FP32`: chỉ chọn trên CPU có RAM ít nhất 16 GiB, sau khi full runtime startup vẫn còn ít nhất 2 GiB headroom, không có swap pressure, và smoke finals PASS.
3. `SAFE_CPU_LOW_RAM`: BF16 fallback cho CPU/RAM thấp. Đây là profile an toàn đã được xác nhận trên host 9 GiB hiện tại.

RAM size đơn độc không đủ để tự động bật FP32.

Sau khi profile được chọn, chạy:

```bash
cd /home/hermes/aic
bash ops/finals_smoke.sh
```

Script smoke chỉ đọc health/API/video; không rebuild DB, không đổi model, không đổi config.

## 4. Environment production hiện tại

Các đường dẫn dưới đây là layout production hiện tại trên host này. Nếu clone sang vị trí khác, thay `/home/hermes/aic` tương ứng; không thay đổi tên model hoặc DB generation tùy ý.

Tạo environment trong mỗi terminal chạy service:

```bash
cd /home/hermes/aic

export PYTHONPATH=/home/hermes/aic/.runtime/site-packages:/home/hermes/aic
export VIDEO_PROCESSED_ROOT=/home/hermes/aic/data/aic-db-v1/runtime
export MODEL_CACHE_DIR=/home/hermes/aic/models
export QWEN3_VL_MODEL_DIR=/home/hermes/aic/models/Qwen3-VL-Embedding-2B
export SEARCH_BACKEND=qwen3_vl
export VIDEO_SOURCE_DIR=/video/video
export VIDEO_CACHE_DIR=/home/hermes/aic/cache/videos

# Production-safe defaults
export QA_ANSWER_BACKEND=extractive
export QUERY_REFINER_ENABLED=false
export SEARCH_ENABLE_OCR=true
export SEARCH_ENABLE_ASR=true
export RERANKER_ENABLED=true

# Explicit LAN origin. Replace SERVER_IP with the actual server IPv4.
export ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000,http://SERVER_IP:3000"
```

Ví dụ khi server có IP `192.168.1.50`:

```bash
export ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000,http://192.168.1.50:3000"
```

Không đặt `ALLOWED_ORIGINS=*` khi không cần. Không bật `remote_llm` trong production mặc định.

## 5. Start full system trên `0.0.0.0`

### Cách dùng đúng với các command hiện tại

Repository không có một command hợp nhất để chạy hai cổng. Dùng hai terminal.

### TERMINAL 1 — BACKEND/API `0.0.0.0:8000`

```bash
cd /home/hermes/aic
export PYTHONPATH=/home/hermes/aic/.runtime/site-packages:/home/hermes/aic
export VIDEO_PROCESSED_ROOT=/home/hermes/aic/data/aic-db-v1/runtime
export MODEL_CACHE_DIR=/home/hermes/aic/models
export QWEN3_VL_MODEL_DIR=/home/hermes/aic/models/Qwen3-VL-Embedding-2B
export SEARCH_BACKEND=qwen3_vl
export VIDEO_SOURCE_DIR=/video/video
export VIDEO_CACHE_DIR=/home/hermes/aic/cache/videos
export QA_ANSWER_BACKEND=extractive
export QUERY_REFINER_ENABLED=false
export SEARCH_ENABLE_OCR=true
export SEARCH_ENABLE_ASR=true
export RERANKER_ENABLED=true
export ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000,http://SERVER_IP:3000"

python3 projectctl.py backend --host 0.0.0.0 --port 8000
```

Thay `SERVER_IP` bằng IP LAN thật nếu dùng explicit origin.

### TERMINAL 2 — FRONTEND CONSOLE `0.0.0.0:3000`

Dùng cùng environment block như Terminal 1, sau đó chạy:

```bash
cd /home/hermes/aic
export PYTHONPATH=/home/hermes/aic/.runtime/site-packages:/home/hermes/aic
export VIDEO_PROCESSED_ROOT=/home/hermes/aic/data/aic-db-v1/runtime
export MODEL_CACHE_DIR=/home/hermes/aic/models
export QWEN3_VL_MODEL_DIR=/home/hermes/aic/models/Qwen3-VL-Embedding-2B
export SEARCH_BACKEND=qwen3_vl
export VIDEO_SOURCE_DIR=/video/video
export VIDEO_CACHE_DIR=/home/hermes/aic/cache/videos
export QA_ANSWER_BACKEND=extractive
export QUERY_REFINER_ENABLED=false
export SEARCH_ENABLE_OCR=true
export SEARCH_ENABLE_ASR=true
export RERANKER_ENABLED=true
export ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000,http://SERVER_IP:3000"

python3 projectctl.py frontend --host 0.0.0.0 --port 3000
```

Lưu ý quan trọng: theo code hiện tại, `projectctl.py frontend` không phải static-file server và không proxy `/api/*` sang port 8000; nó khởi động thêm một FastAPI/Uvicorn instance trên port 3000. Vì vậy UI trên port 3000 gọi API tương đối tại chính port 3000. Cách này có thể load thêm model Qwen lần thứ hai và không phù hợp với host 9 GiB.

Với host 9 GiB hiện tại, chế độ an toàn về bộ nhớ là dùng một backend duy nhất và mở UI tại port 8000:

```bash
python3 projectctl.py backend --host 0.0.0.0 --port 8000
```

Khi đó mở `http://SERVER_IP:8000`. Không chạy thêm `projectctl.py frontend` trên host 9 GiB nếu không chấp nhận việc khởi động thêm một full FastAPI/model instance.

Nếu yêu cầu bắt buộc phải có URL frontend port 3000 mà chỉ được phép một model instance, cần đặt reverse proxy bên ngoài repository (Nginx/Caddy/HAProxy) để forward port 3000 tới frontend/API process phù hợp; `projectctl.py` hiện không cung cấp upstream-proxy option.

## 6. `0.0.0.0` nghĩa là gì

`0.0.0.0` chỉ là bind address, nghĩa là service lắng nghe trên mọi interface IPv4.

Không mở bằng:

```text
http://0.0.0.0:3000
http://0.0.0.0:8000
```

Từ máy khác, dùng IP LAN thật của server:

```text
Frontend: http://SERVER_IP:3000
Backend:  http://SERVER_IP:8000
Health:   http://SERVER_IP:8000/health
```

Ví dụ server có IP `192.168.1.50`:

```text
Frontend: http://192.168.1.50:3000
Backend:  http://192.168.1.50:8000
Health:   http://192.168.1.50:8000/health
```

## 7. Tìm IP server

Trên server:

```bash
hostname -I
ip addr
```

Chọn IPv4 của interface LAN, thường có dạng:

- `192.168.x.x`
- `10.x.x.x`
- `172.16.x.x` đến `172.31.x.x`

Không chọn `127.0.0.1`, địa chỉ Docker nội bộ, hoặc IPv6 nếu laptop đang dùng IPv4.

## 8. Kiểm tra port listen

Trên server:

```bash
ss -lntp | grep -E ':3000|:8000'
```

Khi chạy đúng LAN bind, cần thấy dạng tương tự:

```text
LISTEN ... 0.0.0.0:3000 ...
LISTEN ... 0.0.0.0:8000 ...
```

Nếu chỉ thấy `127.0.0.1:3000` hoặc `127.0.0.1:8000`, máy khác sẽ không truy cập được.

## 9. Health check

Trên server:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://SERVER_IP:8000/health
curl -I http://SERVER_IP:3000/
```

Expected:

- `/health`: HTTP 200, search configured/initialized và production generation hiện diện.
- `http://SERVER_IP:8000/health`: HTTP 200 từ interface LAN.
- `curl -I http://SERVER_IP:3000/`: HTTP 200 nếu frontend process port 3000 đang chạy.

Readiness đầy đủ:

```bash
curl -sS http://SERVER_IP:8000/health/ready
```

## 10. Firewall LAN

Kiểm tra trước, không tự chạy firewall command:

```bash
sudo ufw status
```

Nếu UFW đang active và LAN là `192.168.1.0/24`, giới hạn rule theo subnet:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 3000 proto tcp
sudo ufw allow from 192.168.1.0/24 to any port 8000 proto tcp
```

Nếu chưa biết subnet và chấp nhận mở cho mọi source trên interface firewall:

```bash
sudo ufw allow 3000/tcp
sudo ufw allow 8000/tcp
```

Không expose trực tiếp hai port này ra Internet công cộng nếu chưa có authentication, TLS/reverse proxy, rate limiting và firewall policy phù hợp. Với LAN, ưu tiên rule giới hạn subnet.

## 11. Sử dụng frontend

Mở trên laptop:

```text
http://SERVER_IP:3000
```

Nếu đang dùng chế độ một process an toàn trên port 8000:

```text
http://SERVER_IP:8000
```

Frontend hiện là HTML/CSS/JavaScript thuần, không cần Node.js, npm, bundler, hoặc build step.

### KIS

1. Chọn `Textual KIS`.
2. Nhập mô tả tiếng Việt tự nhiên, ví dụ:

   ```text
   một người phụ nữ đang nấu ăn trong chảo
   ```

3. Bấm Search.
4. Đọc các result cards, score, video ID, frame ID và timestamp.
5. Bấm mở video để xem raw video.
6. Player seek tới timestamp backend trả về.

KIS là ranked retrieval; không phải answer synthesis.

### QA

1. Chọn `Q&A`.
2. Nhập câu hỏi, ví dụ:

   ```text
   Hội chợ game năm nay quy tụ bao nhiêu hãng game?
   ```

3. Bấm Search.
4. Đọc answer và các frame/evidence đi kèm.

QA mặc định hiện tại là extractive:

```text
QA_ANSWER_BACKEND=extractive
```

Nó lấy câu trả lời từ evidence OCR/ASR đã lưu; không gọi remote generative LLM trong default production. `answer_backend=extractive`, `answer_model=none`, và `answer_status` phản ánh trạng thái evidence/answer hiện tại.

### TRAKE

1. Chọn `TRAKE`.
2. Nhập Event 1.
3. Bấm thêm event để nhập Event 2, Event 3, v.v.
4. Thứ tự event nhập vào được giữ nguyên.
5. Bấm Search.

TRAKE tìm một video duy nhất trong đó các event xuất hiện theo thứ tự frame tăng dần. Nếu không có video thỏa mãn đồng thời các event và thứ tự thời gian, backend có thể trả HTTP 400 với thông báo no-match. Đây là semantic no-match, không nên nới lỏng monotonic constraint chỉ để làm smoke pass.

Frontend giới hạn tối đa 20 event; mỗi event phải có text.

### OCR/ASR

Production search đọc evidence đã pre-extract trong DB runtime:

- OCR: text trên frame
- ASR: transcript audio

Weighted fusion hiện tại:

```text
visual = 0.70
OCR    = 0.18
ASR    = 0.12
```

Các giá trị component được hiển thị trong result khi backend trả về. Search không chạy OCR/ASR live trên mỗi query; nó lookup/scoring các spool JSON đã có.

### Frame preview và video preview

Packed production DB hiện không chứa JPEG frame preview. Vì vậy UI có thể hiện:

```text
Frame image unavailable
```

Đây không đồng nghĩa video retrieval thất bại. Fallback chính là:

1. mở raw video;
2. seek tới `timestamp_seconds` backend trả về;
3. dùng `source_frame_index_zero_based`/`frame_id` làm frame identity từ backend.

Không reconstruct frame identity bằng `timestamp * FPS`.

## 12. AI flow hiện tại

### KIS

```text
query text
  -> Qwen3-VL text embedding
  -> FAISS visual search
  -> OCR/ASR evidence lookup and scoring
  -> weighted fusion
  -> ranked results
```

### QA

```text
query text
  -> cùng retrieval flow như KIS
  -> extractive answer handling từ evidence
```

Mặc định:

```text
QA_GENERATIVE_LLM = OFF
QUERY_REFINER = OFF
```

Frontend không gọi LLM trực tiếp; frontend chỉ POST JSON tới `/api/search` và render response.

### TRAKE

```text
multiple event texts
  -> embedding từng event
  -> retrieval cho từng event
  -> candidate grouping theo video
  -> chọn sequence cùng video với frame IDs tăng dần
```

## 13. API smoke thủ công

KIS:

```bash
curl -sS -X POST http://SERVER_IP:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"một người phụ nữ đang nấu ăn trong chảo","query_type":"kis","top_k":3}'
```

QA:

```bash
curl -sS -X POST http://SERVER_IP:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Hội chợ game năm nay quy tụ bao nhiêu hãng game?","query_type":"qa","top_k":3}'
```

TRAKE known-good sequence:

```bash
curl -sS -X POST http://SERVER_IP:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query_type":"trake","events":["cháy rừng bạch đàn","diện tích rừng bị cháy"],"top_k":10}'
```

Video preview:

```bash
curl -I http://SERVER_IP:8000/api/video/L21_V013
curl -sS -D - -o /dev/null -r 0-1023 http://SERVER_IP:8000/api/video/L21_V013
```

Expected video responses:

- full request: HTTP 200
- byte range: HTTP 206 and `Content-Range`

## 14. Start / stop / restart / status

### Start

Không có `projectctl.py start`. Dùng đúng hai command foreground:

```bash
python3 projectctl.py backend --host 0.0.0.0 --port 8000
python3 projectctl.py frontend --host 0.0.0.0 --port 3000
```

### Stop an toàn

Ưu tiên quay lại từng terminal và nhấn `Ctrl+C`. Đây là cách an toàn nhất khi chạy foreground.

Nếu cần tìm process:

```bash
pgrep -af 'projectctl.py (backend|frontend)|uvicorn backend.app.main'
```

Gửi `SIGTERM` cho PID supervisor đúng service:

```bash
kill -TERM PID
```

Không dùng `kill -9` trừ khi process không kết thúc sau khi đã thử SIGTERM và đã xác định đúng PID.

### Status

`projectctl.py status` là diagnostics của project, không phải process manager:

```bash
python3 projectctl.py status
ss -lntp | grep -E ':3000|:8000'
pgrep -af 'projectctl.py (backend|frontend)|uvicorn backend.app.main'
```

### Health

```bash
curl -sS http://SERVER_IP:8000/health
curl -sS http://SERVER_IP:8000/health/ready
```

### Restart

Không có `projectctl.py restart`. Restart an toàn:

1. `Ctrl+C` ở terminal backend.
2. `Ctrl+C` ở terminal frontend.
3. Xác nhận không còn listener bằng `ss`.
4. Chạy lại chính xác các command START ở trên.

Không khởi động instance thứ hai khi port cũ vẫn đang LISTEN.

## 15. Finals smoke

Sau khi hai service đã healthy:

```bash
cd /home/hermes/aic
BACKEND_URL=http://127.0.0.1:8000 FRONTEND_URL=http://127.0.0.1:3000 bash ops/finals_smoke.sh
```

Output chỉ gồm PASS/FAIL và latency cho:

- backend health
- frontend HTTP
- KIS
- QA
- TRAKE
- video preview
- byte-range video

## 16. Troubleshooting

| Problem | Check | Fix |
|---|---|---|
| Frontend không mở từ máy khác | `ss -lntp`, IP server, firewall | Bind `--host 0.0.0.0`; dùng `http://SERVER_IP:3000`; mở rule LAN cho port 3000 |
| Backend health fail | `curl http://127.0.0.1:8000/health`, terminal log | Sửa env/path; kiểm tra `VIDEO_PROCESSED_ROOT`, model, `index/CURRENT`; không spawn duplicate |
| Port chỉ bind `127.0.0.1` | `ss -lntp \| grep -E ':3000|:8000'` | Dùng `--host 0.0.0.0`, không dùng `run_dev.py` hiện tại |
| CORS error | Browser origin và `ALLOWED_ORIGINS` | Thêm chính xác `http://SERVER_IP:3000`, không dùng wildcard không cần thiết |
| Port already in use | `ss -lntp \| grep ':8000'` hoặc `':3000'` | Dừng đúng supervisor bằng `Ctrl+C`/SIGTERM; không dùng port khác tùy ý nếu tài liệu đang trỏ port chuẩn |
| Model not found | `test -f /home/hermes/aic/models/Qwen3-VL-Embedding-2B/model.safetensors` | Đặt `QWEN3_VL_MODEL_DIR` đúng local model directory; không tự đổi model |
| DB not ready | `curl /health/ready`; kiểm tra `VIDEO_PROCESSED_ROOT/index/CURRENT` | Mount đúng DB v1 runtime read-only; kiểm tra generation và mapping/payload |
| Video preview 404 | `curl -I http://SERVER_IP:8000/api/video/VIDEO_ID` | Kiểm tra `VIDEO_SOURCE_DIR` và raw MP4 tồn tại; packed DB index không chứa raw video |
| Frame JPEG 404 | Result/UI báo `Frame image unavailable` | Đây là giới hạn packed DB hiện tại; dùng video preview + timestamp seek; không repack DB chỉ để sửa UI |
| TRAKE HTTP 400/no match | Đọc response detail; thử known-good sequence | Đây có thể là no-match hợp lệ do monotonic same-video constraint; không nới lỏng constraint |
| RAM thiếu | `free -h`, `ps -o rss`, swap | Host <16 GiB dùng BF16; tránh chạy hai full FastAPI/Qwen instances; chạy profile qualification |
| Query rất chậm | `ps`, `/health`, latency smoke | CPU Qwen embedding là bottleneck; không đổi model/precision trong production nếu chưa qualification |
| Frontend port 3000 chạy nhưng UI không dùng backend 8000 | Kiểm tra `projectctl.py frontend` semantics | Current command starts a second FastAPI app at 3000; dùng one-process UI at 8000 hoặc deploy reverse proxy externally |

## 17. Docker deployment

The release Docker architecture uses one FastAPI/Qwen process listening on internal `0.0.0.0:8000`. Both published host ports map to that same internal port:

```text
host 3000 -> container 8000
host 8000 -> container 8000
```

Therefore `http://SERVER_IP:3000` and `http://SERVER_IP:8000` reach the same application instance. `0.0.0.0` is a bind address, not a browser URL. The DB, model, videos, and cache are external mounts; they are not copied into the image.

CPU:

```bash
cp ops/docker/.env.cpu.example .env.cpu
# Replace SERVER_IP in .env.cpu and review paths.
docker compose --env-file .env.cpu -f docker-compose.cpu.yml pull
docker compose --env-file .env.cpu -f docker-compose.cpu.yml up -d
docker compose --env-file .env.cpu -f docker-compose.cpu.yml ps
curl -fsS http://127.0.0.1:8000/health/ready
curl -I http://127.0.0.1:3000/
BACKEND_URL=http://127.0.0.1:8000 FRONTEND_URL=http://127.0.0.1:3000 bash ops/finals_smoke.sh
```

Build instead of pull:

```bash
docker build --pull -t gianguyen14/aic-retrieval:sha-$(git rev-parse --short=12 HEAD) .
```

GPU qualification:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
cp ops/docker/.env.gpu.example .env.gpu
# Replace SERVER_IP and review paths.
docker compose --env-file .env.gpu -f docker-compose.gpu.yml up -d
docker exec aic-retrieval-gpu python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print(torch.version.cuda)'
```

Read `ops/DOCKER_CPU_GUIDE.md` and `ops/DOCKER_GPU_GUIDE.md` before selecting a finals profile. Never bake model/DB/videos/secrets into the image.

- Chỉ mở port cho LAN subnet khi có thể.
- Không đưa API token vào shell history, Git remote URL, `.env` commit, hoặc log.
- Không expose port 8000/3000 trực tiếp ra Internet.
- Không chạy nhiều Uvicorn workers: Qwen weights có thể bị nhân bản trong RAM.
- DB/model nên mount read-only trong production.
- `QA_ANSWER_BACKEND=extractive` là default an toàn; remote LLM là experimental và cần cấu hình riêng.

## Quick reference

```text
Server backend:  http://0.0.0.0:8000  (bind only; not a browser URL)
Server frontend: http://0.0.0.0:3000  (bind only; not a browser URL)
Remote frontend: http://SERVER_IP:3000
Remote backend:  http://SERVER_IP:8000
Health:          http://SERVER_IP:8000/health
```
