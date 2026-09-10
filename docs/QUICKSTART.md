# Quickstart Guide — AIC 2026 Multimodal Video Retrieval

Get the AIC 2026 video retrieval system up and running in 5 minutes.

---

## Architecture at a Glance

* **Production Search Engine:** `Qwen3-VL-Embedding-2B` (1024-dimensional normalized vectors over packed FAISS `IndexFlatIP`).
* **Legacy Mode:** `SigLIP2` (768-D; explicit legacy opt-in only).
* **Operator Frontend ("Chi Lăng"):** Vanilla HTML5, ES6 JavaScript, and CSS. **Zero build steps, zero Node.js/npm dependencies.** Served directly by FastAPI or `run_dev.py`.
* **Search Modes Supported:** Known-Item Search (KIS), Video Question Answering (QA), and Temporal Event Sequences (TRAKE).
* **Image Query Search:** **EXPERIMENTAL** (unsupported on the production Qwen backend; not GPU-validated).

---

## 5-Step Quickstart

### Step 1: Clone the Repository

```bash
git clone git@github.com:gianguyen14/multiv2.git /opt/aic/app
cd /opt/aic/app
```

---

### Step 2: Configure Environment Variables

Copy the template configuration to `.env`:

```bash
cp .env.example .env
```

Ensure the key paths match your data mounts in `.env`:

```ini
# Production Search Backend
SEARCH_BACKEND=qwen3_vl

# Path to packed runtime DB (index/ with CURRENT pointer, ocr/, asr/)
VIDEO_PROCESSED_ROOT=/data/aic-db-v1/runtime

# Path to local Qwen3-VL-Embedding-2B weights
QWEN3_VL_MODEL_DIR=/models/Qwen3-VL-Embedding-2B

# Optional: Path to raw videos for seekable preview playback
VIDEO_SOURCE_DIR=/data/videos

# Question Answering: Extractive default is production safe
QA_ANSWER_BACKEND=extractive
# Note: remote_llm is EXPERIMENTAL due to upstream 502 errors.
# If enabling remote_llm, always configure QA_ANSWER_REMOTE_TOP_N=2
```

---

### Step 3: Run the System

You can run the service using **Docker Compose** or directly with **Uvicorn / `run_dev.py`**:

#### Option A: Using Docker Compose (Recommended)

```bash
# Build and run the 'aic' service (CPU mode)
docker compose up -d aic

# Or with NVIDIA GPU acceleration (GPU required):
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d aic
```

#### Option B: Using Direct Python Server

```bash
# Production server with single worker (prevents weight duplication)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Or run the development proxy (frontend at :3000, API at :8000):
python run_dev.py
```

---

### Step 4: Open Browser

Navigate to the operator web console:
* Direct FastAPI server: [http://localhost:8000](http://localhost:8000)
* Dev runner proxy: [http://localhost:3000](http://localhost:3000)

Verify readiness via the API probe:
```bash
curl -s http://localhost:8000/health/ready
# Expected: {"status": "ready", "generation_id": "gen-A-..."}
```

---

### Step 5: Submit Your First Query

You can submit queries through the web console or via `curl`:

#### 1. Known-Item Search (KIS)
Find keyframes matching a descriptive visual scene:
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Cảnh flycam toàn cảnh đồng lúa chín vàng tại miền Tây",
    "query_type": "kis",
    "top_k": 5
  }'
```

#### 2. Video Question Answering (QA)
Ask natural language questions answered by OCR/ASR evidence:
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Đồng bằng sông Hồng bao gồm bao nhiêu tỉnh thành?",
    "query_type": "qa",
    "top_k": 5
  }'
```
*Candidate results will include `answer`, `answer_backend`, `answer_model`, and `answer_status` fields alongside frame timestamps.*

#### 3. Temporal Retrieval of Actions and Key Events (TRAKE)
Search for an ordered multi-event sequence:
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      "Chiếc xe hơi màu đỏ xuất hiện ở góc ngã tư",
      "Người phụ nữ mặc áo xanh bước xuống xe",
      "Người phụ nữ đi vào sảnh toà nhà kính"
    ],
    "query_type": "trake",
    "top_k": 5
  }'
```

> [!NOTE]
> **TRAKE 400 Bad Request Behavior:** If no single video in the database contains all specified events in strictly monotonic frame order ($f_1 < f_2 < f_3$), the API returns **HTTP 400 Bad Request** (`TRAKE: no single video covers every event in increasing frame order`). This is mathematically correct behavior enforced by the sequence aligner.

---

## Next Steps & Technical References

* **API Endpoints & Schemas:** See [`docs/API_REFERENCE.md`](file:///tmp/docs-agy2/docs/API_REFERENCE.md) for complete REST specifications and HTTP 206 streaming formats.
* **QA Synthesis Pipeline:** See [`docs/QA_SYNTHESIS.md`](file:///tmp/docs-agy2/docs/QA_SYNTHESIS.md) for grounded answering details, candidate budgeting (`Top-N=2`), and abstention contracts.
* **Complete Environment Variables:** See [`docs/RUNTIME_CONFIG.md`](file:///tmp/docs-agy2/docs/RUNTIME_CONFIG.md) for every runtime flag.
* **Production Deployment:** See [`docs/DEPLOYMENT.md`](file:///tmp/docs-agy2/docs/DEPLOYMENT.md) for server directory topologies, GPU profiles, and rollback procedures.
