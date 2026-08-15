# Real Data Validation & RC Freeze Inventory

## 1. Environment Baseline

- **Git Branch**: `feature/m4-faiss-siglip`
- **Git HEAD Commit**: `90863a4bdc2ce7431ef68aff04a19f57b1061d94`
- **Python Version**: `3.12.13` (in `.venv`), Host `3.14.6`
- **Operating System**: Linux Fedora (kernel `7.1.8-200.fc44.x86_64`, x86_64)
- **CPU**: 11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz (8 logical cores)
- **Memory**: Total 7.5 GiB, Free/Available ~2.8–3.1 GiB, Swap 7.5 GiB
- **GPU Status**: `CUDA Unavailable / CPU Mode` (`torch.cuda.is_available() == False`, `nvidia-smi` not found)
- **Docker Engine**: `29.7.2, build a7dcaa6`
- **Docker Compose**: `v5.4.0`
- **Pytest Baseline**: `347 passed, 18 skipped, 0 failed`

---

## 2. Component Readiness & Frozen Architecture

- **Visual Encoder**: `google/siglip2-base-patch16-224` (768-D L2-normalized vectors, cached locally, `READY`)
- **ASR Engine**: `faster-whisper` (`small`, cached locally, `READY`)
- **OCR Engine**: Tesseract (Vietnamese `vie` + English `eng` system language packs `READY`)
- **Query Intelligence**: `QueryPlan` multi-path parsing (`DeterministicQueryParser` + local `Qwen/Qwen2.5-1.5B-Instruct` lazy adapter)
- **Reranker**: Deterministic evidence-aware `CandidateReranker` (Tiers 3/2/1/0 + multi-channel agreement + base RRF score preservation + deterministic tie-break)
- **TRAKE Coherence**: `TRAKECoherenceAnalyzer` (diagnostic mode by default, strictly preserving monotonic DP validity contract)
- **Vector Index Production**: `FAISS IndexFlatIP` (exact cosine similarity)
- **Intended Release Version**: `1.1.0-rc1`
- **Docker Repository**: `gianguyen14/aic-retrieval`

---

## 3. Real Video Asset Inventory

- Total available test videos located in `data/test-videos/`: **31 videos** (`L22_V001.mp4` through `L22_V031.mp4`).
- Previous frozen development reference: `data/processed-validation/three-video-final` (`L22_V001`, `L22_V002`, `L22_V003`, generation `gen-3d51a16ea32b4953820583c3af181b31`).
- Target real-data validation set: A representative 10-video subset (`L22_V001`–`L22_V010`) to validate full ingestion, sequential PyAV display-order frame ID stability, publication integrity, resume caching, and multi-modal query performance within host RAM limits (7.5 GiB).
