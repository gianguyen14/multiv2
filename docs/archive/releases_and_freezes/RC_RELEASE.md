# Release Candidate 1.1.0-rc1 Specification

## Release Metadata
- **Version**: `1.1.0-rc1`
- **Docker Image**: `gianguyen14/aic-retrieval:1.1.0-rc1`
- **OCI Index Digest**: `sha256:975c9f19ead207283583723a03ba84f9d302550d2e61017e1b31494314d75915`
- **Platform Manifest (linux/amd64)**: `sha256:e8ec6dd7cb113394b7bca02ab5680cd1e04488bec187b89b78a02dc0b1078e7d`
- **Git Commit**: `90863a4bdc2ce7431ef68aff04a19f57b1061d94`
- **Git Branch**: `feature/m4-faiss-siglip`
- **Release Status**: VERIFIED & PUSHED

## Core Architecture Contracts
1. **PyAV Authoritative Frame Identity**: Zero-based sequential display-order ordinal.
2. **Visual Embedding**: Google SigLIP2 (`google/siglip2-base-patch16-224`), normalized 768-D vectors.
3. **Indexing**: FAISS `IndexFlatIP` (exact cosine inner product on unit sphere).
4. **Query Refiner**: Local LLM `Qwen/Qwen2.5-1.5B-Instruct` with deterministic fallback.
5. **Reranker**: Deterministic evidence-aware multi-modal fusion with RRF ($K=60$, $w_{vis}=1.0, w_{ocr}=1.0, w_{asr}=0.8$).
6. **Temporal Coherence**: TRAKE dynamic programming with monotonic timeline ordering constraints.

## Quick Start
```bash
docker pull gianguyen14/aic-retrieval:1.1.0-rc1
docker compose -f docker-compose.release.yml up -d
```
