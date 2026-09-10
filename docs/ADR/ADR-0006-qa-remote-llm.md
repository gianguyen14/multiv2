# ADR-0006 — Grounded Q&A Remote LLM Synthesis (agent-lite)

**Status:** Accepted (partially deployed; upstream pending)  
**Date:** 2026-09-10  
**Authors:** Engineering team

## Context

Q&A queries (`query_type=qa`) previously returned the top-100 characters of the nearest OCR/ASR evidence as the `answer` field (extractive, deterministic). This produced low-quality answers when the factual answer was not directly present in frame-level OCR/ASR.

## Decision

Add an **optional, configuration-gated** answer synthesis stage using an OpenAI-compatible LLM endpoint:

- Module: `backend/app/services/qa_answer_synthesizer.py`
- Default backend: `extractive` (unchanged behaviour).
- Opt-in backend: `remote_llm` — sends the user question plus retrieved OCR/ASR evidence to a configured endpoint.
- Hard call budget per query: `QA_ANSWER_REMOTE_TOP_N` (default 5, clamped 1–10, **production target 2**).
- Rows beyond the budget keep deterministic extractive answers.
- Remote abstention (`"Không đủ bằng chứng."`) is preserved with provenance `answer_backend=remote_llm, answer_status=abstained`.
- Any network/timeout/error falls back to extractive without raising.

## Status as of 2026-09-10

The agent-lite endpoint (`http://homenasproxmox.synology.me:20128/v1`) returns HTTP 502 (upstream provider auth/model failure). This is an upstream infrastructure issue, not a code defect. The `remote_llm` backend is **EXPERIMENTAL** until upstream stability is confirmed.

## Consequences

- `QA_ANSWER_BACKEND=extractive` (default): zero change in behaviour.
- `QA_ANSWER_API_KEY` must be supplied as a runtime secret; never committed.
- `QA_ANSWER_REMOTE_TOP_N=2` is the production deployment target.
- Feature must not be documented as `production-ready` until upstream passes sustained smoke tests.
