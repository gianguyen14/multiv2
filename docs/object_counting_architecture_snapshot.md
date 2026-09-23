# Production architecture snapshot before object counting

Baseline: `perf/qwen-adaptive-gpu-ingest` at `59700e5d046d6e96997eb6ec17da37957aaf73cf`.

- `SEARCH_BACKEND` defaults to `qwen3_vl`; `search_dispatch.py` constructs `QwenRuntimeSearch` for the default and keeps `siglip2` as the legacy alternative.
- `QwenRuntimeSearch.handle()` routes KIS and QA through `search_single()`, and TRAKE through `search_trake(events)`. Qwen retrieval embeds one text query, searches the packed FAISS index, then fuses OCR and ASR lexical evidence with fixed weights (0.70 visual, 0.18 OCR, 0.12 ASR).
- The API carries `query_refine`, but the Qwen handler does not read it. `ConfiguredSearch` does use the existing `QueryRefiner` for its legacy SigLIP path.
- `QueryPlan` already contains `visual_queries`, `lexical_terms`, `exact_strings`, `objects`, `attributes`, and `trake_stages`. New fields need defaults so cached legacy plans remain valid.
- Qwen QA answers are currently synthesized only from OCR/ASR evidence. There is no vision or detector evidence in the answer synthesizer.
- Indexed frame payloads originate from `FrameRecord.image_path`; `FrameStore` writes sampled images under `<processed_root>/<video_id>/frames/`. Qwen result rows do not currently expose an image path, so counting must resolve candidate images under that trusted root and reject paths outside it.
- Qwen ingest/index identity and the embedding contract are not part of this feature. Detection is request-time candidate processing only; no detector work belongs in the ingestion pipeline or FAISS generation identity.

## Planned implementation boundary

Use deterministic `QueryRefiner` routing in the Qwen runtime when `query_refine=true`; preserve direct-query behavior when false. A count request retrieves with Qwen first, then runs an optional detector only on the bounded top candidate frames. A temporal count reports that tracking is required and unsupported. OCR/ASR fusion, TRAKE, the Qwen embedding space, and index publication remain unchanged.
