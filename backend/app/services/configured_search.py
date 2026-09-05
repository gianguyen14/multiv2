import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from backend.app.core.config import (
    DEBUG_QUERY_PLAN,
    QUERY_REFINER_ENABLED,
    QUERY_REFINER_RRF_K,
    RERANKER_ENABLED,
)
from backend.app.embeddings.siglip2 import SigLIP2Encoder
from backend.app.retrieval.video_multimodal import lexical_score
from backend.app.services.candidate_reranker import CandidateReranker
from backend.app.services.query_refiner import (
    QueryPlan,
    QueryRefiner,
    VisualQuery,
)
from backend.app.video.frame_index import load_current_frame_index
from backend.app.video.m16_text_pipeline import TextEvidenceStore
from backend.app.video.text_evidence import normalize_text


def _read_modality_weight(name: str, default: float) -> float:
    """Read a fusion weight without allowing invalid values into ranking math."""
    raw_value = os.getenv(name)
    value_source = str(default) if raw_value is None else raw_value
    try:
        value = float(value_source)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def _validated_modality_weight(name: str, value: float) -> float:
    """Validate plan-provided weights with the same contract as environment weights."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return numeric_value


def _match_exact_term(term: str, raw_text: str, norm_text: str) -> bool:
    """Non-destructive exact matching with support for license plate / code separator variations."""
    if not term or not raw_text:
        return False
    # 1. Exact raw or lower substring match
    if term in raw_text or term.lower() in raw_text.lower():
        return True
    norm_term = normalize_text(term)
    if norm_term and norm_term in norm_text:
        return True
    # 2. Separator variants (e.g. 50H-052.03 vs 50H 052.03 vs 50h05203)
    compact_term = re.sub(r"[\s.-]", "", norm_term)
    if len(compact_term) >= 4:
        compact_norm = re.sub(r"[\s.-]", "", norm_text)
        if compact_term in compact_norm:
            return True
    return False


class ConfiguredSearch:
    def __init__(self, processed_root=None, device=None, encoder_factory=None):
        value = processed_root or os.getenv("VIDEO_PROCESSED_ROOT")
        self.processed_root = Path(value) if value else None
        from backend.app.runtime.device_policy import resolve_device
        self.device_selection = resolve_device("visual", "torch", device,
            component_env="VISUAL_DEVICE", compatibility_env="SEARCH_MODEL_DEVICE")
        self.device = self.device_selection.device
        self.requires_local_model = encoder_factory is None
        self.encoder_factory = encoder_factory or (lambda: SigLIP2Encoder(
            device=self.device_selection.requested, force_download=False, local_files_only=True))
        self.enable_ocr = os.getenv("SEARCH_ENABLE_OCR", "true").lower() == "true"
        self.enable_asr = os.getenv("SEARCH_ENABLE_ASR", "true").lower() == "true"
        self.enable_query_refine = os.getenv("QUERY_REFINER_ENABLED", "true").lower() in ("1", "true", "yes")
        self.enable_rerank = os.getenv("RERANKER_ENABLED", "true").lower() in ("1", "true", "yes")
        self._bundle = None
        self._encoder = None
        self._ocr = []
        self._asr = []
        self._query_refiner = None
        self._candidate_reranker = CandidateReranker(enabled=self.enable_rerank)
        self._lock = threading.Lock()
        self._validated_generation_id = None
        self.last_query_plan = None
        self.last_query_metrics = {}

    def _get_query_refiner(self) -> QueryRefiner:
        if self._query_refiner is None:
            cache_dir = (self.processed_root / "query_refine_cache") if self.processed_root else None
            self._query_refiner = QueryRefiner(cache_dir=cache_dir)
        return self._query_refiner

    @property
    def configured(self):
        return self.processed_root is not None

    def _guard_index_backend(self, bundle) -> None:
        """Refuse to run the SigLIP2 encoder against a Qwen3-VL index.

        A Qwen-built generation is schema-compatible with the shared loader, so
        without this guard a misconfigured SigLIP2 deployment would query a Qwen
        DB with the wrong embedding space and produce meaningless scores.
        """
        from backend.app.services.qwen_runtime_search import QWEN_BACKEND_NAMES

        backend = str((bundle.metadata.get("encoder_identity") or {}).get("backend", "")).lower()
        if backend in QWEN_BACKEND_NAMES:
            raise RuntimeError(
                "active generation was built with Qwen3-VL embeddings "
                f"(backend={backend!r}); set SEARCH_BACKEND=qwen3_vl "
                "instead of using the siglip2 backend"
            )

    def _initialize(self):
        if self._bundle is not None:
            return
        if not self.configured:
            raise RuntimeError("VIDEO_PROCESSED_ROOT is not configured")
        with self._lock:
            if self._bundle is None:
                bundle = load_current_frame_index(self.processed_root / "index")
                self._guard_index_backend(bundle)
                encoder = self.encoder_factory()
                ocr, asr = [], []
                store = TextEvidenceStore(self.processed_root)
                for video_id in bundle.metadata.get("video_ids", []):
                    if self.enable_ocr and store._path(video_id, "ocr.json").is_file():
                        ocr.extend(store.load_ocr(video_id))
                    if self.enable_asr and store._path(video_id, "asr.json").is_file():
                        asr.extend(store.load_asr(video_id))
                self._bundle, self._encoder, self._ocr, self._asr = bundle, encoder, ocr, asr

    def status(self):
        return {"configured": self.configured, "initialized": self._bundle is not None,
            "visual_device": self.device, "visual_device_requested": self.device_selection.requested,
            "visual_device_source": self.device_selection.source,
            "visual_device_fallback": self.device_selection.fallback,
            "capabilities": {
                "kis": True,
                "qa": True,
                "trake": True,
                "image": True,
                "thumbnails": True,
                "raw_video_preview": True,
            }}

    def readiness(self):
        if not self.configured:
            return {"ready": False, "reason": "VIDEO_PROCESSED_ROOT is not configured"}
        try:
            from backend.app.video.frame_index import (
                current_generation_id,
                validate_generation,
            )
            generation_id = current_generation_id(self.processed_root / "index")
            if not generation_id:
                return {"ready": False, "reason": "CURRENT index generation is missing"}
            if generation_id != self._validated_generation_id:
                bundle = validate_generation(
                    self.processed_root / "index" / "generations" / generation_id,
                    generation_id,
                )
                self._guard_index_backend(bundle)
                self._validated_generation_id = generation_id
            if self.requires_local_model:
                from backend.app.model_cache import visual_status
                if not visual_status()["cached"]:
                    return {"ready": False, "reason": "SigLIP2 model is not available locally"}
            return {"ready": True, "generation_id": generation_id}
        except Exception as exc:
            if "SEARCH_BACKEND" in str(exc) or "Qwen3-VL embeddings" in str(exc):
                return {"ready": False, "reason": str(exc)}
            return {"ready": False, "reason": f"invalid search artifacts: {type(exc).__name__}"}

    def _search_single_query(self, query, top_k=100):
        self._initialize()
        vw = _read_modality_weight("VISUAL_WEIGHT", 1.0)
        ow = _read_modality_weight("OCR_WEIGHT", 1.0) if self.enable_ocr else 0.0
        aw = _read_modality_weight("ASR_WEIGHT", 1.0) if self.enable_asr else 0.0

        vector = self._encoder.encode_text([query])[0]
        hits = self._bundle.index.search(vector, max(top_k * 2, 200))

        candidates = {}
        vis_scores_map = {}
        for h in hits:
            payload = self._bundle.resolver.resolve(h["frame_id"])
            key = (payload["video_id"], payload["source_frame_index_zero_based"])
            candidates[key] = payload
            vis_scores_map[key] = float(h["score"])

        # Multimodal candidate expansion (OCR)
        if ow > 0:
            for o in self._ocr:
                s = lexical_score(query, o.normalized_text)
                if s >= 0.15:
                    key = (o.video_id, o.source_frame_index_zero_based)
                    if key not in candidates:
                        frame_uid = f"{o.video_id}:{str(o.source_frame_index_zero_based).zfill(9)}"
                        candidates[key] = {
                            "video_id": o.video_id,
                            "frame_id": o.source_frame_index_zero_based,
                            "source_frame_index_zero_based": o.source_frame_index_zero_based,
                            "submission_frame_id": o.source_frame_index_zero_based,
                            "frame_uid": frame_uid,
                            "timestamp_seconds": o.timestamp_seconds,
                            "image_path": f"frames/{str(o.source_frame_index_zero_based).zfill(9)}.jpg"
                        }
                        vis_scores_map[key] = 0.0

        # Multimodal candidate expansion (ASR)
        if aw > 0:
            for a in self._asr:
                s = lexical_score(query, a.normalized_text)
                if s >= 0.15 and a.start_frame is not None:
                    key = (a.video_id, a.start_frame)
                    if key not in candidates:
                        frame_uid = f"{a.video_id}:{str(a.start_frame).zfill(9)}"
                        candidates[key] = {
                            "video_id": a.video_id,
                            "frame_id": a.start_frame,
                            "source_frame_index_zero_based": a.start_frame,
                            "submission_frame_id": a.start_frame,
                            "frame_uid": frame_uid,
                            "timestamp_seconds": a.start_seconds,
                            "image_path": f"frames/{str(a.start_frame).zfill(9)}.jpg"
                        }
                        vis_scores_map[key] = 0.0

        if not candidates:
            return []

        vis_scores = list(vis_scores_map.values())
        v_min, v_max = min(vis_scores), max(vis_scores)
        v_rng = max(v_max - v_min, 1e-9)

        results = []
        for key, payload in candidates.items():
            vid, fid = key
            v_raw = vis_scores_map.get(key, 0.0)
            v_norm = (v_raw - v_min) / v_rng if v_raw > 0 else 0.0

            ocr_texts = [
                item.normalized_text for item in self._ocr
                if item.video_id == vid and item.source_frame_index_zero_based == fid
            ]
            asr_texts = [
                item.normalized_text for item in self._asr
                if item.video_id == vid and item.start_frame is not None
                and item.start_frame <= fid <= (item.end_frame or item.start_frame)
            ]

            ocr_s = lexical_score(query, " ".join(ocr_texts)) if ow > 0 else 0.0
            asr_s = lexical_score(query, " ".join(asr_texts)) if aw > 0 else 0.0

            text_boost = ow * ocr_s * 1.5 + aw * asr_s * 1.5
            fused = vw * v_norm + text_boost

            filename = Path(payload.get("image_path", f"{str(fid).zfill(9)}.jpg")).name
            results.append({
                "video_id": vid,
                "frame_id": payload.get("submission_frame_id", fid),
                "source_frame_index_zero_based": fid,
                "frame_uid": payload.get("frame_uid", f"{vid}:{str(fid).zfill(9)}"),
                "timestamp_seconds": payload.get("timestamp_seconds"),
                "visual_score": v_raw,
                "ocr_score": ocr_s,
                "asr_score": asr_s,
                "score": fused,
                "image_url": f"/api/frames/{vid}/{filename}"
            })

        raw_sorted = sorted(results, key=lambda item: (-item["score"], item["frame_uid"]))
        if len(raw_sorted) <= top_k:
            return raw_sorted

        # Temporal NMS (suppress near-duplicate frames within 60 frames = 2.0s per video)
        selected = []
        selected_by_vid = {}
        for r in raw_sorted:
            vid = r["video_id"]
            fid = r["source_frame_index_zero_based"]
            too_close = False
            if vid in selected_by_vid:
                for sf in selected_by_vid[vid]:
                    if abs(fid - sf) < 60:
                        too_close = True
                        break
            if not too_close:
                selected.append(r)
                selected_by_vid.setdefault(vid, []).append(fid)
                if len(selected) >= top_k:
                    break

        return selected

    def _search_multi_path(self, plan: QueryPlan, top_k: int = 100, rerank: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Executes multi-path visual, lexical, and exact retrieval with rank-safe Reciprocal Rank Fusion."""
        self._initialize()
        visual_weight = _read_modality_weight("VISUAL_WEIGHT", 1.0)
        ocr_weight = _read_modality_weight("OCR_WEIGHT", 1.0) if self.enable_ocr else 0.0
        asr_weight = _read_modality_weight("ASR_WEIGHT", 0.8) if self.enable_asr else 0.0
        t_start = time.perf_counter()
        timings = {
            "visual_vi_ms": 0.0,
            "visual_en_ms": 0.0,
            "visual_retrieval_vi_ms": 0.0,
            "visual_retrieval_en_ms": 0.0,
            "lexical_ocr_ms": 0.0,
            "lexical_asr_ms": 0.0,
            "lexical_retrieval_ms": 0.0,
            "fusion_ms": 0.0,
            "rerank_ms": 0.0,
            "dedup_ms": 0.0,
            "existing_rerank_ms": 0.0,
            "total_query_ms": 0.0,
            "total_ms": 0.0,
            "number_visual_variants": len(plan.visual_queries),
            "visual_variant_count": len(plan.visual_queries),
            "number_exact_terms": len(plan.exact_strings),
            "lexical_term_count": len(plan.lexical_terms) + len(plan.exact_strings),
            "number_candidates_before_merge": 0,
            "number_candidates_after_merge": 0,
            "candidate_union_count": 0,
        }

        # 1. Multi-path visual retrieval
        channel_hits: Dict[str, Dict[Tuple[str, int], Tuple[float, Dict[str, Any]]]] = {}
        channel_weights: Dict[str, float] = {}

        for idx, vq in enumerate(plan.visual_queries):
            variant_weight = _validated_modality_weight(
                f"visual query weight at index {idx}", vq.weight
            )
            effective_weight = _validated_modality_weight(
                f"effective visual query weight at index {idx}",
                visual_weight * variant_weight,
            )
            if effective_weight == 0.0:
                continue
            t_v = time.perf_counter()
            chan_name = f"{vq.channel}_{idx}" if vq.channel in ("visual_vi", "visual_en") else f"{vq.language}_{idx}"
            vector = self._encoder.encode_text([vq.text])[0]
            hits = self._bundle.index.search(vector, max(top_k * 2, 200))
            dt_v = (time.perf_counter() - t_v) * 1000.0
            if vq.language == "vi":
                timings["visual_vi_ms"] += dt_v
                timings["visual_retrieval_vi_ms"] += dt_v
            else:
                timings["visual_en_ms"] += dt_v
                timings["visual_retrieval_en_ms"] += dt_v

            chan_dict = {}
            for h in hits:
                payload = self._bundle.resolver.resolve(h["frame_id"])
                key = (payload["video_id"], payload["source_frame_index_zero_based"])
                chan_dict[key] = (float(h["score"]), payload)
            channel_hits[chan_name] = chan_dict
            channel_weights[chan_name] = effective_weight

        # 2. Lexical & Exact OCR / ASR retrieval
        exact_terms = [ex.strip() for ex in plan.exact_strings if ex.strip()]
        lex_terms = [lt.strip() for lt in (plan.lexical_terms + plan.kept_vi_terms) if lt.strip()]
        if not lex_terms and not exact_terms:
            lex_terms = [plan.original_query]

        # OCR Channel
        t_ocr = time.perf_counter()
        ocr_diagnostics = {
            "invoked": bool(self.enable_ocr and ocr_weight > 0.0 and self._ocr),
            "matched_terms": [],
            "candidate_count": 0,
        }
        if self.enable_ocr and ocr_weight > 0.0 and self._ocr:
            ocr_chan: Dict[Tuple[str, int], Tuple[float, Dict[str, Any]]] = {}
            for o in self._ocr:
                score = 0.0
                # Priority 1: Exact string match
                for ex in exact_terms:
                    if _match_exact_term(ex, o.raw_text, o.normalized_text):
                        score = max(score, 1.0)
                        if ex not in ocr_diagnostics["matched_terms"]:
                            ocr_diagnostics["matched_terms"].append(ex)
                # Priority 2: Lexical term overlap
                if score < 1.0:
                    for lt in lex_terms:
                        s = lexical_score(lt, o.normalized_text)
                        if s >= 0.15:
                            score = max(score, s)
                            if lt not in ocr_diagnostics["matched_terms"]:
                                ocr_diagnostics["matched_terms"].append(lt)

                if score > 0.0:
                    key = (o.video_id, o.source_frame_index_zero_based)
                    payload = {
                        "video_id": o.video_id,
                        "frame_id": o.source_frame_index_zero_based,
                        "source_frame_index_zero_based": o.source_frame_index_zero_based,
                        "submission_frame_id": o.source_frame_index_zero_based,
                        "frame_uid": f"{o.video_id}:{str(o.source_frame_index_zero_based).zfill(9)}",
                        "timestamp_seconds": o.timestamp_seconds,
                        "image_path": f"frames/{str(o.source_frame_index_zero_based).zfill(9)}.jpg"
                    }
                    if key not in ocr_chan or score > ocr_chan[key][0]:
                        ocr_chan[key] = (score, payload)
            if ocr_chan:
                channel_hits["ocr"] = ocr_chan
                channel_weights["ocr"] = ocr_weight
            ocr_diagnostics["candidate_count"] = len(ocr_chan)
        timings["lexical_ocr_ms"] = (time.perf_counter() - t_ocr) * 1000.0
        timings["ocr_routing"] = ocr_diagnostics

        # ASR Channel
        t_asr = time.perf_counter()
        asr_diagnostics = {
            "invoked": bool(self.enable_asr and asr_weight > 0.0 and self._asr),
            "matched_terms": [],
            "candidate_count": 0,
        }
        if self.enable_asr and asr_weight > 0.0 and self._asr:
            asr_chan: Dict[Tuple[str, int], Tuple[float, Dict[str, Any]]] = {}
            for a in self._asr:
                if a.start_frame is None:
                    continue
                score = 0.0
                for ex in exact_terms:
                    if _match_exact_term(ex, a.raw_text, a.normalized_text):
                        score = max(score, 1.0)
                        if ex not in asr_diagnostics["matched_terms"]:
                            asr_diagnostics["matched_terms"].append(ex)
                if score < 1.0:
                    for lt in lex_terms:
                        s = lexical_score(lt, a.normalized_text)
                        if s >= 0.15:
                            score = max(score, s)
                            if lt not in asr_diagnostics["matched_terms"]:
                                asr_diagnostics["matched_terms"].append(lt)

                if score > 0.0:
                    key = (a.video_id, a.start_frame)
                    payload = {
                        "video_id": a.video_id,
                        "frame_id": a.start_frame,
                        "source_frame_index_zero_based": a.start_frame,
                        "submission_frame_id": a.start_frame,
                        "frame_uid": f"{a.video_id}:{str(a.start_frame).zfill(9)}",
                        "timestamp_seconds": a.start_seconds,
                        "image_path": f"frames/{str(a.start_frame).zfill(9)}.jpg"
                    }
                    if key not in asr_chan or score > asr_chan[key][0]:
                        asr_chan[key] = (score, payload)
            if asr_chan:
                channel_hits["asr"] = asr_chan
                channel_weights["asr"] = asr_weight
            asr_diagnostics["candidate_count"] = len(asr_chan)
        timings["lexical_asr_ms"] = (time.perf_counter() - t_asr) * 1000.0
        timings["asr_routing"] = asr_diagnostics
        timings["lexical_retrieval_ms"] = timings["lexical_ocr_ms"] + timings["lexical_asr_ms"]

        # 3. Candidate Union and Rank-Safe Reciprocal Rank Fusion (RRF)
        t_fus = time.perf_counter()
        k_rrf = float(QUERY_REFINER_RRF_K)
        # Compute channel rank mapping for each channel
        channel_ranks: Dict[str, Dict[Tuple[str, int], int]] = {}
        candidate_payloads: Dict[Tuple[str, int], Dict[str, Any]] = {}

        total_before = sum(len(hits_map) for hits_map in channel_hits.values())
        timings["number_candidates_before_merge"] = total_before

        for chan_name, hits_map in channel_hits.items():
            # Sort items by score descending
            sorted_items = sorted(hits_map.items(), key=lambda item: (-item[1][0], item[0]))
            rank_map = {}
            for rank_idx, (key, (score, payload)) in enumerate(sorted_items, start=1):
                rank_map[key] = rank_idx
                if key not in candidate_payloads:
                    candidate_payloads[key] = payload
            channel_ranks[chan_name] = rank_map

        timings["number_candidates_after_merge"] = len(candidate_payloads)
        timings["candidate_union_count"] = len(candidate_payloads)
        if not candidate_payloads:
            tot_ms = (time.perf_counter() - t_start) * 1000.0
            timings["total_query_ms"] = tot_ms
            timings["total_ms"] = tot_ms
            return [], timings

        fused_results = []
        for key, payload in candidate_payloads.items():
            vid, fid = key
            rrf_score = 0.0
            matched_by = []
            channels_scores = {}

            for chan_name, rank_map in channel_ranks.items():
                if key in rank_map:
                    r = rank_map[key]
                    w = channel_weights.get(chan_name, 1.0)
                    rrf_score += w / (k_rrf + r)
                    matched_by.append(chan_name)
                    channels_scores[chan_name] = {
                        "rank": r,
                        "raw_score": channel_hits[chan_name][key][0],
                    }

            filename = Path(payload.get("image_path", f"{str(fid).zfill(9)}.jpg")).name
            fused_results.append({
                "video_id": vid,
                "frame_id": payload.get("submission_frame_id", fid),
                "source_frame_index_zero_based": fid,
                "frame_uid": payload.get("frame_uid", f"{vid}:{str(fid).zfill(9)}"),
                "timestamp_seconds": payload.get("timestamp_seconds"),
                "score": float(rrf_score),
                "matched_by": matched_by,
                "channels": channels_scores,
                "image_url": f"/api/frames/{vid}/{filename}",
            })

        fused_results.sort(
            key=lambda item: (
                -item["score"],
                item["video_id"],
                item["source_frame_index_zero_based"],
            )
        )

        timings["fusion_ms"] = (time.perf_counter() - t_fus) * 1000.0

        # 4. Evidence-Aware Reranking
        t_rerank = time.perf_counter()
        if rerank and self.enable_rerank and self._candidate_reranker:
            reranked_results = self._candidate_reranker.rerank(
                fused_results,
                plan,
                ocr_evidence=self._ocr if self.enable_ocr else None,
                asr_evidence=self._asr if self.enable_asr else None,
            )
        else:
            reranked_results = fused_results
        dt_rerank = (time.perf_counter() - t_rerank) * 1000.0
        timings["rerank_ms"] = dt_rerank

        # 5. Temporal NMS / Deduplication
        t_nms = time.perf_counter()
        raw_sorted = reranked_results
        if len(raw_sorted) <= top_k:
            selected = raw_sorted
        else:
            selected = []
            selected_by_vid = {}
            for r in raw_sorted:
                vid = r["video_id"]
                fid = r["source_frame_index_zero_based"]
                too_close = False
                if vid in selected_by_vid:
                    for sf in selected_by_vid[vid]:
                        if abs(fid - sf) < 60:
                            too_close = True
                            break
                if not too_close:
                    selected.append(r)
                    selected_by_vid.setdefault(vid, []).append(fid)
                    if len(selected) >= top_k:
                        break

        dt_nms = (time.perf_counter() - t_nms) * 1000.0
        timings["dedup_ms"] = dt_nms
        timings["existing_rerank_ms"] = dt_nms + dt_rerank
        tot_ms = (time.perf_counter() - t_start) * 1000.0
        timings["total_query_ms"] = tot_ms
        timings["total_ms"] = tot_ms

        return selected, timings

    def search(self, query, top_k=100, query_refine=True, rerank=True):
        if not query_refine or not self.enable_query_refine:
            return self._search_single_query(query, top_k)

        refiner = self._get_query_refiner()
        plan, refiner_timings = refiner.refine(query, task_type="kis")
        results, search_timings = self._search_multi_path(plan, top_k, rerank=rerank)
        self.last_query_plan = plan
        self.last_query_metrics = {**refiner_timings, **search_timings}
        return results

    def search_image(self, image_or_path, top_k=100, deduplicate=True):
        """Retrieve visually similar frames for a query image using the existing FAISS index."""
        self._initialize()
        if isinstance(image_or_path, (str, Path)):
            img_path = Path(image_or_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            with Image.open(img_path) as opened:
                img = opened.convert("RGB")
        elif isinstance(image_or_path, Image.Image):
            img = image_or_path.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_or_path).__name__}")

        vector = self._encoder.encode_image(img)
        hits = self._bundle.index.search(vector, max(top_k * 2, 200))

        candidates = {}
        vis_scores_map = {}
        for h in hits:
            payload = self._bundle.resolver.resolve(h["frame_id"])
            key = (payload["video_id"], payload["source_frame_index_zero_based"])
            candidates[key] = payload
            vis_scores_map[key] = float(h["score"])

        if not candidates:
            return []

        results = []
        for key, payload in candidates.items():
            vid, fid = key
            v_raw = vis_scores_map.get(key, 0.0)
            filename = Path(payload.get("image_path", f"{str(fid).zfill(9)}.jpg")).name
            results.append({
                "video_id": vid,
                "frame_id": payload.get("submission_frame_id", fid),
                "source_frame_index_zero_based": fid,
                "frame_uid": payload.get("frame_uid", f"{vid}:{str(fid).zfill(9)}"),
                "timestamp_seconds": payload.get("timestamp_seconds"),
                "visual_score": v_raw,
                "ocr_score": 0.0,
                "asr_score": 0.0,
                "score": v_raw,
                "image_url": f"/api/frames/{vid}/{filename}"
            })

        raw_sorted = sorted(results, key=lambda item: (-item["score"], item["frame_uid"]))
        if not deduplicate or len(raw_sorted) <= top_k:
            return raw_sorted[:top_k]

        selected = []
        selected_by_vid = {}
        for r in raw_sorted:
            vid = r["video_id"]
            fid = r["source_frame_index_zero_based"]
            too_close = False
            if vid in selected_by_vid:
                for sf in selected_by_vid[vid]:
                    if abs(fid - sf) < 60:
                        too_close = True
                        break
            if not too_close:
                selected.append(r)
                selected_by_vid.setdefault(vid, []).append(fid)
                if len(selected) >= top_k:
                    break

        return selected

    def _qa_evidence_for_row(self, row, radius=150):
        video_id = row["video_id"]
        frame_id = row["source_frame_index_zero_based"]
        evidence = []
        for ocr in self._ocr:
            if (ocr.video_id == video_id
                    and abs(ocr.source_frame_index_zero_based - frame_id) <= radius):
                evidence.append({"id": ocr.frame_uid, "text": ocr.raw_text})
        for asr in self._asr:
            if (asr.video_id == video_id and asr.start_frame is not None
                    and (asr.start_frame - radius) <= frame_id
                    <= ((asr.end_frame or asr.start_frame) + radius)):
                evidence.append({"id": asr.segment_id, "text": asr.raw_text})
        seen_texts = set()
        unique = []
        for item in evidence:
            if item["text"] not in seen_texts:
                seen_texts.add(item["text"])
                unique.append(item)
        return unique

    def search_trake(self, events, top_k=100, temporal_refine=True, query_refine=True, rerank=True):
        from backend.app.retrieval.trake import EventCandidate, TRAKEAligner
        from backend.app.services.temporal_refiner import TemporalRefiner
        self._initialize()

        raw_events = events
        if isinstance(events, str):
            raw_events = [e.strip() for e in events.split("|") if e.strip()]
        if not raw_events:
            return []

        t_coarse_start = time.perf_counter()
        candidates = []
        event_labels = []

        refine_query_enabled = (
            query_refine
            and self.enable_query_refine
        )

        def _get_stage_rows(query_or_plan):
            if isinstance(query_or_plan, QueryPlan):
                if self._encoder is not None and self._bundle is not None:
                    rows, _ = self._search_multi_path(query_or_plan, top_k, rerank=rerank)
                    return rows
                query_str = query_or_plan.original_query
            else:
                query_str = query_or_plan
            try:
                return self.search(query_str, top_k, query_refine=refine_query_enabled, rerank=rerank)
            except TypeError:
                return self.search(query_str, top_k)

        if refine_query_enabled:
            refiner = self._get_query_refiner()
            if isinstance(events, list) and len(events) >= 2:
                for ev in raw_events:
                    st_plan, _ = refiner.refine(ev, task_type="kis")
                    stage_rows = _get_stage_rows(st_plan)
                    candidates.append([
                        EventCandidate(row["video_id"], row["source_frame_index_zero_based"], row["score"])
                        for row in stage_rows
                    ])
                    event_labels.append(ev)
            else:
                query_text = events if isinstance(events, str) else " | ".join(events)
                trake_plan, _ = refiner.refine(query_text, task_type="trake")
                self.last_query_plan = trake_plan

                if trake_plan.trake_stages and (not raw_events or len(trake_plan.trake_stages) == len(raw_events)):
                    for st in trake_plan.trake_stages:
                        stage_plan = QueryPlan(
                            task_type="trake",
                            original_query=st.original_text or st.visual_vi,
                            visual_queries=[
                                VisualQuery(language="vi", text=st.visual_vi, channel="visual_vi"),
                                VisualQuery(language="en", text=st.visual_en, channel="visual_en"),
                            ],
                            exact_strings=st.exact_strings,
                            kept_vi_terms=st.kept_vi_terms,
                            lexical_terms=st.exact_strings + st.kept_vi_terms,
                            refinement_used=True,
                            refinement_backend=trake_plan.refinement_backend,
                        )
                        stage_rows = _get_stage_rows(stage_plan)
                        candidates.append([
                            EventCandidate(row["video_id"], row["source_frame_index_zero_based"], row["score"])
                            for row in stage_rows
                        ])
                        event_labels.append(st.visual_vi)
                else:
                    for ev in raw_events:
                        stage_rows = _get_stage_rows(ev)
                        candidates.append([
                            EventCandidate(row["video_id"], row["source_frame_index_zero_based"], row["score"])
                            for row in stage_rows
                        ])
                        event_labels.append(ev)
        else:
            for ev in raw_events:
                stage_rows = _get_stage_rows(ev)
                candidates.append([
                    EventCandidate(row["video_id"], row["source_frame_index_zero_based"], row["score"])
                    for row in stage_rows
                ])
            event_labels = list(raw_events)

        coarse_retrieval_ms = (time.perf_counter() - t_coarse_start) * 1000.0

        metrics = {
            "refinement_used": False,
            "number_stages": len(event_labels),
            "refinement_regions": 0,
            "regions_considered": 0,
            "regions_refined": 0,
            "dense_frames_decoded": 0,
            "dense_frames_embedded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "coarse_retrieval_ms": coarse_retrieval_ms,
            "region_build_ms": 0.0,
            "dense_decode_ms": 0.0,
            "dense_embedding_ms": 0.0,
            "temporal_scoring_ms": 0.0,
            "temporal_refinement_ms": 0.0,
            "trake_dp_ms": 0.0,
            "trake_alignment_ms": 0.0,
            "total_ms": 0.0,
        }
        refine_temporal_enabled = (
            temporal_refine
            and os.getenv("TRAKE_TEMPORAL_REFINE_ENABLED", "true").lower() in ("1", "true", "yes")
        )
        if refine_temporal_enabled:
            refiner = TemporalRefiner(processed_root=self.processed_root, encoder=self._encoder)
            aligned_candidates, refiner_metrics = refiner.refine_trake_candidates(event_labels, candidates, encoder=self._encoder)
            metrics.update(refiner_metrics)
            metrics["coarse_retrieval_ms"] = coarse_retrieval_ms
            metrics["temporal_refinement_ms"] = refiner_metrics.get("total_ms", 0.0)
            metrics["refinement_regions"] = refiner_metrics.get("regions_refined", 0)
            t_align_start = time.perf_counter()
            result = TRAKEAligner().align(aligned_candidates)
            dt_align = (time.perf_counter() - t_align_start) * 1000.0
            metrics["trake_dp_ms"] = dt_align
            metrics["trake_alignment_ms"] = dt_align
            metrics["total_ms"] = coarse_retrieval_ms + refiner_metrics.get("total_ms", 0.0) + dt_align
        else:
            t_align_start = time.perf_counter()
            result = TRAKEAligner().align(candidates)
            dt_align = (time.perf_counter() - t_align_start) * 1000.0
            metrics["trake_dp_ms"] = dt_align
            metrics["trake_alignment_ms"] = dt_align
        from backend.app.services.trake_coherence import TRAKECoherenceAnalyzer
        coherence_analyzer = TRAKECoherenceAnalyzer()
        if result and result.frame_ids:
            diag = coherence_analyzer.analyze(result.video_id, result.frame_ids)
            metrics["frame_gaps"] = diag.frame_gaps
            metrics["max_frame_gap"] = diag.max_gap
            metrics["mean_frame_gap"] = diag.mean_gap
            metrics["median_frame_gap"] = diag.median_gap
            metrics["total_frame_span"] = diag.total_frame_span
            metrics["coherence_mode"] = coherence_analyzer.mode
            metrics["normalized_dispersion"] = diag.normalized_dispersion
        else:
            metrics["frame_gaps"] = []
            metrics["max_frame_gap"] = 0
            metrics["mean_frame_gap"] = 0.0
            metrics["median_frame_gap"] = 0.0
            metrics["total_frame_span"] = 0
            metrics["coherence_mode"] = coherence_analyzer.mode
            metrics["normalized_dispersion"] = 0.0

        self.last_trake_metrics = metrics

        if result:
            diag_dict = coherence_analyzer.analyze(result.video_id, result.frame_ids).to_dict() if result.frame_ids else {}
            res_item = {
                "video_id": result.video_id,
                "frame_ids": result.frame_ids,
                "frame_id": result.frame_ids[0] if result.frame_ids else None,
                "events": [{"frame_id": fid} for fid in result.frame_ids],
                "score": result.score,
                "image_url": f"/api/frames/{result.video_id}/{str(result.frame_ids[0]).zfill(9)}.jpg" if result.frame_ids else None,
                "refinement_used": metrics.get("refinement_used", False),
                "refinement_regions": metrics.get("regions_refined", 0),
                "coherence": diag_dict,
            }
            return [res_item]
        return []

    def handle(self, request):
        query_type = request.get("query_type", "kis")
        top_k = int(request.get("top_k", 100))
        query_refine = request.get("query_refine", True)
        rerank = request.get("rerank", True)

        if query_type == "image":
            image_input = request.get("image") or request.get("image_path")
            if not image_input:
                raise ValueError("Image query requires 'image' or 'image_path'")
            return self.search_image(image_input, top_k)

        if query_type == "trake":
            events = request.get("events", [])
            refine = request.get("temporal_refine", True)
            return self.search_trake(events, top_k=top_k, temporal_refine=refine, query_refine=query_refine, rerank=rerank)

        query = request.get("query", "")
        if query_type == "qa":
            from backend.app.retrieval.qa_query_decomposition import QAQueryDecomposer
            from backend.app.retrieval.video_qa import ExtractiveAnswerer
            decomposer = QAQueryDecomposer()
            decomp = decomposer.decompose(query)
            try:
                rows = self.search(decomp["retrieval_query"], top_k, query_refine=query_refine, rerank=rerank)
            except TypeError:
                rows = self.search(decomp["retrieval_query"], top_k)
            answerer = ExtractiveAnswerer()
            results = []
            for row in rows:
                qa_res = answerer.answer(query, self._qa_evidence_for_row(row))
                results.append({
                    **row,
                    "answer": qa_res["answer"],
                    "confidence": qa_res["confidence"],
                    "evidence_sources": qa_res["evidence_sources"],
                    "decomposed_query": decomp["retrieval_query"]
                })
            return results

        try:
            return self.search(query, top_k, query_refine=query_refine, rerank=rerank)
        except TypeError:
            return self.search(query, top_k)
