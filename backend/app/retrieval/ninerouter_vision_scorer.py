import base64
import json
import math
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from urllib import error, request

from backend.app.retrieval.model_scorer import ModelScorer, validate_scores


class NineRouterError(RuntimeError):
    pass


class NineRouterHTTPError(NineRouterError):
    def __init__(self, status: int, message: str = ""):
        super().__init__(f"9Router HTTP {status}{': ' + message if message else ''}")
        self.status = status


def _default_transport(url, headers, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    call = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(call, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")[:500]
        raise NineRouterHTTPError(exc.code, message) from exc
    except (error.URLError, TimeoutError) as exc:
        raise NineRouterError("9Router connection failed") from exc


class NineRouterVisionScorer(ModelScorer):
    backend_name = "9router_multimodal_relevance"
    signal_type = "joint_multimodal_relevance"
    RETRYABLE_STATUS = {429, 502, 503, 504}
    SUPPORTED_MIME = {"image/png", "image/jpeg", "image/webp"}
    MODEL_IDENTITIES = {
        "cx/gpt-5.6-sol": "gpt-5.6-sol",
        "gpt-5.6-sol": "gpt-5.6-sol",
    }

    @classmethod
    def normalize_model_identity(cls, model):
        if not isinstance(model, str):
            return None
        return cls.MODEL_IDENTITIES.get(model.strip().lower())

    def __init__(
        self,
        base_url="http://localhost:20128/v1",
        api_key=None,
        model="cx/gpt-5.6-sol",
        timeout_seconds=120,
        max_concurrency=2,
        max_retries=2,
        transport=None,
        sleep=time.sleep,
    ):
        if max_concurrency <= 0 or timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("invalid 9Router scorer configuration")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("M14_9ROUTER_API_KEY")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.transport = transport or _default_transport
        self.sleep = sleep
        self._diagnostics = {}

    @staticmethod
    def _candidate_id(candidate):
        return str(candidate.get("candidate_id", candidate.get("frame_id", "")))

    @classmethod
    def _image_data_url(cls, candidate):
        image = candidate.get("image", candidate.get("image_path", candidate.get("path")))
        if image is None:
            raise ValueError("candidate image payload is required")
        if not isinstance(image, (str, Path)):
            raise ValueError("9Router scorer requires a local image path")
        path = Path(image)
        if not path.is_file():
            raise ValueError("candidate image path does not exist")
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in cls.SUPPORTED_MIME:
            raise ValueError("unsupported candidate image type")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _payload(self, query, candidate):
        return {
            "model": self.model,
            "temperature": 0,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a visual search relevance judge. Judge only how well the image satisfies the search query. "
                        "Return exactly one JSON object with one numeric field named score and no other text. "
                        "Use 90-100 for direct strong matches, 70-89 for clear relevance with a minor mismatch, "
                        "40-69 for partial or ambiguous relevance, 10-39 for related hard negatives, and 0-9 for irrelevance."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Search query: {query}"},
                        {"type": "image_url", "image_url": {"url": self._image_data_url(candidate)}},
                    ],
                },
            ],
        }

    @staticmethod
    def _response_score(response):
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("9Router response must contain exactly one choice")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("9Router response content is missing")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("9Router response is not valid JSON") from exc
        if not isinstance(parsed, dict) or set(parsed) != {"score"}:
            raise ValueError("9Router response must contain only score")
        score = parsed["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("9Router score must be numeric")
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError("9Router score must be finite and between 0 and 100")
        return score

    def _score_candidate(self, query, candidate):
        if not self.api_key:
            raise NineRouterError("M14_9ROUTER_API_KEY is not configured")
        payload = self._payload(query, candidate)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        retries = 0
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                status, response = self.transport(
                    f"{self.base_url}/chat/completions", headers, payload, self.timeout_seconds
                )
                if status >= 400:
                    raise NineRouterHTTPError(status)
                score = self._response_score(response)
                usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
                choice = response["choices"][0]
                return {
                    "candidate_id": self._candidate_id(candidate),
                    "score": score,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "retries": retries,
                    "actual_model": response.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                    "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "output_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "cost": usage.get("cost"),
                    "routing": response.get("routing"),
                }
            except NineRouterHTTPError as exc:
                if exc.status not in self.RETRYABLE_STATUS or attempt == self.max_retries:
                    raise
            except NineRouterError:
                if attempt == self.max_retries:
                    raise
            retries += 1
            self.sleep(min(0.25 * (2**attempt), 2.0))
        raise NineRouterError("9Router retry budget exhausted")

    def score_batch(self, query, candidates):
        candidates = list(candidates)
        identifiers = [self._candidate_id(candidate) for candidate in candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate IDs must be unique")
        started = time.perf_counter()
        results = [None] * len(candidates)
        failures = []
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {executor.submit(self._score_candidate, query, candidate): index for index, candidate in enumerate(candidates)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    failures.append(type(exc).__name__)
        completed = [result for result in results if result is not None]
        response_models = sorted({result["actual_model"] for result in completed if result["actual_model"]})
        normalized_requested_model = self.normalize_model_identity(self.model)
        normalized_response_models = sorted({self.normalize_model_identity(model) for model in response_models})
        fixed_model_preserved = (
            normalized_requested_model is not None
            and normalized_response_models == [normalized_requested_model]
        )
        costs = [result["cost"] for result in completed if isinstance(result["cost"], (int, float))]
        self._diagnostics = {
            "reranker_backend": self.backend_name,
            "signal_type": self.signal_type,
            "requested_model": self.model,
            "requested_route_model": self.model,
            "actual_models": response_models,
            "response_models": response_models,
            "normalized_model_family": normalized_requested_model,
            "normalized_response_models": normalized_response_models,
            "fixed_model_preserved": fixed_model_preserved,
            "candidate_count": len(candidates),
            "successful_scores": len(completed),
            "failed_scores": len(failures),
            "batch_size": 1,
            "batch_count": len(candidates),
            "max_concurrency": self.max_concurrency,
            "request_count": len(candidates) + sum(result["retries"] for result in completed),
            "retry_count": sum(result["retries"] for result in completed),
            "failed_request_count": len(failures),
            "input_tokens": sum(result["input_tokens"] for result in completed),
            "output_tokens": sum(result["output_tokens"] for result in completed),
            "api_latency_ms": [result["latency_ms"] for result in completed],
            "total_rerank_ms": (time.perf_counter() - started) * 1000,
            "finish_reasons": sorted({result["finish_reason"] for result in completed if result["finish_reason"]}),
            "cost": sum(costs) if len(costs) == len(completed) and completed else None,
            "routing_metadata_available": any(result["routing"] is not None for result in completed),
        }
        if failures:
            raise NineRouterError(f"candidate scoring failed: {failures[0]}")
        if completed and not fixed_model_preserved:
            raise NineRouterError("fixed-model execution was not preserved")
        scores = validate_scores([result["score"] for result in results], len(candidates))
        return scores

    @property
    def diagnostics(self):
        return dict(self._diagnostics)
