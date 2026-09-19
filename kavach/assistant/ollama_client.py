"""Small local Ollama HTTP client with response and resource benchmarking."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from time import monotonic
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


class OllamaError(RuntimeError):
    """Base class for expected local Ollama failures."""


class OllamaUnavailableError(OllamaError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaModelError(OllamaError):
    """Raised when the requested local model is not available."""


@dataclass(frozen=True)
class OllamaResponse:
    """One non-streaming Ollama chat response and timing counters."""

    model: str
    content: str
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    raw: Mapping[str, object] | None = None

    @property
    def total_duration_seconds(self) -> float | None:
        """Return Ollama's reported duration when supplied."""

        if self.total_duration_ns is None:
            return None
        return self.total_duration_ns / 1_000_000_000.0


@dataclass(frozen=True)
class OllamaBenchmark:
    """Practical latency and loaded-model resource measurements."""

    model: str
    requests: int
    average_latency_seconds: float
    minimum_latency_seconds: float
    maximum_latency_seconds: float
    prompt_tokens: int | None
    response_tokens: int | None
    resource_usage: Mapping[str, object] | None

    def to_dict(self) -> dict[str, object]:
        """Return benchmark data for documentation or logging."""

        return {
            "model": self.model,
            "requests": self.requests,
            "average_latency_seconds": self.average_latency_seconds,
            "minimum_latency_seconds": self.minimum_latency_seconds,
            "maximum_latency_seconds": self.maximum_latency_seconds,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "resource_usage": (
                None if self.resource_usage is None else dict(self.resource_usage)
            ),
        }


class OllamaClient:
    """Call a local Ollama server using its JSON HTTP API.

    The client never downloads a model automatically. The requested model must
    already exist in the local Ollama installation.
    """

    def __init__(
        self,
        *,
        host: str = DEFAULT_OLLAMA_HOST,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout_seconds: float = 120.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        host_value = str(host).strip().rstrip("/")
        model_value = str(model).strip()
        timeout = float(timeout_seconds)
        if not host_value:
            raise OllamaError("Ollama host cannot be empty")
        if not model_value:
            raise OllamaModelError("Ollama model cannot be empty")
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise OllamaError("timeout_seconds must be positive and finite")
        self.host = host_value
        self.model = model_value
        self.timeout_seconds = timeout
        self._opener = opener or urlopen

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: Mapping[str, object] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.host}{endpoint}",
            data=body,
            headers=headers,
            method=method,
        )
        timeout = self.timeout_seconds if timeout_seconds is None else float(
            timeout_seconds
        )
        try:
            with self._opener(request, timeout=timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            if exc.code == 404 and endpoint == "/api/chat":
                raise OllamaModelError(
                    f"Ollama model {self.model!r} was not found: {detail}"
                ) from exc
            raise OllamaError(
                f"Ollama HTTP error {exc.code} at {endpoint}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaUnavailableError(
                f"could not reach local Ollama at {self.host}: {exc}"
            ) from exc
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama returned invalid JSON from {endpoint}") from exc
        if not isinstance(decoded, dict):
            raise OllamaError(f"Ollama response from {endpoint} was not an object")
        if decoded.get("error"):
            detail = str(decoded["error"])
            if endpoint == "/api/chat":
                raise OllamaModelError(detail)
            raise OllamaError(detail)
        return decoded

    def list_models(self) -> tuple[str, ...]:
        """Return names of models currently known to local Ollama."""

        payload = self._request("GET", "/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise OllamaError("Ollama model listing has an invalid shape")
        names = []
        for model in models:
            if isinstance(model, Mapping) and model.get("name"):
                names.append(str(model["name"]))
        return tuple(names)

    def is_available(self) -> bool:
        """Return whether the local service responds successfully."""

        try:
            self._request("GET", "/api/tags", timeout_seconds=5.0)
        except OllamaError:
            return False
        return True

    def model_available(self) -> bool:
        """Return whether the configured model is installed locally."""

        return self.model in self.list_models()

    @staticmethod
    def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
        value = payload.get(key)
        if value is None or isinstance(value, bool):
            return None
        try:
            integer = int(value)
        except (TypeError, ValueError):
            return None
        return integer if integer >= 0 else None

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        options: Mapping[str, object] | None = None,
        keep_alive: str | int | None = None,
        timeout_seconds: float | None = None,
    ) -> OllamaResponse:
        """Send grounded messages to Ollama without streaming."""

        if not messages:
            raise OllamaError("chat messages cannot be empty")
        normalized_messages = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise OllamaError("each chat message must be a mapping")
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", ""))
            if role not in {"system", "user", "assistant"} or not content:
                raise OllamaError("chat messages require a valid role and content")
            normalized_messages.append({"role": role, "content": content})
        payload: dict[str, object] = {
            "model": self.model,
            "messages": normalized_messages,
            "stream": False,
        }
        if options is not None:
            payload["options"] = dict(options)
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        response = self._request(
            "POST",
            "/api/chat",
            payload,
            timeout_seconds=timeout_seconds,
        )
        message = response.get("message")
        if not isinstance(message, Mapping):
            raise OllamaError("Ollama chat response has no message")
        content = str(message.get("content", "")).strip()
        if not content:
            raise OllamaError("Ollama chat response has empty content")
        return OllamaResponse(
            model=str(response.get("model", self.model)),
            content=content,
            total_duration_ns=self._optional_int(response, "total_duration"),
            load_duration_ns=self._optional_int(response, "load_duration"),
            prompt_eval_count=self._optional_int(response, "prompt_eval_count"),
            eval_count=self._optional_int(response, "eval_count"),
            raw=response,
        )

    def resource_usage(self) -> dict[str, object]:
        """Return loaded-model size and VRAM counters reported by Ollama.

        These are model allocation counters from the Ollama API, not a full
        operating-system RAM measurement.
        """

        payload = self._request("GET", "/api/ps")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise OllamaError("Ollama process listing has an invalid shape")
        normalized_models: list[dict[str, object]] = []
        total_size = 0
        total_vram = 0
        for model in models:
            if not isinstance(model, Mapping):
                continue
            size = self._optional_int(model, "size") or 0
            size_vram = self._optional_int(model, "size_vram") or 0
            total_size += size
            total_vram += size_vram
            normalized_models.append(
                {
                    "name": str(model.get("name", "")),
                    "size_bytes": size,
                    "vram_bytes": size_vram,
                }
            )
        return {
            "models": normalized_models,
            "total_model_size_bytes": total_size,
            "total_vram_bytes": total_vram,
            "measurement": "Ollama /api/ps model allocation counters",
        }

    def benchmark(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        repeats: int = 1,
        options: Mapping[str, object] | None = None,
    ) -> OllamaBenchmark:
        """Measure end-to-end local chat latency and practical model usage."""

        if int(repeats) <= 0:
            raise OllamaError("repeats must be positive")
        try:
            resources = self.resource_usage()
        except OllamaError:
            resources = None
        latencies: list[float] = []
        prompt_tokens: list[int] = []
        response_tokens: list[int] = []
        for _ in range(int(repeats)):
            started = monotonic()
            response = self.chat(messages, options=options)
            elapsed = monotonic() - started
            reported = response.total_duration_seconds
            latencies.append(elapsed if reported is None else max(elapsed, reported))
            if response.prompt_eval_count is not None:
                prompt_tokens.append(response.prompt_eval_count)
            if response.eval_count is not None:
                response_tokens.append(response.eval_count)
        try:
            post_resources = self.resource_usage()
        except OllamaError:
            post_resources = None
        return OllamaBenchmark(
            model=self.model,
            requests=int(repeats),
            average_latency_seconds=sum(latencies) / len(latencies),
            minimum_latency_seconds=min(latencies),
            maximum_latency_seconds=max(latencies),
            prompt_tokens=(
                None if not prompt_tokens else round(sum(prompt_tokens) / len(prompt_tokens))
            ),
            response_tokens=(
                None
                if not response_tokens
                else round(sum(response_tokens) / len(response_tokens))
            ),
            resource_usage=post_resources or resources,
        )
