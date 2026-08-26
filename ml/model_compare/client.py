"""Minimal OpenAI-compatible streaming client with timing instrumentation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any


_STREAM_OPTIONS_UNSUPPORTED: set[str] = set()


@dataclass(frozen=True)
class Endpoint:
    """Connection details for one model under evaluation."""

    label: str
    base_url: str
    model_id: str
    api_key: str
    extra_headers: dict[str, str] = field(default_factory=dict)
    request_body: dict[str, Any] = field(default_factory=dict)

    @property
    def chat_url(self) -> str:
        return normalize_chat_url(self.base_url)


@dataclass
class Completion:
    """A completion plus latency and token-accounting information."""

    text: str
    tool_calls: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    usage_source: str
    ttft_seconds: float
    decode_seconds: float
    e2e_seconds: float
    finish_reason: str | None
    attempts: int
    response_mode: str

    @property
    def decode_tokens_per_second(self) -> float:
        return self.decode_token_count / max(self.decode_seconds, 0.001)

    @property
    def decode_token_count(self) -> int:
        if self.response_mode == "stream":
            return max(self.completion_tokens - 1, 0)
        return self.completion_tokens


class APIError(RuntimeError):
    """An endpoint returned an unusable response."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def normalize_chat_url(base_url: str) -> str:
    """Accept a host, a /v1 root, or a full chat-completions URL."""
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if not value.endswith("/v1"):
        value += "/v1"
    return value + "/chat/completions"


def estimate_tokens(text: str) -> int:
    """Conservative tokenizer-free fallback, clearly marked in reports."""
    if not text:
        return 0
    pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    byte_adjustment = max(0, len(text.encode("utf-8")) - len(text)) // 4
    return max(1, len(pieces) + byte_adjustment)


def _headers(endpoint: Endpoint) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {endpoint.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    headers.update(endpoint.extra_headers)
    return headers


def _append_tool_delta(
    accumulators: dict[int, dict[str, Any]], tool_deltas: list[dict[str, Any]]
) -> None:
    for position, delta in enumerate(tool_deltas):
        index = int(delta.get("index", position))
        current = accumulators.setdefault(
            index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if delta.get("id"):
            current["id"] += str(delta["id"])
        function = delta.get("function") or {}
        if function.get("name"):
            current["function"]["name"] += str(function["name"])
        if function.get("arguments"):
            current["function"]["arguments"] += str(function["arguments"])


def _finalize_tool_calls(accumulators: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for index in sorted(accumulators):
        item = accumulators[index]
        raw_arguments = item["function"].get("arguments", "")
        try:
            arguments: Any = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            arguments = raw_arguments
        calls.append(
            {
                "name": item["function"].get("name", ""),
                "arguments": arguments,
                "id": item.get("id", ""),
            }
        )
    return calls


def _parse_json_response(response: Any, started: float, attempts: int) -> Completion:
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise APIError("response contained no choices", response.status_code)
    choice = choices[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    calls = []
    for item in message.get("tool_calls") or []:
        function = item.get("function") or {}
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments: Any = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError):
            arguments = raw_arguments
        calls.append(
            {"name": function.get("name", ""), "arguments": arguments, "id": item.get("id", "")}
        )
    finished = time.perf_counter()
    usage = payload.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    usage_source = "endpoint"
    if completion_tokens < 1:
        completion_tokens = estimate_tokens(text or json.dumps(calls, sort_keys=True))
        prompt_tokens = 0
        usage_source = "estimated"
    elapsed = finished - started
    return Completion(
        text=text,
        tool_calls=calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        usage_source=usage_source,
        ttft_seconds=elapsed,
        decode_seconds=max(elapsed, 0.001),
        e2e_seconds=elapsed,
        finish_reason=choice.get("finish_reason"),
        attempts=attempts,
        response_mode="non_stream",
    )


def _parse_stream_response(response: Any, started: float, attempts: int) -> Completion:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_accumulators: dict[int, dict[str, Any]] = {}
    first_emission: float | None = None
    usage: dict[str, Any] = {}
    finish_reason: str | None = None

    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        if not line or not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise APIError(f"invalid SSE JSON: {raw[:160]}", response.status_code) from exc
        if event.get("usage"):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        tool_deltas = delta.get("tool_calls") or []
        if content:
            text_parts.append(str(content))
        if reasoning:
            reasoning_parts.append(str(reasoning))
        if tool_deltas:
            _append_tool_delta(tool_accumulators, tool_deltas)
        if first_emission is None and (content or reasoning or tool_deltas):
            first_emission = time.perf_counter()

    finished = time.perf_counter()
    text = "".join(text_parts)
    calls = _finalize_tool_calls(tool_accumulators)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    usage_source = "endpoint"
    if completion_tokens < 1:
        measured_text = text + "".join(reasoning_parts) + json.dumps(calls, sort_keys=True)
        completion_tokens = estimate_tokens(measured_text)
        prompt_tokens = 0
        usage_source = "estimated"
    first = first_emission or finished
    return Completion(
        text=text,
        tool_calls=calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        usage_source=usage_source,
        ttft_seconds=max(first - started, 0.0),
        decode_seconds=max(finished - first, 0.001),
        e2e_seconds=max(finished - started, 0.001),
        finish_reason=finish_reason,
        attempts=attempts,
        response_mode="stream",
    )


def complete(
    endpoint: Endpoint,
    *,
    messages: list[dict[str, Any]],
    max_tokens: int | None,
    timeout_seconds: float,
    temperature: float = 0.0,
    tools: list[dict[str, Any]] | None = None,
    retries: int = 2,
) -> Completion:
    """Call one endpoint and measure TTFT, E2E latency, and decode throughput."""
    import requests

    body: dict[str, Any] = {
        "model": endpoint.model_id,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if endpoint.chat_url not in _STREAM_OPTIONS_UNSUPPORTED:
        body["stream_options"] = {"include_usage": True}
    body.update(endpoint.request_body)
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        started = time.perf_counter()
        try:
            response = requests.post(
                endpoint.chat_url,
                headers=_headers(endpoint),
                json=body,
                stream=True,
                timeout=(15, timeout_seconds),
            )
            if not response.ok and response.status_code in {400, 422} and "stream_options" in body:
                unsupported_detail = response.text[:500].lower()
                if "stream_options" in unsupported_detail or "include_usage" in unsupported_detail:
                    response.close()
                    body.pop("stream_options", None)
                    _STREAM_OPTIONS_UNSUPPORTED.add(endpoint.chat_url)
                    response = requests.post(
                        endpoint.chat_url,
                        headers=_headers(endpoint),
                        json=body,
                        stream=True,
                        timeout=(15, timeout_seconds),
                    )
            if not response.ok:
                detail = response.text[:500].replace("\n", " ")
                response.close()
                raise APIError(f"HTTP {response.status_code}: {detail}", response.status_code)
            content_type = response.headers.get("content-type", "").lower()
            try:
                if "text/event-stream" in content_type:
                    return _parse_stream_response(response, started, attempt)
                return _parse_json_response(response, started, attempt)
            finally:
                response.close()
        except Exception as exc:  # noqa: BLE001 - remote failures must be retained
            last_error = exc
            status = getattr(exc, "status_code", None)
            retryable = status in {408, 409, 425, 429, 500, 502, 503, 504} or status is None
            if attempt > retries or not retryable:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    assert last_error is not None
    raise last_error
