"""Talk to any server that speaks the OpenAI ``/chat/completions`` API.

One backend, many engines
-------------------------
Ollama, llama.cpp's ``llama-server``, LM Studio, vLLM, text-generation-webui
and OpenAI itself all expose the same endpoint. Supporting that one wire format
covers the entire local-inference ecosystem and is also the scaling story: move
the server to a bigger machine, change ``base_url``, and nothing else changes.

Only ``httpx`` is required, so this backend works in a bare install.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from kumaru.backends.base import Backend, GenerationResult
from kumaru.core.config import BackendConfig
from kumaru.core.errors import BackendError, ModelNotAvailableError
from kumaru.core.logging import get_logger
from kumaru.core.types import ChatRequest, StreamChunk, Usage

log = get_logger("backends.openai_compat")


class OpenAICompatBackend(Backend):
    """HTTP client for an OpenAI-compatible chat completions endpoint."""

    name = "openai_compat"

    def __init__(
        self, config: BackendConfig, *, client: httpx.Client | None = None
    ) -> None:
        super().__init__(config)
        if not config.model:
            raise ModelNotAvailableError(
                "backend.model must be set for the openai_compat backend",
                hint="e.g. model: llama3.1:8b for Ollama, or the name your server reports",
            )
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_s,
            headers=self._headers(),
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.config.api_key
        if key:
            # Local servers ignore this; OpenAI and vLLM require it.
            headers["Authorization"] = "Bearer " + key
        return headers

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": request.as_dicts(),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.stop:
            payload["stop"] = request.stop
        payload.update(self.config.extra)
        return payload

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            if exc.response.status_code == 404:
                raise ModelNotAvailableError(
                    f"model '{self.config.model}' not found at {self.config.base_url}",
                    hint="check the model name, and that the server has it loaded",
                ) from exc
            raise BackendError(
                f"backend returned HTTP {exc.response.status_code}: {body}"
            ) from exc
        except httpx.RequestError as exc:
            raise BackendError(
                f"could not reach {self.config.base_url}: {exc}",
                hint="is your local model server running?",
            ) from exc
        return response

    @staticmethod
    def _usage(data: dict[str, Any]) -> Usage:
        usage = data.get("usage") or {}
        return Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )

    def generate(self, request: ChatRequest) -> GenerationResult:
        data = self._post(self._payload(request, stream=False)).json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"unexpected response shape: {data!r}"[:500]) from exc
        return GenerationResult(text=text, usage=self._usage(data))

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        payload = self._payload(request, stream=True)
        usage = Usage()
        try:
            with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise BackendError(
                        f"backend returned HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        log.warning("skipping malformed SSE chunk")
                        continue
                    if event.get("usage"):
                        usage = self._usage(event)
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    if delta:
                        yield StreamChunk(delta=delta)
        except httpx.RequestError as exc:
            raise BackendError(
                f"stream from {self.config.base_url} failed: {exc}",
                hint="is your local model server running?",
            ) from exc
        yield StreamChunk(done=True, usage=usage)

    def health(self) -> dict[str, object]:
        info: dict[str, object] = {
            "backend": self.name,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "ready": False,
        }
        try:
            response = self._client.get("/models")
            info["ready"] = response.status_code < 500
        except httpx.RequestError as exc:
            info["error"] = str(exc)
        return info

    def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            self._client.close()
