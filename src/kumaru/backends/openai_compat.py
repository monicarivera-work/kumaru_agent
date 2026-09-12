"""Backend for any server that speaks the OpenAI chat-completions protocol.

One adapter, many engines: **vLLM**, **SGLang**, **TGI**, **Ollama**,
**LM Studio**, **llama.cpp server**, and hosted APIs all expose this same
interface. Pointing `backend.base_url` at a different process is therefore the
cheapest possible scaling step you can take.

Why this matters more than it looks
-----------------------------------
`transformers_backend` is excellent for development and for one user at a time,
but it decodes one request at a time. A dedicated serving engine like vLLM adds
**continuous batching** and **PagedAttention**: it interleaves many requests into
the same decode step and manages KV-cache memory in fixed-size pages rather than
one contiguous slab per sequence. The practical effect is roughly an order of
magnitude more throughput on the exact same GPU.

So the migration path is: develop against `transformers`, then start
``vllm serve Qwen/Qwen2.5-7B-Instruct`` and change two lines of YAML. Nothing
above this file changes. That is the whole point of the backend seam.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from kumaru.backends.base import Backend, StopSequenceTracker
from kumaru.core.config import BackendConfig
from kumaru.core.errors import BackendError
from kumaru.core.logging import get_logger
from kumaru.core.registry import BACKENDS
from kumaru.core.types import BackendInfo, GenerationParams, Message, StreamChunk, Usage

logger = get_logger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"  # Ollama's default


@BACKENDS.register("openai_compat", aliases=("ollama", "vllm", "remote"))
class OpenAICompatBackend(Backend):
    """Delegate generation to a remote OpenAI-compatible HTTP endpoint."""

    name = "openai_compat"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._client: Any = None
        self._base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")

    async def _load(self) -> None:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(self.config.timeout_s, connect=10.0),
        )

        # Fail fast and loudly at startup rather than on the user's first
        # message. A connection error here has an obvious cause (server not
        # running); the same error mid-conversation looks like a model bug.
        try:
            resp = await self._client.get("/models")
            resp.raise_for_status()
        except Exception as exc:
            raise BackendError(
                f"Cannot reach OpenAI-compatible server at {self._base_url}: {exc}",
                hint=(
                    "Start one first, e.g. `ollama serve` (port 11434) or "
                    "`vllm serve <model> --port 8000`, then set backend.base_url."
                ),
            ) from exc

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._loaded = False

    def info(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            model_id=self.config.model,
            device="remote",
            dtype="remote",
            context_length=self.config.max_context_length,
            supports_streaming=True,
            extra={"base_url": self._base_url},
        )

    def _payload(
        self, messages: Sequence[Message], params: GenerationParams, *, stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_openai() for m in messages],
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": stream,
        }
        if params.stop:
            payload["stop"] = params.stop[:4]  # most servers cap this at 4
        if params.seed is not None:
            payload["seed"] = params.seed
        if stream:
            # vLLM and newer llama.cpp honour this and send a final usage chunk;
            # servers that do not simply ignore the field.
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def stream(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> AsyncIterator[StreamChunk]:
        if not self._loaded:
            await self.load()

        # Stop sequences are enforced locally as well: server-side support is
        # inconsistent, and silently returning text past a stop sequence breaks
        # downstream parsing.
        tracker = StopSequenceTracker(params.stop)
        usage = Usage()
        finish_reason: Any = "stop"

        async with self._client.stream(
            "POST", "/chat/completions", json=self._payload(messages, params, stream=True)
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")[:500]
                raise BackendError(f"Upstream returned {response.status_code}: {body}")

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    logger.debug("skipping malformed SSE payload: %r", data[:120])
                    continue

                if event.get("usage"):
                    usage = Usage(
                        prompt_tokens=event["usage"].get("prompt_tokens", 0),
                        completion_tokens=event["usage"].get("completion_tokens", 0),
                    )

                for choice in event.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content") or ""
                    if delta:
                        safe = tracker.push(delta)
                        if safe:
                            yield StreamChunk(delta=safe)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                if tracker.stopped:
                    break

        tail = tracker.flush()
        if tail:
            yield StreamChunk(delta=tail)
        yield StreamChunk(finish_reason=finish_reason, usage=usage)

    async def list_remote_models(self) -> list[str]:
        """Model ids the upstream server currently has loaded."""
        if not self._loaded:
            await self.load()
        resp = await self._client.get("/models")
        resp.raise_for_status()
        return [m.get("id", "") for m in resp.json().get("data", [])]
