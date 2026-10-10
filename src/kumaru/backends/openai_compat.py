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

from kumaru.backends.base import Backend, ReasoningSplitter, StopSequenceTracker
from kumaru.core.config import BackendConfig
from kumaru.core.errors import BackendError
from kumaru.core.logging import get_logger
from kumaru.core.registry import BACKENDS
from kumaru.core.types import (
    BackendInfo,
    FinishReason,
    GenerationParams,
    Message,
    StreamChunk,
    Usage,
)

logger = get_logger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"  # Ollama's default

# Upstream finish reasons Kumaru's domain model does not distinguish. Anything
# unrecognised maps to "stop" rather than failing validation mid-stream.
_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "eos": "stop",
    "length": "length",
    "error": "error",
    "cancelled": "cancelled",
}


def normalize_finish_reason(value: Any) -> FinishReason:
    return _FINISH_REASONS.get(str(value), "stop")


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

    def _connection_lost(self, exc: Exception) -> BackendError:
        """The error raised when the HTTP connection fails during a request."""
        import httpx

        if isinstance(exc, httpx.ReadTimeout):
            # Connected, but silent for a whole read timeout: busy or stuck, not gone.
            return BackendError(
                f"{self._base_url} sent nothing for {self.config.timeout_s:.0f}s.",
                hint="The server may be overloaded or still reading a very long prompt. "
                "If that is expected, raise backend.timeout_s.",
            )
        return BackendError(
            f"Lost the connection to {self._base_url} ({type(exc).__name__}: {exc}).",
            hint="Check that the model server is still running and reachable.",
        )

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
        if params.presence_penalty is not None:
            payload["presence_penalty"] = params.presence_penalty  # standard OpenAI field
        if stream:
            # vLLM and newer llama.cpp honour this and send a final usage chunk;
            # servers that do not simply ignore the field.
            payload["stream_options"] = {"include_usage": True}
        if not self.strict_openai:
            payload.update(self._extended_fields(params))
        return payload

    @property
    def strict_openai(self) -> bool:
        """Send only fields the official OpenAI API accepts.

        Self-hosted engines (vLLM, SGLang, llama.cpp, Ollama) accept extra
        sampling knobs; hosted APIs reject unknown fields with a 400. Set
        ``backend.extra.strict_openai: true`` when pointing at a hosted API.
        """
        return bool(self.config.extra.get("strict_openai", False))

    def _extended_fields(self, params: GenerationParams) -> dict[str, Any]:
        """Non-standard fields understood by self-hosted engines.

        Subclasses extend this for engine-specific features; the base payload
        above stays the portable core.
        """
        fields: dict[str, Any] = {}
        if params.top_k > 0:
            fields["top_k"] = params.top_k
        if params.min_p is not None:
            fields["min_p"] = params.min_p
        if params.repetition_penalty != 1.0:
            fields["repetition_penalty"] = params.repetition_penalty
        if params.thinking is not None:
            # The Qwen3 / DeepSeek / GLM convention, honoured by vLLM, SGLang
            # and llama.cpp: a variable passed straight into the chat template.
            fields["chat_template_kwargs"] = {"enable_thinking": params.thinking}
        return fields

    async def stream(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> AsyncIterator[StreamChunk]:
        if not self._loaded:
            await self.load()

        # Stop sequences are enforced locally as well: server-side support is
        # inconsistent, and silently returning text past a stop sequence breaks
        # downstream parsing. They apply to the answer only, never to reasoning.
        tracker = StopSequenceTracker(params.stop)
        # Fallback for servers that return thinking inline as <think>...</think>
        # instead of in a separate `reasoning_content` field.
        splitter = ReasoningSplitter()
        usage = Usage()
        finish_reason: FinishReason = "stop"

        import httpx

        try:
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

                    if event.get("error"):
                        raise BackendError(f"Upstream error: {event['error']}")

                    if event.get("usage"):
                        usage = Usage(
                            prompt_tokens=event["usage"].get("prompt_tokens", 0),
                            completion_tokens=event["usage"].get("completion_tokens", 0),
                        )

                    for choice in event.get("choices", []):
                        delta = choice.get("delta") or {}
                        # vLLM/llama.cpp/DeepSeek use `reasoning_content`; Ollama and
                        # OpenRouter use `reasoning`.
                        thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        text = delta.get("content") or ""
                        if text:
                            inline_thought, text = splitter.push(text)
                            thought += inline_thought
                        safe = tracker.push(text) if text else ""
                        if thought or safe:
                            yield StreamChunk(delta=safe, reasoning=thought)
                        if choice.get("finish_reason"):
                            finish_reason = normalize_finish_reason(choice["finish_reason"])
                    if tracker.stopped:
                        break
        except httpx.TransportError as exc:
            # Refused, reset or timed out: the server is gone or stuck. Say so in
            # Kumaru's terms (with a hint) instead of leaking an httpx exception.
            raise self._connection_lost(exc) from exc

        thought, text = splitter.flush()
        tail = (tracker.push(text) if text else "") + tracker.flush()
        if thought or tail:
            yield StreamChunk(delta=tail, reasoning=thought)
        yield StreamChunk(finish_reason=finish_reason, usage=usage)

    async def list_remote_models(self) -> list[str]:
        """Model ids the upstream server currently has loaded."""
        if not self._loaded:
            await self.load()
        resp = await self._client.get("/models")
        resp.raise_for_status()
        return [m.get("id", "") for m in resp.json().get("data", [])]
