"""Canonical data types passed between every layer of Kumaru.

Design rule
-----------
These types are the *lingua franca*. The UI, server, agent, backends, and
training data loaders all speak in `Message` objects. Because every layer agrees
on one shape, you can replace any single layer without touching the others —
that is the mechanical reason this system stays extensible as it grows.

They are Pydantic models (not plain dataclasses) because they cross a network
boundary (the HTTP API) and a disk boundary (JSONL training data), so we get
validation and (de)serialization for free at both edges.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator

FinishReason = Literal["stop", "length", "error", "cancelled"]


class Role(str, Enum):
    """Who produced a message.

    Mirrors the OpenAI chat schema on purpose: it is the de-facto interchange
    format, so tools, datasets, and client libraries in the wider ecosystem work
    with Kumaru unmodified.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """One turn in a conversation."""

    role: Role
    content: str = ""
    # Populated only when role == TOOL, or when an assistant turn requests a
    # tool call. Unused while Kumaru is chat-only; present so the tool
    # layer can be switched on later without a schema migration.
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    # The model's private chain of thought ("thinking"), kept apart from the
    # answer in `content`. Reasoning models (Qwen3.x, DeepSeek-R1, ...) spend
    # tokens here before answering; that spend is where most of their accuracy
    # gain on hard questions comes from. Accepts the OpenAI-ecosystem name
    # `reasoning_content` as well.
    reasoning: str | None = Field(
        default=None, validation_alias=AliasChoices("reasoning", "reasoning_content")
    )

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_none_content(cls, v: Any) -> str:
        # Some clients send `"content": null` on assistant tool-call turns.
        return "" if v is None else v

    def to_openai(self, *, include_reasoning: bool = False) -> dict[str, Any]:
        """Render in the wire format that inference servers expect.

        Reasoning is omitted by default: model authors (Qwen included) train on
        histories that contain only final answers, and old thinking would burn
        context for no benefit. Pass `include_reasoning=True` for the rare case
        that wants it (agentic "interleaved thinking" across tool calls).
        """
        out: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            out["name"] = self.name
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        if include_reasoning and self.reasoning:
            out["reasoning_content"] = self.reasoning
        return out

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str, reasoning: str | None = None) -> Message:
        return cls(role=Role.ASSISTANT, content=content, reasoning=reasoning)


class GenerationParams(BaseModel):
    """Decoding knobs, normalized across every backend.

    Each backend translates these into its own native arguments. One normalized
    struct means a config written for the GPU backend still makes sense when you
    switch to llama.cpp or a remote vLLM server.
    """

    max_tokens: int = Field(default=1024, ge=1, le=131_072)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.05, ge=0.5, le=2.0)
    # Optional knobs: None means "do not send; let the engine use its default".
    # min_p drops tokens whose probability is below min_p * p(top token).
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    # Flat penalty on any token already present in the output (OpenAI semantics).
    # Model authors recommend it for quantized reasoning models to curb loops.
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None
    # Reasoning ("thinking") control for hybrid models such as Qwen3.x.
    # None = the model's own default; True/False forces it on/off.
    thinking: bool | None = None
    # Hard cap on thinking tokens. When reached, the engine injects a short
    # "time is up" message and forces the answer. None = unlimited.
    thinking_budget: int | None = Field(default=None, ge=0)

    def merged(self, **overrides: Any) -> GenerationParams:
        """Return a copy with non-None overrides applied.

        Layers request-level params on top of config-level defaults without
        mutating shared state.
        """
        clean = {k: v for k, v in overrides.items() if v is not None}
        return self.model_copy(update=clean)


class Usage(BaseModel):
    """Token accounting. The unit of cost, latency, and capacity planning."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class StreamChunk(BaseModel):
    """One incremental piece of a streaming response.

    `delta` is the newly generated text only — never the accumulated string.
    Consumers append. This keeps bandwidth O(output) instead of O(output^2).
    `reasoning` is the same idea for the model's thinking stream; a chunk can
    carry either or both.
    """

    delta: str = ""
    reasoning: str = ""
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


class ChatResult(BaseModel):
    """A completed (non-streaming) generation."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "kumaru"
    message: Message
    finish_reason: FinishReason = "stop"
    usage: Usage = Field(default_factory=Usage)
    # Wall-clock timings, always measured. You cannot make scaling decisions
    # about a system you are not instrumenting.
    latency_ms: float = 0.0
    time_to_first_token_ms: float | None = None

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms <= 0:
            return 0.0
        return self.usage.completion_tokens / (self.latency_ms / 1000.0)


class BackendInfo(BaseModel):
    """Self-description of a loaded backend, surfaced at /v1/models and in the UI."""

    name: str
    model_id: str
    device: str = "unknown"
    dtype: str = "unknown"
    context_length: int = 0
    supports_streaming: bool = True
    quantization: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
