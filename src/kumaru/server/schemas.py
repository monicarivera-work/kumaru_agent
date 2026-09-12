"""Wire schemas for the HTTP API.

These are separate from `kumaru.core.types` on purpose. Core types are the
*internal* vocabulary and may be refactored freely; these are the *published
contract* that external clients depend on and must stay stable. Collapsing the
two saves a few lines today and makes every internal rename a breaking API
change forever.

The shapes mirror OpenAI's chat-completions API so existing clients work
unmodified.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kumaru.core.types import ChatResult, Message, Role


class ChatMessageIn(BaseModel):
    """One inbound message. Tolerant of what clients actually send."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = ""
    name: str | None = None
    tool_call_id: str | None = None

    def to_domain(self) -> Message:
        return Message(
            role=Role(self.role),
            content=self.content or "",
            name=self.name,
            tool_call_id=self.tool_call_id,
        )


class ChatCompletionRequest(BaseModel):
    """Inbound `/v1/chat/completions` body.

    Every generation field is `None` by default so that "absent" is
    distinguishable from "explicitly zero". Only fields the client actually set
    override the server config — otherwise a client library that helpfully sends
    `temperature=1.0` would silently stomp your tuned defaults.
    """

    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessageIn]
    stream: bool = False

    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)  # newer OpenAI alias
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    repetition_penalty: float | None = Field(default=None, ge=0.5, le=2.0)
    stop: list[str] | str | None = None
    seed: int | None = None

    def generation_overrides(self) -> dict[str, Any]:
        stop = [self.stop] if isinstance(self.stop, str) else self.stop
        return {
            "max_tokens": self.max_tokens or self.max_completion_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "stop": stop,
            "seed": self.seed,
        }

    def to_domain(self) -> list[Message]:
        return [m.to_domain() for m in self.messages]


class UsageOut(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceOut(BaseModel):
    index: int = 0
    message: dict[str, Any]
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """Non-streaming response body."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChoiceOut]
    usage: UsageOut
    # Non-standard, additive: clients that do not know about it ignore it, and
    # the UI uses it to show real tok/s instead of a guess.
    kumaru: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_result(cls, result: ChatResult) -> ChatCompletionResponse:
        return cls(
            id=result.id,
            created=result.created,
            model=result.model,
            choices=[
                ChoiceOut(
                    message={"role": "assistant", "content": result.message.content},
                    finish_reason=result.finish_reason,
                )
            ],
            usage=UsageOut(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
            ),
            kumaru={
                "latency_ms": round(result.latency_ms, 1),
                "time_to_first_token_ms": (
                    round(result.time_to_first_token_ms, 1)
                    if result.time_to_first_token_ms
                    else None
                ),
                "tokens_per_second": round(result.tokens_per_second, 2),
            },
        )


def sse_chunk(
    *,
    completion_id: str,
    model: str,
    delta: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one `chat.completion.chunk` event body."""
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": (
                    {"role": role}
                    if role
                    else ({"content": delta} if delta is not None else {})
                ),
                "finish_reason": finish_reason,
            }
        ],
    }
    if extra:
        payload["kumaru"] = extra
    return payload


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "kumaru"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    version: str
    backend: str
    model: str
    loaded: bool
    detail: str | None = None


class ErrorBody(BaseModel):
    message: str
    type: str
    hint: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
