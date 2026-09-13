"""Pydantic models for the HTTP API.

These are the *wire* contract and are intentionally separate from
``kumaru.core.types``: the internal ``Message`` may change shape freely as long
as the translation here keeps working, so refactors never silently break a
browser that is still open.

Field constraints double as input validation - a request asking for a million
tokens is rejected by FastAPI with a 422 before it reaches a model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from kumaru.core.types import Message as CoreMessage
from kumaru.core.types import Role


class MessageModel(BaseModel):
    """One message on the wire."""

    role: Literal["system", "user", "assistant"]
    content: str

    def to_core(self) -> CoreMessage:
        return CoreMessage(role=Role(self.role), content=self.content)

    @classmethod
    def from_core(cls, message: CoreMessage) -> MessageModel:
        return cls(role=message.role.value, content=message.content)


class ChatRequestModel(BaseModel):
    """Body of ``POST /api/chat`` and ``POST /api/chat/stream``."""

    message: str = Field(min_length=1, max_length=100_000)
    session_id: str = Field(default="default", min_length=1, max_length=128)


class UsageModel(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponseModel(BaseModel):
    """Body of a successful non-streamed chat response."""

    reply: str
    session_id: str
    usage: UsageModel = UsageModel()


class HistoryResponseModel(BaseModel):
    session_id: str
    messages: list[MessageModel] = Field(default_factory=list)


class HealthModel(BaseModel):
    """``GET /api/health`` - what the UI status pill reads."""

    status: Literal["ok", "degraded"] = "ok"
    version: str
    backend: str
    model: str | None = None
    ready: bool = True
    detail: dict[str, object] = Field(default_factory=dict)


class ErrorModel(BaseModel):
    error: str
    message: str
    hint: str | None = None
