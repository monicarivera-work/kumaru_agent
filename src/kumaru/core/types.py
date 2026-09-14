"""Value types shared by every layer.

Design rationale
----------------
These are *plain, immutable data carriers*. They are intentionally not tied to
any backend SDK: a :class:`Message` produced by the browser UI, by the CLI, or
by a test fixture is the same object, so a backend never has to care where the
conversation came from.

``Message`` uses ``slots=True`` and ``frozen=True`` because a long chat holds
thousands of them; immutability also means memory trimming can share objects
instead of copying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Who produced a message.

    Subclassing ``str`` means a ``Role`` serialises to plain JSON ("user")
    without a custom encoder, while still being comparable to string literals.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of a conversation."""

    role: Role
    content: str

    def __post_init__(self) -> None:
        # Accept plain strings at the boundary, normalise to Role internally.
        if not isinstance(self.role, Role):
            object.__setattr__(self, "role", Role(str(self.role)))

    def to_dict(self) -> dict[str, str]:
        """Return the OpenAI-style ``{"role": ..., "content": ...}`` form."""
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Build a message from an OpenAI-style dict."""
        return cls(role=Role(data["role"]), content=str(data["content"]))

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(Role.USER, content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(Role.ASSISTANT, content)


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for a single generation.

    Backends that cannot report real token counts leave the fields at ``0``
    rather than guessing, so dashboards can distinguish "free" from "unknown".
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """One incremental piece of a streamed reply.

    ``done`` marks the final chunk; only the final chunk carries ``usage``.
    """

    delta: str = ""
    done: bool = False
    usage: Usage | None = None


@dataclass(slots=True)
class ChatRequest:
    """Everything a backend needs to produce one reply.

    Sampling parameters live here (not on the backend) so a single loaded model
    can serve requests with different settings without being re-created.
    """

    messages: list[Message] = field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)

    def as_dicts(self) -> list[dict[str, str]]:
        """Messages in OpenAI wire format."""
        return [m.to_dict() for m in self.messages]
