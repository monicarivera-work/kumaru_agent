"""The contract every generation engine must satisfy.

The whole point of this file is that ``kumaru.agent`` never learns the
difference between a 7B model on your GPU, a quantised GGUF on your CPU, and a
remote OpenAI-compatible server. It only knows :class:`Backend`.

Two methods, one of which is free
---------------------------------
A backend must implement :meth:`Backend.generate`. Streaming
(:meth:`Backend.stream`) has a default implementation that yields the complete
answer as a single chunk, so a new backend is useful after one method and can
be upgraded to true token streaming later without touching callers.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass

from kumaru.core.config import BackendConfig
from kumaru.core.types import ChatRequest, StreamChunk, Usage


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """A complete, non-streamed reply."""

    text: str
    usage: Usage = Usage()


class Backend(abc.ABC):
    """Base class for anything that turns messages into text."""

    #: Registry name, set by subclasses; used in logs and ``/api/health``.
    name: str = "backend"

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    # -- required ---------------------------------------------------------
    @abc.abstractmethod
    def generate(self, request: ChatRequest) -> GenerationResult:
        """Produce a full reply. Must raise ``BackendError`` on failure."""

    # -- optional ---------------------------------------------------------
    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Yield the reply incrementally.

        The default is correct but not incremental: it blocks, then emits one
        chunk. Override it when the underlying engine supports streaming.
        """
        result = self.generate(request)
        if result.text:
            yield StreamChunk(delta=result.text)
        yield StreamChunk(done=True, usage=result.usage)

    def health(self) -> dict[str, object]:
        """Cheap liveness/identity info for ``/api/health``."""
        return {
            "backend": self.name,
            "model": self.config.model or None,
            "ready": True,
        }

    def close(self) -> None:
        """Release sockets, GPU memory, file handles. Must be idempotent."""

    def __enter__(self) -> Backend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token).

    Used only by backends that cannot report real counts, so the UI can show a
    ballpark figure. Never use this for context-window arithmetic - tokenise
    properly for that.
    """
    return max(1, len(text) // 4) if text else 0
