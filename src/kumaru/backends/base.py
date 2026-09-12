"""The `Backend` contract that every inference engine implements.

Two design decisions carry most of the weight here.

**1. Streaming is the primitive; batch generation is derived.**
`stream()` is abstract, `generate()` is a concrete method that simply drains the
stream. A new backend therefore implements *one* generation method, not two that
can drift apart. It also means every backend gets low time-to-first-token
behaviour in the UI for free, which is the single largest perceived-speed win in
a chat application.

**2. The interface is async, even though torch is not.**
The server must serve many connections from one process; a blocking
`model.generate()` call would freeze all of them. `iterate_in_thread()` below is
the bridge: it runs a blocking generator on a worker thread and hands items back
to the event loop. Async at the interface keeps the door open to genuinely
concurrent engines (vLLM, TGI) without changing a single caller.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from typing import Any, TypeVar

from kumaru.core.config import BackendConfig
from kumaru.core.logging import get_logger
from kumaru.core.types import (
    BackendInfo,
    ChatResult,
    GenerationParams,
    Message,
    StreamChunk,
    Usage,
)

logger = get_logger(__name__)
T = TypeVar("T")

_SENTINEL = object()


async def iterate_in_thread(
    make_iterator: Any,
    *,
    queue_size: int = 256,
) -> AsyncIterator[Any]:
    """Run a *blocking* iterator on a worker thread and yield its items async.

    `make_iterator` is a zero-argument callable returning an iterable. It is
    invoked on the worker thread, so even expensive setup inside it (tokenizing
    a long prompt, moving tensors to the GPU) never blocks the event loop.

    Backpressure is real: the bounded queue means a fast GPU cannot run
    thousands of tokens ahead of a slow HTTP client and exhaust memory.
    """
    loop = asyncio.get_running_loop()
    items: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
    error: list[BaseException] = []

    def worker() -> None:
        try:
            for item in make_iterator():
                items.put(item)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            error.append(exc)
        finally:
            items.put(_SENTINEL)

    thread = threading.Thread(target=worker, name="kumaru-generate", daemon=True)
    thread.start()

    try:
        while True:
            # `queue.get` blocks, so it must run in the default executor rather
            # than on the event loop thread.
            item = await loop.run_in_executor(None, items.get)
            if item is _SENTINEL:
                break
            yield item
    finally:
        if error:
            raise error[0]


class Backend(ABC):
    """Abstract inference engine.

    Lifecycle::

        backend = SomeBackend(config)
        await backend.load()          # idempotent; heavy work happens here
        async for chunk in backend.stream(messages, params): ...
        await backend.aclose()

    Construction must stay cheap and side-effect free. All expensive work
    (downloading weights, allocating VRAM, opening sockets) belongs in `load()`,
    so that `kumaru config` and `--help` never touch a GPU.
    """

    #: Registry name, set by subclasses. Used in logs and `/v1/models`.
    name: str = "base"

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._loaded = False

    # -- lifecycle ----------------------------------------------------------

    async def load(self) -> None:
        """Prepare the backend for inference. Idempotent."""
        if self._loaded:
            return
        started = time.perf_counter()
        await self._load()
        self._loaded = True
        logger.info(
            "backend '%s' ready: %s (%.1fs)",
            self.name,
            self.config.model,
            time.perf_counter() - started,
        )

    @abstractmethod
    async def _load(self) -> None:
        """Subclass hook: do the actual loading."""

    async def aclose(self) -> None:
        """Release sockets, file handles, and device memory. Idempotent."""
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # -- introspection ------------------------------------------------------

    @abstractmethod
    def info(self) -> BackendInfo:
        """Describe the loaded model. Surfaced at `/v1/models` and in the UI."""

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Token count for a conversation.

        The default is a deliberately crude character heuristic (~3.6 chars per
        token for English). Backends with a real tokenizer override this. The
        heuristic exists so remote backends — which cannot tokenize locally —
        still produce a usable number for memory-trimming decisions.
        """
        chars = sum(len(m.content) + len(m.role.value) + 4 for m in messages)
        return int(chars / 3.6) + 8

    @property
    def context_length(self) -> int:
        return self.config.max_context_length

    # -- generation ---------------------------------------------------------

    @abstractmethod
    def stream(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> AsyncIterator[StreamChunk]:
        """Yield incremental `StreamChunk`s until generation finishes.

        The final chunk must carry a `finish_reason` and, when the backend can
        supply it, a populated `usage`.
        """

    async def generate(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> ChatResult:
        """Non-streaming generation, derived from `stream()`.

        Concrete on purpose: subclasses get correct timing instrumentation and
        usage accounting without reimplementing it (and without it drifting out
        of sync with the streaming path).
        """
        started = time.perf_counter()
        first_token_at: float | None = None
        parts: list[str] = []
        finish_reason: Any = "stop"
        usage = Usage()

        async for chunk in self.stream(messages, params):
            if chunk.delta:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                parts.append(chunk.delta)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.usage:
                usage = chunk.usage

        text = "".join(parts)
        elapsed_ms = (time.perf_counter() - started) * 1000

        if usage.total_tokens == 0:
            # Backend gave us no accounting; estimate so dashboards stay useful.
            usage = Usage(
                prompt_tokens=self.count_tokens(messages),
                completion_tokens=max(1, len(text) // 4),
            )

        return ChatResult(
            model=self.config.model,
            message=Message.assistant(text),
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=elapsed_ms,
            time_to_first_token_ms=(
                (first_token_at - started) * 1000 if first_token_at else None
            ),
        )

    # -- helpers for subclasses --------------------------------------------

    @staticmethod
    def apply_stop_sequences(text: str, stop: Iterable[str]) -> tuple[str, bool]:
        """Truncate `text` at the earliest stop sequence.

        Returns `(text, hit)`. Needed because several engines only support stop
        sequences approximately, or not at all, so we enforce them uniformly at
        this layer rather than trusting each engine.
        """
        earliest = len(text)
        hit = False
        for s in stop:
            if not s:
                continue
            idx = text.find(s)
            if idx != -1 and idx < earliest:
                earliest, hit = idx, True
        return (text[:earliest], True) if hit else (text, False)

    def __repr__(self) -> str:
        state = "loaded" if self._loaded else "unloaded"
        return f"<{type(self).__name__} model={self.config.model!r} {state}>"


class StopSequenceTracker:
    """Streaming-safe stop-sequence detection.

    A stop sequence can straddle two chunks ("</" then "s>"), so naive per-chunk
    matching misses it. This buffers the smallest suffix that could still become
    a stop sequence and releases everything else immediately — correct detection
    with no added latency.
    """

    def __init__(self, stop: Sequence[str]) -> None:
        self.stop = [s for s in stop if s]
        self.max_len = max((len(s) for s in self.stop), default=0)
        self._buffer = ""
        self.stopped = False

    def push(self, delta: str) -> str:
        """Feed a new delta; return the text that is safe to emit now."""
        if not self.stop or self.stopped:
            return delta

        self._buffer += delta
        for s in self.stop:
            idx = self._buffer.find(s)
            if idx != -1:
                self.stopped = True
                out, self._buffer = self._buffer[:idx], ""
                return out

        # Hold back a possible partial match at the tail.
        hold = self.max_len - 1
        if hold <= 0 or len(self._buffer) <= hold:
            return ""
        out, self._buffer = self._buffer[:-hold], self._buffer[-hold:]
        return out

    def flush(self) -> str:
        """Emit whatever remains once generation has finished."""
        out, self._buffer = self._buffer, ""
        return "" if self.stopped else out


def chunks_from_text(text: str, size: int = 4) -> Iterator[str]:
    """Split text into small pieces — used by test/demo backends to fake streaming."""
    for i in range(0, len(text), size):
        yield text[i : i + size]
