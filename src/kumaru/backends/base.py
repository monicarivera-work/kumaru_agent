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
import contextlib
import json
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Iterator, Sequence
from typing import Any, TypeVar

from kumaru.core.config import BackendConfig
from kumaru.core.errors import ConfigError
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
) -> AsyncGenerator[Any, None]:
    """Run a *blocking* iterator on a worker thread and yield its items async.

    `make_iterator` is a zero-argument callable returning an iterable. It is
    invoked on the worker thread, so even expensive setup inside it (tokenizing
    a long prompt, moving tensors to the GPU) never blocks the event loop.

    Backpressure is real: the bounded queue means a fast GPU cannot run
    thousands of tokens ahead of a slow HTTP client and exhaust memory.

    Abandoning the stream stops the work. When the consumer stops early (the
    user pressed Stop, the client hung up, a caller broke out of its loop), the
    worker notices at its next item, stops pulling, and closes the iterator *on
    the worker thread*, so a generator's ``finally`` block runs there. That is
    where a backend halts ``model.generate`` or releases its model lock, so
    generation ends about a token later instead of running on to
    ``max_tokens``. A producer's exception reaches the consumer only if the
    consumer is still listening.
    """
    loop = asyncio.get_running_loop()
    items: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
    error: list[BaseException] = []
    abandoned = threading.Event()

    def put(item: Any) -> bool:
        """Queue `item`, waiting for room; False once the consumer has gone."""
        while not abandoned.is_set():
            try:
                items.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def worker() -> None:
        iterator: Iterator[Any] | None = None
        try:
            iterator = iter(make_iterator())
            for item in iterator:
                if not put(item):
                    break
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            error.append(exc)
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001 - nobody is left to receive it
                    logger.warning("closing an abandoned generation failed", exc_info=True)
            if not put(_SENTINEL):
                # An executor thread may still be blocked in items.get() for a
                # consumer that was cancelled mid-wait; this releases it.
                with contextlib.suppress(queue.Full):
                    items.put_nowait(_SENTINEL)

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
        abandoned.set()
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
    #: True if the backend attaches `backend.adapter_path` (a LoRA adapter from
    #: `kumaru train`). `load()` refuses an adapter everywhere else, because a
    #: backend that ignored one would quietly serve, or benchmark, the base model.
    supports_adapters: bool = False
    #: True if `stream(..., tools=...)` sends tool declarations to the model and
    #: yields the calls it makes. See src/kumaru/agent/tools.py.
    supports_tools: bool = False

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._loaded = False
        self._load_lock: asyncio.Lock | None = None

    # -- lifecycle ----------------------------------------------------------

    async def load(self) -> None:
        """Prepare the backend for inference. Idempotent and concurrency-safe.

        The lock matters for process-backed engines: four requests arriving
        before the first load finishes must not start four model servers.
        """
        if self._loaded:
            return
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        async with self._load_lock:
            if self._loaded:
                return
            if self.config.adapter_path and not self.supports_adapters:
                raise ConfigError(
                    f"The {self.name} backend cannot attach a LoRA adapter "
                    f"(backend.adapter_path is {self.config.adapter_path!r}).",
                    hint="Use a transformers profile, such as -c default, with the adapter's base model. "
                    "GGUF models need the adapter converted first: see docs/TRAINING.md.",
                )
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
        chars = sum(
            len(m.content) + len(m.role.value) + 4
            + (len(json.dumps(m.tool_calls)) if m.tool_calls else 0)
            for m in messages
        )
        return int(chars / 3.6) + 8

    @property
    def context_length(self) -> int:
        return self.config.max_context_length

    def check_tools(self, tools: Sequence[dict[str, Any]] | None) -> None:
        """Refuse tool declarations this backend cannot pass to its model.

        A backend that dropped them silently would leave the model describing
        tool calls in prose that nothing ever runs.
        """
        if tools and not self.supports_tools:
            raise ConfigError(
                f"The {self.name} backend cannot call tools, and tools are enabled "
                "(agent.enabled_tools, or kumaru chat --workspace).",
                hint="Use a backend that can: llamacpp_server (e.g. -c qwen36) or "
                "openai_compat with a server that supports tool calls.",
            )

    # -- generation ---------------------------------------------------------

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        *,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield incremental `StreamChunk`s until generation finishes.

        The final chunk must carry a `finish_reason` and, when the backend can
        supply it, a populated `usage`.

        `tools` are OpenAI-style function declarations. A backend with
        `supports_tools` sends them to the model and puts the calls the model
        makes on the final chunk (`tool_calls`, finish reason "tool_calls");
        every other backend calls `check_tools`, which refuses them.
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
        reasoning_parts: list[str] = []
        finish_reason: Any = "stop"
        usage = Usage()

        async for chunk in self.stream(messages, params):
            if (chunk.delta or chunk.reasoning) and first_token_at is None:
                first_token_at = time.perf_counter()
            if chunk.reasoning:
                reasoning_parts.append(chunk.reasoning)
            if chunk.delta:
                parts.append(chunk.delta)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.usage:
                usage = chunk.usage

        text = "".join(parts)
        reasoning = "".join(reasoning_parts)
        elapsed_ms = (time.perf_counter() - started) * 1000

        if usage.total_tokens == 0:
            # Backend gave us no accounting; estimate so dashboards stay useful.
            usage = Usage(
                prompt_tokens=self.count_tokens(messages),
                completion_tokens=max(1, (len(text) + len(reasoning)) // 4),
            )

        return ChatResult(
            model=self.config.model,
            message=Message.assistant(text, reasoning=reasoning or None),
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


def _partial_suffix_len(text: str, tag: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `tag`."""
    for k in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class ReasoningSplitter:
    """Split a raw token stream into *reasoning* and *answer* streams.

    Engines without native reasoning parsing return the model's thinking inline,
    wrapped in tags: ``<think>\\n...\\n</think>\\n\\nThe answer``. This parser
    routes text before ``</think>`` to the reasoning channel and the rest to the
    answer channel, incrementally: tags split across chunks ("</th" + "ink>") are
    handled by holding back only the shortest ambiguous tail, so nothing is
    delayed by more than a few characters.

    An opening tag is recognised only at the very start of the output, which is
    the only place reasoning models emit one; a literal "<think>" later in an
    answer (say, in an HTML example) passes through untouched.

    ``start_in_reasoning=True`` is for chat templates that pre-open the block in
    the prompt (the generation prompt already ends in ``<think>``), so the
    output begins mid-thought and only the closing tag ever appears.
    """

    def __init__(
        self,
        *,
        start_in_reasoning: bool = False,
        open_tag: str = "<think>",
        close_tag: str = "</think>",
    ) -> None:
        self.open_tag = open_tag
        self.close_tag = close_tag
        self._state = "reasoning" if start_in_reasoning else "start"
        self._strip_leading = start_in_reasoning
        # A pre-opened block may still be re-opened by the model; tolerate it.
        self._allow_reopen = start_in_reasoning
        self._buffer = ""

    @property
    def in_reasoning(self) -> bool:
        return self._state == "reasoning"

    def push(self, delta: str) -> tuple[str, str]:
        """Feed raw text; return ``(reasoning, answer)`` that is safe to emit now."""
        self._buffer += delta
        reasoning: list[str] = []
        answer: list[str] = []

        while self._buffer:
            if self._state == "start":
                head = self._buffer.lstrip()
                if not head:
                    break  # only whitespace so far: undecided
                if head.startswith(self.open_tag):
                    self._buffer = head[len(self.open_tag) :]
                    self._state, self._strip_leading = "reasoning", True
                    continue
                if self.open_tag.startswith(head):
                    break  # could still become "<think>"
                self._state = "answer"
                continue

            if self._strip_leading:
                self._buffer = self._buffer.lstrip()
                if not self._buffer:
                    break
                self._strip_leading = False

            if self._state == "reasoning":
                if self._allow_reopen:
                    if self._buffer.startswith(self.open_tag):
                        self._buffer = self._buffer[len(self.open_tag) :]
                        self._strip_leading = True
                        continue
                    if self.open_tag.startswith(self._buffer):
                        break
                    self._allow_reopen = False
                idx = self._buffer.find(self.close_tag)
                if idx != -1:
                    reasoning.append(self._buffer[:idx])
                    self._buffer = self._buffer[idx + len(self.close_tag) :]
                    self._state, self._strip_leading = "answer", True
                    continue
                hold = _partial_suffix_len(self._buffer, self.close_tag)
                cut = len(self._buffer) - hold
                reasoning.append(self._buffer[:cut])
                self._buffer = self._buffer[cut:]
                break

            answer.append(self._buffer)
            self._buffer = ""

        return "".join(reasoning), "".join(answer)

    def flush(self) -> tuple[str, str]:
        """Release anything held back once generation has finished.

        Unterminated reasoning (the model ran out of tokens mid-thought) stays
        reasoning: it was never an answer, and showing it as one would mislead.
        """
        rest, self._buffer = self._buffer, ""
        if self._state == "reasoning":
            return rest, ""
        return "", rest.lstrip() if self._strip_leading else rest


def chunks_from_text(text: str, size: int = 4) -> Iterator[str]:
    """Split text into small pieces — used by test/demo backends to fake streaming."""
    for i in range(0, len(text), size):
        yield text[i : i + size]
