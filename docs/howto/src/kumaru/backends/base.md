# `src\kumaru\backends\base.py`

> Defines Kumaru's async inference backend contract and shared streaming helpers.

**Read this when:** you are writing a backend, debugging streaming, checking token accounting, or adapting a blocking model engine to Kumaru's async server.

---

## What it does

`base.py` defines the `Backend` abstract base class used by all inference engines. It makes `stream()` the primitive operation and derives `generate()` from it, so streaming and non-streaming paths cannot drift apart. It also provides a thread bridge for blocking token generators, stop-sequence handling that works across chunk boundaries, and a small chunking helper for fake/demo backends.

## Why it exists

Kumaru can switch among transformers, llama.cpp, OpenAI-compatible servers, and echo without changing the agent, server, or UI. Without this contract, every caller would need backend-specific branching for loading, streaming, cleanup, timing, stop sequences, and usage accounting. The async interface also prevents a blocking `model.generate()` call from freezing the HTTP event loop.

## Mental model

1. Implement `stream()`; inherited `generate()` drains it and records latency, time-to-first-token, and usage.
2. Keep `__init__()` cheap. Put weights, sockets, downloads, and GPU allocation in `_load()`.
3. The final stream chunk must contain `finish_reason`; include `Usage` when the backend can provide it.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `iterate_in_thread` | `async def iterate_in_thread(make_iterator: Any, *, queue_size: int = 256) -> AsyncIterator[Any]` | Runs a blocking iterable factory on a daemon thread with bounded queue backpressure. |
| `Backend` | `class Backend(config: BackendConfig)` | Abstract inference engine. |
| `Backend.name` | `str = "base"` | Registry/logging name; subclasses override it. |
| `Backend.load` | `async def load(self) -> None` | Idempotently calls `_load()`, times it, logs readiness, and marks loaded. |
| `Backend._load` | `async def _load(self) -> None` | Abstract hook for expensive subclass setup. |
| `Backend.aclose` | `async def aclose(self) -> None` | Releases resources and marks unloaded; subclasses override as needed. |
| `Backend.is_loaded` | `property -> bool` | Reports lifecycle state. |
| `Backend.info` | `def info(self) -> BackendInfo` | Abstract model/backend self-description surfaced to API/UI. |
| `Backend.count_tokens` | `def count_tokens(self, messages: Sequence[Message]) -> int` | Default rough token estimate: about one token per 3.6 characters plus overhead. |
| `Backend.context_length` | `property -> int` | Returns `config.max_context_length`. |
| `Backend.stream` | `def stream(self, messages: Sequence[Message], params: GenerationParams) -> AsyncIterator[StreamChunk]` | Abstract async stream of response deltas. |
| `Backend.generate` | `async def generate(self, messages: Sequence[Message], params: GenerationParams) -> ChatResult` | Non-streaming generation derived from `stream()`. |
| `Backend.apply_stop_sequences` | `staticmethod def apply_stop_sequences(text: str, stop: Iterable[str]) -> tuple[str, bool]` | Truncates complete text at the earliest stop sequence. |
| `StopSequenceTracker` | `class StopSequenceTracker(stop: Sequence[str])` | Streaming-safe stop detector. |
| `StopSequenceTracker.push` | `def push(self, delta: str) -> str` | Adds a delta and returns text safe to emit now. |
| `StopSequenceTracker.flush` | `def flush(self) -> str` | Emits any buffered tail unless a stop was hit. |
| `chunks_from_text` | `def chunks_from_text(text: str, size: int = 4) -> Iterator[str]` | Splits text into small chunks for fake streaming. |

## How to use it

### Generate a non-streaming result

```python
from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS
from kumaru.core.types import Message

cfg = load_config("test")
backend = BACKENDS.create(cfg.backend.name, cfg.backend)
await backend.load()
try:
    result = await backend.generate([Message.user("hello")], cfg.generation)
    print(result.message.content)
    print(result.latency_ms, result.time_to_first_token_ms)
finally:
    await backend.aclose()
```

### Wrap a blocking producer

```python
from kumaru.backends.base import iterate_in_thread
from kumaru.core.types import StreamChunk

async def stream_tokens(sync_model, prompt: str):
    def produce():
        for token in sync_model.generate_tokens(prompt):
            yield StreamChunk(delta=token)
        yield StreamChunk(finish_reason="stop")

    async for chunk in iterate_in_thread(produce, queue_size=64):
        yield chunk
```

### Enforce stop sequences across chunks

```python
from kumaru.backends.base import StopSequenceTracker

tracker = StopSequenceTracker(["</answer>"])
for delta in ["hel", "lo</", "answer>ignored"]:
    safe = tracker.push(delta)
    if safe:
        print(safe, end="")
    if tracker.stopped:
        break
print(tracker.flush(), end="")
```

### Exercise the contract with real profiles

```powershell
kumaru chat -c test -q "Does streaming work?"
kumaru serve -c default
kumaru config -c low-vram
```

## Tuning / operational notes

| Knob | Where | Trade-off |
|---|---|---|
| `queue_size` | `iterate_in_thread(..., queue_size=256)` | Higher lets a fast producer run farther ahead; lower applies backpressure earlier for slow clients. |
| `backend.max_context_length` | `Backend.context_length` / config | Hard context cap for memory trimming and backend prompt budgets. |
| `agent.max_history_tokens` | config | Must leave room for `generation.max_tokens`; default 6144 history + 1024 output fits under 8192 context. |
| Final `StreamChunk.usage` | backend stream end | Accurate usage improves dashboards; if absent, `generate()` estimates prompt and completion tokens. |
| Stop string length | `StopSequenceTracker` | The tracker holds back up to `max_stop_length - 1` characters to catch split stop sequences. |

Expected behavior: one `generate()` call produces a `ChatResult` with total latency, optional time-to-first-token, finish reason, and usage. Backends with real tokenizers should override `count_tokens()`; remote or fake backends can use the base heuristic.

## How to extend it

A complete minimal backend can live in `src\kumaru\backends\reverse_backend.py`:

```python
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from kumaru.backends.base import Backend, chunks_from_text
from kumaru.core.config import BackendConfig
from kumaru.core.registry import BACKENDS
from kumaru.core.types import BackendInfo, GenerationParams, Message, StreamChunk, Usage

@BACKENDS.register("reverse")
class ReverseBackend(Backend):
    name = "reverse"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)

    async def _load(self) -> None:
        return None

    def info(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            model_id=self.config.model,
            device="cpu",
            dtype="none",
            context_length=self.config.max_context_length,
            supports_streaming=True,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
    ) -> AsyncIterator[StreamChunk]:
        text = next((m.content for m in reversed(messages) if m.role.value == "user"), "")
        reply = text[::-1][: params.max_tokens]
        for piece in chunks_from_text(reply, size=8):
            yield StreamChunk(delta=piece)
        yield StreamChunk(
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=self.count_tokens(messages),
                completion_tokens=max(1, len(reply) // 4),
            ),
        )
```

Register it lazily in `src\kumaru\core\registry.py` if importing it is expensive:

```python
BACKENDS.register_lazy("reverse", "kumaru.backends.reverse_backend:ReverseBackend")
```

Use it from YAML:

```yaml
backend:
  name: reverse
  model: reverse-demo
  max_context_length: 8192

generation:
  max_tokens: 128
```

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Server freezes during generation | Backend calls a blocking generator on the event loop. | Wrap it with `iterate_in_thread()`. |
| Stop sequence appears in output | Backend checks only per chunk and misses cross-chunk stops. | Use `StopSequenceTracker(params.stop)`. |
| Non-streaming results have rough usage | Final stream chunk omitted `Usage`. | Emit `StreamChunk(finish_reason=..., usage=Usage(...))`. |
| Prompt rejected as too long | Prompt tokens exceed backend context budget. | Lower `agent.max_history_tokens` or `generation.max_tokens`, or raise supported context. |
| `/v1/models` shows stale data | `info()` does not reflect loaded model/device/config. | Populate `BackendInfo` after load with actual backend state. |
| Memory grows for slow clients | Producer runs too far ahead of consumer. | Keep the bounded queue; lower `queue_size` for high-concurrency services. |

## Related files

- `src\kumaru\core\types.py` — canonical `Message`, `GenerationParams`, `StreamChunk`, `Usage`, `ChatResult`, and `BackendInfo`.
- `src\kumaru\core\config.py` — `BackendConfig` and generation defaults consumed here.
- `src\kumaru\core\registry.py` — resolves backend names to `Backend` subclasses.
- `src\kumaru\backends\echo.py` — simplest concrete backend.
- `src\kumaru\backends\transformers_backend.py` — local Hugging Face implementation.
- `src\kumaru\backends\llamacpp_backend.py` — GGUF/llama.cpp implementation.
- `src\kumaru\backends\openai_compat.py` — remote HTTP implementation.
