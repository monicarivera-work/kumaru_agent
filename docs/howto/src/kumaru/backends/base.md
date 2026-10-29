# `src\kumaru\backends\base.py`

> Defines Kumaru's async inference backend contract and the shared streaming helpers: thread bridge, stop sequences, and the reasoning/answer splitter.

**Read this when:**
- you are writing a backend;
- you are debugging streaming or thinking output;
- you are checking token accounting;
- you are adapting a blocking model engine to Kumaru's async server.

---

## What it does

`base.py` defines the `Backend` abstract base class used by all inference engines. `stream()` is the primitive operation, and `generate()` is derived from it, so streaming and non-streaming paths cannot drift apart. Shared helpers:
- `iterate_in_thread()` — a thread bridge for blocking token generators;
- `StopSequenceTracker` — stop sequences that work across chunk boundaries;
- `ReasoningSplitter` — new in v0.2; splits raw `<think>…</think>` output into a reasoning stream and an answer stream, incrementally;
- `chunks_from_text()` — fake streaming for test and demo backends.

v0.2 added a second channel to the contract. A `StreamChunk` carries `delta` (answer text), `reasoning` (thinking text), or both. `generate()` collects them into `Message.content` and `Message.reasoning`.

Tool calling is an optional part of the contract. `stream(messages, params, tools=[...])` passes OpenAI-style function declarations; a backend that can send them to its model sets `supports_tools = True` and reports the model's calls on its final chunk. Every other backend refuses them with `check_tools()`. `ChatAgent` runs the calls (`agent\chat.md`, "The tool loop"); backends never run tools.

## Why it exists

Kumaru can switch among transformers, llama.cpp (in-process or as a managed `llama-server`), OpenAI-compatible servers and echo without changing the agent, server or UI. Without this contract, every caller would need backend-specific branching for loading, streaming, cleanup, timing, stop sequences and usage accounting. The async interface also prevents a blocking `model.generate()` call from freezing the HTTP event loop.

Reasoning models deliver their thinking in two incompatible ways:
- in a separate field (`reasoning_content` from llama-server, vLLM and DeepSeek; `reasoning` from Ollama and OpenRouter);
- inline as `<think>…</think>` in the raw text (llama-cpp-python, transformers).

Every backend normalises both into `StreamChunk.reasoning`. Everything above therefore handles one shape: SSE `delta.reasoning_content`, the dim text in `kumaru chat`, and the UI's thinking panel and `thinking_ms` timing.

## Mental model

1. Implement `stream()`. The inherited `generate()` drains it and records latency, time to first token and usage. The first `delta` *or* `reasoning` counts as the first token.
2. Keep `__init__()` cheap. Weights, sockets, downloads, child processes and GPU allocation belong in `_load()`. `load()` is idempotent and serialised by an `asyncio.Lock`, so four requests that arrive before the first load finishes start one engine, not four.
3. `delta` and `reasoning` are increments, never accumulated text. A chunk can carry both.
4. The final chunk carries `finish_reason`, and `usage` when the backend has it. The streaming server builds its final SSE frame (`latency_ms`, `completion_tokens`, `tokens_per_second`, `thinking_ms`) the moment it sees `finish_reason`, so usage sent after that chunk is missing from the stats.
5. Adapters are opt-in. `supports_adapters` (default `False`) declares that `_load()` attaches `backend.adapter_path`. On every other backend, `load()` refuses an adapter with `ConfigError` before `_load()` runs: a backend that ignored it would quietly serve, or benchmark, the base model under the adapter's name. Only `transformers` sets it.
6. Tool calling is opt-in too. `supports_tools` (default `False`) declares that `stream(..., tools=...)` sends the declarations to the model and reports the calls it makes, all on the final chunk: `StreamChunk(finish_reason="tool_calls", tool_calls=[...], usage=...)`. A backend without it calls `check_tools(tools)` first thing in `stream()`, and `ChatAgent` calls it when it is built, so enabled tools fail with `ConfigError` before the model loads instead of being silently dropped. `echo` (scripted calls) and `openai_compat`, and so `llamacpp_server`, set it.

Reasoning invariants. The shipped backends keep them; a new backend must too.
- **Stops apply to the answer only.** Feed only answer text into `StopSequenceTracker`. A stop string inside the reasoning must not end the reply. The shipped backends keep this locally, but they also forward `stop` to engines that match it on raw output, reasoning included; see Failure modes.
- **Reasoning tokens count as completion tokens.** The model generated them; they cost time and context.
- **Reasoning never enters history.** `Message.to_openai()` drops `reasoning` unless `include_reasoning=True`. Old thoughts would burn context, and chat templates expect answer-only history.
- **Unterminated reasoning stays reasoning.** If the model runs out of tokens mid-thought, the answer is empty. A half-thought is never promoted to an answer.
- **No thinking capability flag.** A backend supports thinking by honouring `params.thinking` and `params.thinking_budget` and emitting `reasoning`. A backend that cannot control thinking ignores those fields; it does not raise.

Thinking support in the shipped backends:

| Backend | `thinking` toggle | `thinking_budget` | Reasoning source |
|---|---|---|---|
| `llamacpp_server` | `chat_template_kwargs.enable_thinking` | `reasoning_budget_tokens` | `reasoning_content` field |
| `openai_compat` | `chat_template_kwargs.enable_thinking` (not with `strict_openai`) | ignored | `reasoning_content` / `reasoning` field, or inline tags |
| `transformers` | template variable `enable_thinking` | ignored | inline tags, via `ReasoningSplitter` |
| `llamacpp` | ignored | ignored | inline tags, via `ReasoningSplitter` |
| `echo` | emulated | emulated (budget = number of 8-char pieces) | synthetic |

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `iterate_in_thread` | `async def iterate_in_thread(make_iterator: Any, *, queue_size: int = 256) -> AsyncGenerator[Any, None]` | Calls the zero-argument factory on a daemon thread named `kumaru-generate` and yields its items to async callers. The bounded queue gives backpressure. If the reader leaves early, the worker stops at the next item and closes the producer on its own thread, so the producer's `finally` runs there; see [Wrap a blocking producer](#wrap-a-blocking-producer). A producer exception is re-raised in the consumer while it is still reading. The return type is an async *generator*, so it has the `aclose()` that `contextlib.aclosing` needs. |
| `Backend` | `class Backend(config: BackendConfig)` | Abstract inference engine. `__init__` only stores `self.config` and lifecycle state. |
| `Backend.name` | `name: str = "base"` | Registry/logging name; subclasses override it. |
| `Backend.supports_adapters` | `supports_adapters: bool = False` | Class attribute. `True` only if `_load()` attaches `backend.adapter_path`; only `TransformersBackend` sets it. |
| `Backend.supports_tools` | `supports_tools: bool = False` | Class attribute. `True` only if `stream(..., tools=...)` sends the declarations to the model and yields its calls; `EchoBackend` and `OpenAICompatBackend` (so `LlamaServerBackend`) set it. |
| `Backend.check_tools` | `def check_tools(self, tools: Sequence[dict[str, Any]] \| None) -> None` | Raises `ConfigError` (`The <name> backend cannot call tools, and tools are enabled (agent.enabled_tools, or kumaru chat --workspace).`) when `tools` is non-empty and `supports_tools` is `False`; otherwise does nothing. |
| `Backend.load` | `async def load(self) -> None` | Idempotent and lock-serialised. Raises `ConfigError` if `adapter_path` is set and `supports_adapters` is `False`; otherwise calls `_load()` once and logs `backend '<name>' ready: <model> (<s>s)`. |
| `Backend._load` | `async def _load(self) -> None` (abstract) | Hook for expensive subclass setup. |
| `Backend.aclose` | `async def aclose(self) -> None` | Base version only marks the backend unloaded. Subclasses release resources and must reset `_loaded` too. |
| `Backend.is_loaded` | `property -> bool` | Lifecycle state. |
| `Backend.info` | `def info(self) -> BackendInfo` (abstract) | Self-description surfaced at `/v1/models`, `/v1/info` and in the UI. |
| `Backend.count_tokens` | `def count_tokens(self, messages: Sequence[Message]) -> int` | Default heuristic: `int(Σ(len(content) + len(role) + 4 + len(json.dumps(tool_calls))) / 3.6) + 8`, where `tool_calls` counts only when a message has them. Override with a real tokenizer. |
| `Backend.context_length` | `property -> int` | Returns `config.max_context_length`. The agent sizes the history budget from it. |
| `Backend.stream` | `def stream(self, messages: Sequence[Message], params: GenerationParams, *, tools: Sequence[dict[str, Any]] \| None = None) -> AsyncIterator[StreamChunk]` (abstract) | Implement as an `async def` generator that yields chunks. `tools` are OpenAI-style function declarations; see `supports_tools`. |
| `Backend.generate` | `async def generate(self, messages: Sequence[Message], params: GenerationParams) -> ChatResult` | Drains `stream()`. Returns `message=Message.assistant(text, reasoning=reasoning or None)` plus `finish_reason`, `usage`, `latency_ms` and `time_to_first_token_ms`. If the stream reported no usage, it estimates: prompt via `count_tokens`, completion `max(1, (len(text) + len(reasoning)) // 4)`. |
| `Backend.apply_stop_sequences` | `@staticmethod apply_stop_sequences(text: str, stop: Iterable[str]) -> tuple[str, bool]` | Truncates complete text at the earliest stop. Returns `(text, hit)`. |
| `StopSequenceTracker` | `class StopSequenceTracker(stop: Sequence[str])` | Streaming-safe stop detector. Empty strings are ignored. Public attributes: `stop`, `max_len`, `stopped`. |
| `StopSequenceTracker.push` | `def push(self, delta: str) -> str` | Adds a delta and returns the text that is safe to emit now. On a match it sets `stopped` and returns only the text before the stop. |
| `StopSequenceTracker.flush` | `def flush(self) -> str` | Emits the held-back tail, or `""` if a stop was hit. |
| `ReasoningSplitter` | `class ReasoningSplitter(*, start_in_reasoning: bool = False, open_tag: str = "<think>", close_tag: str = "</think>")` | Incremental parser for engines that return thinking inline. |
| `ReasoningSplitter.in_reasoning` | `property -> bool` | `True` while inside the thinking block. |
| `ReasoningSplitter.push` | `def push(self, delta: str) -> tuple[str, str]` | Feeds raw text and returns `(reasoning, answer)` that is safe to emit now. |
| `ReasoningSplitter.flush` | `def flush(self) -> tuple[str, str]` | Releases held-back text at the end. Unterminated reasoning stays reasoning. |
| `chunks_from_text` | `def chunks_from_text(text: str, size: int = 4) -> Iterator[str]` | Splits text into fixed-size pieces for fake streaming. |

`normalize_finish_reason(value: Any) -> FinishReason` lives in `kumaru.backends.openai_compat`, not here. It maps `stop`/`eos` to `stop`, `tool_calls` and the older `function_call` to `tool_calls`, and keeps `length`, `error` and `cancelled`; anything else (`None` included) becomes `stop`. `llamacpp_backend` imports it from there. Use it for any engine whose finish reasons fall outside `StreamChunk.finish_reason`'s `Literal["stop", "length", "tool_calls", "error", "cancelled"]`; any other value fails validation mid-stream.

### `ReasoningSplitter` rules

- **The open tag counts only at the very start of the output**, after leading whitespace. That is the only place reasoning models emit it. A literal `<think>` later in an answer, say in an HTML example, passes through as answer text.
- **Tags split across chunks are handled** by holding back only the ambiguous tail (`"</th"`), so nothing is delayed by more than a few characters. Output that is so far only whitespace, or a prefix of `<think>`, is held until it is decided.
- **Whitespace right after `<think>` and right after `</think>` is stripped**, so answers do not start with `\n\n`.
- **`start_in_reasoning=True` is for chat templates whose generation prompt already ends in `<think>`.** The output then starts mid-thought and only `</think>` appears. A re-opened `<think>` at the very start is tolerated. The transformers backend derives it from the rendered prompt: `prompt.rstrip().endswith("<think>")`.

A verified trace with default arguments:

```text
push('<thi')         -> ('', '')        # could still become <think>; held
push('nk>\nLet me')  -> ('Let me', '')  # in_reasoning is now True
push(' add.</th')    -> (' add.', '')   # '</th' held back
push('ink>\n\n4')    -> ('', '4')       # whitespace after </think> stripped
push(' is it')       -> ('', ' is it')
flush()              -> ('', '')
```

## How to use it

### Generate a non-streaming result, including reasoning

```python
import asyncio

from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS
from kumaru.core.types import Message


async def main() -> None:
    cfg = load_config("test")  # echo backend: no model, no GPU
    backend = BACKENDS.create(cfg.backend.name, cfg.backend)
    await backend.load()
    try:
        params = cfg.generation.merged(thinking=True, thinking_budget=3)
        result = await backend.generate([Message.user("2+2?")], params)
        print(result.message.reasoning)  # 'The user asked: 2+2?. Ec' (3 echo pieces)
        print(result.message.content)
        print(result.finish_reason, result.usage, result.latency_ms, result.time_to_first_token_ms)
    finally:
        await backend.aclose()


asyncio.run(main())
```

### Wrap a blocking producer

```python
import contextlib
import threading

from kumaru.backends.base import iterate_in_thread
from kumaru.core.types import StreamChunk

_engine_lock = threading.Lock()  # one engine context, not thread-safe


async def stream_tokens(sync_model, prompt: str):
    def produce():
        with _engine_lock:  # taken on the worker thread, never on the event loop
            tokens = sync_model.generate_tokens(prompt)
            try:
                for token in tokens:
                    yield StreamChunk(delta=token)
            finally:  # runs on the worker thread, also when the reader left early
                getattr(tokens, "close", lambda: None)()
            yield StreamChunk(finish_reason="stop")

    async with contextlib.aclosing(iterate_in_thread(produce, queue_size=64)) as chunks:
        async for chunk in chunks:
            yield chunk
```

This is the shape of the `llamacpp` backend. The reader can leave early: a client disconnect, the UI's Stop, a `break`, a cancelled task. When it does, `iterate_in_thread` raises an internal flag. The worker sees the flag when the producer hands over its next item, or within 0.1 s if it is waiting for room in a full queue. It then stops pulling and calls the producer's `close()` on the `kumaru-generate` thread. That raises `GeneratorExit` at the suspended `yield`, so the producer's `finally` blocks run there in order: here the engine stream is closed, then the lock is released. Engine cleanup therefore belongs in the producer's `finally`; the event loop never touches the engine.

`contextlib.aclosing` closes the inner iterator as soon as your own `stream()` is closed. Without it, closing waits for async-generator finalization, which CPython usually runs at once but does not guarantee.

Each running stream uses two threads: the `kumaru-generate` worker, and, while the reader waits for its next item, a thread of the event loop's default executor blocked in `queue.get`. If the reader is cancelled during that wait, the worker's end-of-stream sentinel releases the executor thread, so cancelled requests do not use up the executor. `tests\test_backends.py` pins this.

Measured with a fake 1,000-token model, `queue_size=64`, and a reader that leaves after 5 chunks. In both cases the `finally` ran on `kumaru-generate`, and no thread was left behind.
- A producer much faster than the reader had made 70 items when the reader left (5 read, 64 queued, 1 in hand). It stopped there, and the thread exited about 0.1 s later.
- A producer slower than the reader (10 ms per token) stopped after 6 items.

Limits:
- **Abandonment is seen between items.** A step that blocks for a long time, such as prefilling a long prompt before the first token, finishes first.
- **Push-based engines need a signal.** An engine that runs on its own thread, such as transformers' `generate`, does not stop when its iterator is closed. The producer's `finally` must tell it to stop. The transformers backend sets a `threading.Event` that a `StoppingCriteria` checks after every token.
- **A producer still runs until its first item.** A producer that blocks on a lock before its first item therefore does that work once it gets the lock, even if its reader has gone. In `llamacpp`, a request cancelled while it waited for the model still evaluates its prompt when its turn comes, and decodes until it has a first chunk to hand over (a token or a few).
- **Late errors go nowhere.** An exception from `close()` is logged as the warning `closing an abandoned generation failed`. An exception the producer raises after the reader left is dropped.

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

### Split inline thinking from a raw token stream

This is the exact pattern the transformers, llamacpp and openai_compat backends use. The splitter runs first, and only the answer channel goes through the stop tracker:

```python
from kumaru.backends.base import ReasoningSplitter, StopSequenceTracker
from kumaru.core.types import StreamChunk


def to_chunks(raw_pieces, stop, *, prompt: str):
    splitter = ReasoningSplitter(start_in_reasoning=prompt.rstrip().endswith("<think>"))
    tracker = StopSequenceTracker(stop)
    for raw in raw_pieces:
        thought, text = splitter.push(raw)
        safe = tracker.push(text) if text else ""
        if thought or safe:
            yield StreamChunk(delta=safe, reasoning=thought)
        if tracker.stopped:
            break
    thought, text = splitter.flush()
    tail = (tracker.push(text) if text else "") + tracker.flush()
    if thought or tail:
        yield StreamChunk(delta=tail, reasoning=thought)
    # ...then the final StreamChunk(finish_reason=..., usage=...)
```

### Exercise the contract without a model

```powershell
kumaru chat -c test -q "Does streaming work?"
kumaru chat -c test --think -q "Does the reasoning channel work?"   # dim reasoning, then the answer
kumaru chat -c test --think --thinking-budget 2 -q "Short thought"  # reasoning cut to 2 pieces
```

## Tuning / operational notes

| Knob | Where | Trade-off |
|---|---|---|
| `queue_size` | `iterate_in_thread(..., queue_size=256)` | Higher lets a fast producer run farther ahead; lower applies backpressure earlier for slow clients. It also bounds the wasted work when a slow reader leaves: up to `queue_size` computed items are discarded. |
| `backend.max_context_length` | `Backend.context_length` / config | Hard context cap for memory trimming and backend prompt budgets. |
| `agent.max_history_tokens` | config | The agent clamps the history budget to `min(max_history_tokens, max(512, context_length − min(max_tokens, context_length // 2) − 128 − tool_tokens))`, where `tool_tokens` estimates the tool declarations sent with the request (0 without tools). `max_tokens` is the resolved value, including `thinking_mode` when thinking is on, so a long reply budget shrinks the history, but never reserves more than half the window. The agent then trims the request's `max_tokens` to the room left, `context_length − count_tokens(prompt) − tool_tokens` (at least 1), because strict servers reject prompt plus `max_tokens` beyond the window. |
| Final `StreamChunk.usage` | backend stream end | Accurate usage improves dashboards. If absent, `generate()` estimates prompt and completion tokens (reasoning included). |
| Stop string length | `StopSequenceTracker` | The tracker holds back up to `max_len - 1` characters to catch split stop sequences. |
| `start_in_reasoning` | `ReasoningSplitter(...)` | Must match the rendered prompt. If it is wrong in one direction, thinking leaks into the answer; in the other, the whole answer is hidden as reasoning. |

Expected behavior: one `generate()` call produces a `ChatResult` with:
- total latency;
- optional time to first token, where a reasoning piece counts;
- finish reason;
- usage;
- `message.reasoning`, which is `None` when the model did not think.

Backends with real tokenizers should override `count_tokens()`; remote or fake backends can use the base heuristic.

## How to extend it

A complete minimal backend that honours the whole v0.2 contract can live in `src\kumaru\backends\reverse_backend.py`. It covers thinking on/off/budget, stop sequences on the answer only, `max_tokens` across both channels, and usage in the final chunk:

```python
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from kumaru.backends.base import Backend, StopSequenceTracker, chunks_from_text
from kumaru.core.registry import BACKENDS
from kumaru.core.types import BackendInfo, FinishReason, GenerationParams, Message, StreamChunk, Usage


@BACKENDS.register("reverse")
class ReverseBackend(Backend):
    """Toy engine: thinks briefly, then answers with the last user message reversed."""

    name = "reverse"

    async def _load(self) -> None:
        return None  # real backends: load weights, spawn the server or open the client here

    def info(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            model_id=self.config.model,
            device="cpu",
            dtype="none",
            context_length=self.config.max_context_length,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        *,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.check_tools(tools)  # this toy cannot call tools, so it refuses them
        text = next((m.content for m in reversed(messages) if m.role.value == "user"), "")
        pieces: list[tuple[str, str]] = []  # (channel, piece); one piece = one "token"
        if params.thinking:  # None = the model's default, which is "off" for this toy
            thought = list(chunks_from_text(f"Reversing {len(text)} characters.", size=8))
            pieces += [("reasoning", p) for p in thought[: params.thinking_budget]]
        pieces += [("answer", p) for p in chunks_from_text(text[::-1], size=8)]

        finish: FinishReason = "length" if len(pieces) > params.max_tokens else "stop"
        tracker = StopSequenceTracker(params.stop)  # answer channel only
        generated = 0
        for channel, piece in pieces[: params.max_tokens]:  # max_tokens covers reasoning too
            generated += 1
            if channel == "reasoning":
                yield StreamChunk(reasoning=piece)
                continue
            if safe := tracker.push(piece):
                yield StreamChunk(delta=safe)
            if tracker.stopped:
                finish = "stop"
                break
        if tail := tracker.flush():
            yield StreamChunk(delta=tail)

        yield StreamChunk(
            finish_reason=finish,
            usage=Usage(prompt_tokens=self.count_tokens(messages), completion_tokens=generated),
        )
```

Verified behaviour for the user message `"hello world"`:

| `GenerationParams(...)` | `reasoning` | `content` | `finish_reason` | `completion_tokens` |
|---|---|---|---|---|
| `thinking=True, thinking_budget=2, stop=["ol"]` | `'Reversing 11 cha'` | `'dlrow '` | `stop` | 3 |
| defaults | `None` | `'dlrow olleh'` | `stop` | 2 |
| `thinking=True, max_tokens=3` | `'Reversing 11 characters.'` | `''` | `length` | 3 |

The third row is the "thought until the budget ran out" case. The answer stays empty rather than receiving half a thought.

Register it lazily in `src\kumaru\core\registry.py`, next to the built-ins, so that `import kumaru` stays cheap:

```python
BACKENDS.register_lazy("reverse", "kumaru.backends.reverse_backend:ReverseBackend")
```

A backend shipped in a separate pip package registers through the `kumaru.backends` entry-point group instead. `Registry.load_entry_points()` scans that group the first time a name is not found.

Use it from YAML:

```yaml
backend:
  name: reverse
  model: reverse-demo
  max_context_length: 8192

generation:
  max_tokens: 128
```

Checklist for a real backend:

1. Do all expensive work in `_load()`: weights, child process, HTTP client. Import heavy optional packages there too. If one is missing, raise `DependencyMissingError(package, extra=...)` from `kumaru.core.errors`. Raise `BackendError(message, hint=...)` for load or generation failures. Both carry an `http_status` (501 and 503) that the server's `KumaruError` handler returns.
2. Override `aclose()` to free resources and set `self._loaded = False`, as every shipped backend does. Make it safe to call twice.
3. Wrap blocking token iterators in `iterate_in_thread()`, inside `contextlib.aclosing(...)`. Put engine cleanup in the producer's `finally`: stop the engine, release locks. Take engine locks inside the producer, on the worker thread (see [Wrap a blocking producer](#wrap-a-blocking-producer)).
4. Map every `GenerationParams` field the engine supports. `min_p`, `presence_penalty`, `seed`, `thinking` and `thinking_budget` use `None` to mean "do not send; engine default". Send them only when they are set. Ignore the fields you cannot honour, and say so in the backend's how-to page.
5. Route thinking to `StreamChunk.reasoning`. Use the engine's separate field if it has one. Otherwise run raw text through `ReasoningSplitter`, and set `start_in_reasoning` from the rendered prompt.
6. Apply stop sequences to answer text only, after the splitter.
7. Map engine finish reasons with `normalize_finish_reason()`. End with one chunk that carries `finish_reason` and `usage`. Completion tokens include reasoning tokens.
8. Make `info()` reflect the loaded state: real device, quantization, context length.
9. Leave `supports_adapters = False` unless `_load()` attaches `self.config.adapter_path`. If it does, set it to `True` and update `test_transformers_is_the_backend_that_attaches_adapters` in `tests\test_backends.py`, which pins the list of adapter backends.
10. Leave `supports_tools = False`, and call `self.check_tools(tools)` first in `stream()`, unless the engine can take tool declarations. If it can: send `tools`, collect the model's calls in the OpenAI shape (`{"id", "type": "function", "function": {"name", "arguments": <JSON string>}}`), and put them on the final chunk with `finish_reason="tool_calls"`. A reply that did not finish keeps its reason (`length`, `error` or `cancelled`), and `ChatAgent` never runs calls from it. The backend never runs them either; the agent does, after asking. `merge_tool_call_delta()` and `finish_tool_calls()` in `openai_compat.py` assemble streamed call fragments.
11. Add tests modelled on `tests\test_backends.py`. Stub the engine so that CI needs no model.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Server freezes during generation | Backend calls a blocking generator on the event loop. | Wrap it with `iterate_in_thread()`. |
| GPU stays busy after a disconnect or Stop | The producer's `finally` does not stop the engine; a push-based engine needs an explicit signal. Or one long step, such as a prefill, has not finished: abandonment is seen between items. | Stop the engine in the producer's `finally` ([Wrap a blocking producer](#wrap-a-blocking-producer)). Bound generation with `max_tokens`. |
| Every request hangs after one was abandoned | An engine lock is taken on the event loop, or released outside the producer's `finally`. | Hold it with `with lock:` inside the producer, around the whole generation. |
| Stop sequence appears in output | Backend checks only per chunk and misses cross-chunk stops. | Use `StopSequenceTracker(params.stop)`. |
| Reply ends mid-thought, or as soon as the model *thinks* a stop string | Raw text, including the `<think>` block, is fed to the stop tracker or to the engine's own stop matching. | Run `ReasoningSplitter` first and track stops on the answer only. The shipped backends also forward `stop` to the engine, which matches raw output: llama-cpp-python (`llamacpp`) and llama-server (`llamacpp_server`, `openai_compat`) both do. Leave `generation.stop` empty for thinking models, or use strings that cannot appear in prose. |
| `<think>`/`</think>` tags, or the model's thinking, show up in the answer | The engine returns thinking inline and the backend does not split it. Or the prompt already ends with `<think>` but the splitter was built with `start_in_reasoning=False`, so only a stray `</think>` appears. | Use `ReasoningSplitter(start_in_reasoning=prompt.rstrip().endswith("<think>"))`. |
| Whole reply lands in reasoning; answer empty; `finish_reason == "length"` | The model used all of `max_tokens` before closing `</think>`. Unterminated reasoning stays reasoning by design. | Raise `max_tokens` (or `thinking_mode.max_tokens`), or set `thinking_budget` on a backend that enforces it (`llamacpp_server`). |
| Thinking toggle has no effect | Backend ignores `params.thinking` (`llamacpp`; `openai_compat` with `strict_openai`; a model without a thinking template). | Use `llamacpp_server`, or check the thinking-support table above. |
| `ValidationError` from `StreamChunk(finish_reason=...)` mid-stream | Engine-specific finish reason such as `"eos"` or `"function_call"` passed through unmapped. | Wrap it with `normalize_finish_reason()`. |
| Final SSE frame shows `completion_tokens: 0` and `tokens_per_second: 0.0` | `usage` was emitted after the `finish_reason` chunk, or was zero or never sent. | Put real `usage` on the final `finish_reason` chunk. |
| Non-streaming results have rough usage | Final stream chunk omitted `Usage`, so `generate()` estimated it. | Emit `StreamChunk(finish_reason=..., usage=Usage(...))`. |
| Prompt rejected as too long (`Prompt is N tokens but the context window is M.` from `transformers`, HTTP 400 from a server) | One message alone is over budget: `sliding_window` always keeps the two newest messages. Or, on a backend without a real tokenizer (`llamacpp_server`, `openai_compat`), the `count_tokens` heuristic (~3.6 characters per token, tuned for English) counted fewer tokens than the engine. | Shorten the message, lower `agent.max_history_tokens`, or raise `backend.max_context_length`. Override `count_tokens` with a real tokenizer where you have one. Lowering `max_tokens` does not help: it only enlarges the history budget. |
| `The <name> backend cannot attach a LoRA adapter (backend.adapter_path is '<path>').` | `adapter_path` or `--adapter` is set, and the backend's `supports_adapters` is `False` | Serve the adapter with `transformers` and its base model (`-c default`); GGUF models need the adapter converted first (`docs\TRAINING.md`). |
| `The <name> backend cannot call tools, and tools are enabled (agent.enabled_tools, or kumaru chat --workspace).` | Tools are enabled, and the backend's `supports_tools` is `False` (`transformers`, `llamacpp`). | Use `llamacpp_server` (for example `-c qwen36`) or `openai_compat` with a server that supports tool calls, or turn the tools off. |
| `/v1/models` shows stale data | `info()` does not reflect the loaded model, device or config. | Populate `BackendInfo` from actual backend state after load. |
| Memory grows for slow clients | Producer runs too far ahead of the consumer. | Keep the bounded queue; lower `queue_size` for high-concurrency services. |
| `Unknown backend: '<name>'.` | Name neither registered, lazily registered, nor found in entry points. | Add the `register_lazy` line, or check the spelling against the hint's list. |

## Related files

- `src\kumaru\core\types.py` — canonical `Message` (with `reasoning`), `GenerationParams`, `StreamChunk`, `Usage`, `ChatResult` and `BackendInfo`.
- `src\kumaru\core\config.py` — `BackendConfig`, generation defaults and the `thinking_mode` preset consumed upstream of here.
- `src\kumaru\core\registry.py` — resolves backend names to `Backend` subclasses.
- `src\kumaru\core\errors.py` — `BackendError`, `DependencyMissingError`, and `ConfigError` (the adapter refusal in `load()`, the tool refusal in `check_tools()`).
- `src\kumaru\agent\chat.py` — the tool loop, which calls `stream(..., tools=...)` and runs the calls.
- `src\kumaru\backends\echo.py` — simplest concrete backend; emulates thinking.
- `src\kumaru\backends\transformers_backend.py` — local Hugging Face implementation; inline tags via `ReasoningSplitter`.
- `src\kumaru\backends\llamacpp_backend.py` — in-process GGUF implementation.
- `src\kumaru\backends\openai_compat.py` — remote HTTP implementation; home of `normalize_finish_reason`.
- `src\kumaru\backends\llamacpp_server.py` — managed `llama-server`; the reference for native reasoning plus thinking budgets.
- `src\kumaru\server\routes_chat.py` and `src\kumaru\server\schemas.py` — turn chunks into SSE (`delta.reasoning_content`) and the final stats frame.
- `tests\test_backends.py` — contract tests, including the splitter, stop-tracker and abandoned-stream cases.
