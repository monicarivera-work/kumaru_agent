# `src\kumaru\core\types.py`

> Canonical Pydantic data models passed between Kumaru's UI, server, agent, backends, and data loaders: messages, decoding parameters, stream chunks, results, and backend metadata.

**Read this when:** you add a chat schema field or a sampling knob, implement a backend, consume streaming output (answer and reasoning), or need the exact shape of messages and generation results.

---

## What it does

This file defines the shared message, generation, streaming, usage, result, and backend-metadata types. The models validate data at the API and disk boundaries and keep one shape across all layers. Backends and the agent work with these models, not with backend-specific payloads.

v0.2 adds two things:

- **A reasoning channel.** `Message.reasoning` holds a reasoning model's private chain of thought, separate from the answer in `content`. `StreamChunk.reasoning` streams it.
- **Decoding knobs.** `GenerationParams` gains:
  - the tri-state `thinking` switch and a `thinking_budget` cap;
  - two optional sampling knobs, `min_p` and `presence_penalty`, which are not sent when `None`.

## Why it exists

Every layer must agree on the same conversation and result shapes. Without these models, the server, UI, training data, and backends would each invent slightly different schemas. That would force adapters everywhere and make swapping a layer risky.

Reasoning is a separate field, not text mixed into `content`, because the two have different lifecycles:

- **Reasoning** is shown to the user, and the server times it (`thinking_ms`). It is never sent back to the model.
- **The answer** is what history, the CLI transcript, and API clients consume.

`to_openai()` omits reasoning by default, for two reasons. Model authors, Qwen included, train on histories that contain only final answers. Old thinking would also burn context for no accuracy benefit.

## Mental model

- **`Message`** is the unit of conversation.
  - `content` is the answer and is always a string; `None` is coerced to `""`.
  - `reasoning` is optional thinking, carried for display.
  - On input, `reasoning` accepts either `reasoning` or the OpenAI-ecosystem name `reasoning_content`.
  - `model_dump()` emits the field name `reasoning`. Only `to_openai(include_reasoning=True)` emits `reasoning_content`.
- **`GenerationParams`** is the normalised decoding contract, and every backend translates it to native arguments.
  - Fields typed `X | None` mean "not sent; the engine uses its own default".
  - `thinking` is tri-state: `None` uses the model's default, `True`/`False` forces thinking on/off.
  - The agent resolves the final params by layering config defaults, the `thinking_mode` preset, and per-request overrides through `merged()`; see `src\kumaru\agent\chat.py`.
- **`StreamChunk`** carries increments on two independent channels, `delta` (answer) and `reasoning` (thinking). A chunk may carry either, both, or neither. Consumers append each channel to its own buffer. The final chunk carries `finish_reason` and, when the backend knows it, `usage`.
- **`ChatResult`** is a completed generation. `Backend.generate()` builds it from the stream: it joins the reasoning into `message.reasoning` (`None` if empty) and records latency and time to first token. Time to first token counts the first reasoning token, not the first answer token.

Invariants:

- `delta` and `reasoning` are never accumulated text.
- `to_openai()` never emits `reasoning_content` unless asked.
- No `count_tokens` implementation counts `reasoning`.
- `merged()` ignores `None` overrides and does not re-validate.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `FinishReason` | `Literal["stop", "length", "error", "cancelled"]` | Allowed generation termination reasons. |
| `Role` | `class Role(str, Enum)` — `SYSTEM = "system"`, `USER = "user"`, `ASSISTANT = "assistant"`, `TOOL = "tool"` | Who produced a message; mirrors the OpenAI chat schema. |
| `Message` | `Message(role: Role, content: str = "", name: str \| None = None, tool_call_id: str \| None = None, tool_calls: list[dict[str, Any]] \| None = None, reasoning: str \| None = None)` | One conversation turn. `content=None` becomes `""`. `reasoning` has `validation_alias=AliasChoices("reasoning", "reasoning_content")`. |
| `Message.to_openai` | `to_openai(self, *, include_reasoning: bool = False) -> dict[str, Any]` | Wire format: always `role` and `content`; `name`/`tool_call_id`/`tool_calls` when truthy; `reasoning_content` only when `include_reasoning` is set and `reasoning` is non-empty. |
| `Message.system` | `@classmethod system(cls, content: str) -> Message` | Build a system message. |
| `Message.user` | `@classmethod user(cls, content: str) -> Message` | Build a user message. |
| `Message.assistant` | `@classmethod assistant(cls, content: str, reasoning: str \| None = None) -> Message` | Build an assistant message, optionally with its reasoning. |
| `GenerationParams` | `GenerationParams(max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 50, repetition_penalty: float = 1.05, min_p: float \| None = None, presence_penalty: float \| None = None, stop: list[str] = [], seed: int \| None = None, thinking: bool \| None = None, thinking_budget: int \| None = None)` | Backend-neutral decoding parameters. Bounds: `max_tokens` 1–131072, `temperature` 0–2, `top_p` 0–1, `top_k` ≥ 0, `repetition_penalty` 0.5–2.0, `min_p` 0–1, `presence_penalty` −2–2, `thinking_budget` ≥ 0. |
| `GenerationParams.merged` | `merged(self, **overrides: Any) -> GenerationParams` | Copy with the non-`None` overrides applied (`model_copy(update=...)`, so not re-validated). |
| `Usage` | `Usage(prompt_tokens: int = 0, completion_tokens: int = 0)` | Token accounting. |
| `Usage.total_tokens` | `property -> int` | `prompt_tokens + completion_tokens`. |
| `StreamChunk` | `StreamChunk(delta: str = "", reasoning: str = "", finish_reason: FinishReason \| None = None, usage: Usage \| None = None)` | One incremental piece of a streaming response; `delta` is answer text, `reasoning` is thinking text. |
| `ChatResult` | `ChatResult(id: str = "chatcmpl-<24 hex>", created: int = <epoch seconds>, model: str = "kumaru", message: Message, finish_reason: FinishReason = "stop", usage: Usage = Usage(), latency_ms: float = 0.0, time_to_first_token_ms: float \| None = None)` | Completed non-streaming generation; `message.reasoning` holds the joined thinking. |
| `ChatResult.tokens_per_second` | `property -> float` | `completion_tokens / (latency_ms / 1000)`; `0.0` when `latency_ms <= 0`. Completion tokens include thinking tokens. |
| `BackendInfo` | `BackendInfo(name: str, model_id: str, device: str = "unknown", dtype: str = "unknown", context_length: int = 0, supports_streaming: bool = True, quantization: str \| None = None, extra: dict[str, Any] = {})` | Loaded-backend metadata shown by the API and UI. |

## How to use it

### Build a conversation

```python
from kumaru.core.types import Message

messages = [
    Message.system("You are a concise operator assistant."),
    Message.user("What is the current backend?"),
]
payload = [message.to_openai() for message in messages]   # OpenAI-compatible wire format
```

### Keep reasoning out of the wire format (the default)

```python
reply = Message.assistant("42", reasoning="6 * 7 = 42")
reply.to_openai()                        # {'role': 'assistant', 'content': '42'}
reply.to_openai(include_reasoning=True)  # {'role': 'assistant', 'content': '42', 'reasoning_content': '6 * 7 = 42'}
```

Use `include_reasoning=True` only when the model expects its earlier thinking back. The usual case is agentic interleaved thinking across tool calls inside one turn. Kumaru's own chat paths never set it.

### Parse ecosystem payloads that use `reasoning_content`

```python
m = Message.model_validate({"role": "assistant", "content": None, "reasoning_content": "6*7"})
assert (m.content, m.reasoning) == ("", "6*7")
m.model_dump()["reasoning"]              # dumps use the field name, not the alias
```

### Layer request overrides on defaults, including thinking

```python
from kumaru.core.types import GenerationParams

defaults = GenerationParams(max_tokens=1024, temperature=0.7)
p = defaults.merged(max_tokens=128, temperature=None, thinking=True, thinking_budget=512)
assert (p.max_tokens, p.temperature, p.thinking, p.thinking_budget) == (128, 0.7, True, 512)

# None means "no override", so merged() cannot reset a field to None.
assert p.merged(thinking=None).thinking is True
back_to_model_default = p.model_copy(update={"thinking": None})
```

### Consume a stream on both channels

```python
answer: list[str] = []
thought: list[str] = []
for chunk in chunks:                      # or `async for` over agent.stream(...)
    if chunk.reasoning:
        thought.append(chunk.reasoning)   # render dimmed or collapsed; never feed back to the model
    if chunk.delta:
        answer.append(chunk.delta)
    if chunk.finish_reason:
        usage = chunk.usage               # may be None for backends without accounting

history_entry = Message.assistant("".join(answer))   # history keeps the answer only
```

### Describe a loaded backend

```python
from kumaru.core.types import BackendInfo

info = BackendInfo(name="echo", model_id="echo", device="cpu", dtype="none", context_length=8192)
```

## How to extend it

Add fields only when the shape is a contract shared across layers. Prefer optional or defaulted fields, so existing API clients and JSONL data keep validating.

**Add an optional sampling knob.** v0.2 added `min_p` and `presence_penalty` this way:

1. Add it to `GenerationParams` as `knob: float | None = Field(default=None, ge=..., le=...)`. `None` must mean "do not send", so engines keep their own defaults.
2. Translate it in each backend, and omit it when `None`. For examples, see `openai_compat.py` and `llamacpp_backend.py`, which check `is not None`. Backends that cannot honour a knob ignore it: `transformers` has no `presence_penalty`.
3. Expose it on the HTTP request: add a field to `ChatCompletionRequest` and a key in `generation_overrides()` in `src\kumaru\server\schemas.py`.
4. If it belongs to the thinking preset, add it to `ThinkingModeConfig` in `src\kumaru\core\config.py`. Document it in `configs\default.yaml`.

**Accept an ecosystem field name without renaming yours.** Use `validation_alias=AliasChoices("yours", "theirs")`, as `reasoning` does. Inputs accept both names; dumps keep yours.

**Expose a backend capability in metadata.** Add a defaulted field to `BackendInfo`, for example `supports_tools: bool = False`. Then set it in every backend's `info()` that supports the capability. Do not put backend-specific knobs on `Message`. Use backend config or `BackendInfo.extra` instead.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Pydantic validation fails for `GenerationParams` | A value is outside the declared bounds, such as `temperature=3.0`, `max_tokens=0`, or `thinking_budget=-1`. | Use values within the field constraints. |
| An out-of-range value (e.g. `max_tokens=200000`) reaches the backend | `merged()` uses `model_copy(update=...)`, which skips validation. | Validate at the boundary. Config and CLI values are validated by `load_config`; HTTP fields by `ChatCompletionRequest`, which has no upper bound on `max_tokens`. |
| `merged(seed=None)` or `merged(thinking=None)` does not clear the field | `None` means "no override". | Use `model_copy(update={"field": None})`. |
| Streamed text repeats itself | A consumer treats `delta` as accumulated text. | Append each chunk's `delta`; never re-emit the buffer. |
| Thinking text shows up inside the answer | The consumer concatenates `chunk.reasoning` into the answer buffer. | Keep two buffers; store only the answer in history. |
| Multi-turn context fills fast, or the model echoes old thinking | Reasoning was re-sent, via `to_openai(include_reasoning=True)` or by pasting it into `content`. | Keep the default `include_reasoning=False`; store `Message.assistant(answer)`. |
| `reasoning_content` missing from `model_dump()` output | The alias applies to input only; dumps use the field name `reasoning`. | Read `reasoning`, or call `to_openai(include_reasoning=True)` for the ecosystem name. |
| Assistant tool-call messages fail because `content` is `null` | Some clients send `null` content for tool-call turns. | Use `Message`; its validator coerces `None` content to `""`. |
| Throughput shows `0.0` | `ChatResult.latency_ms` is zero or negative. | Ensure generation code records elapsed wall-clock time; `Backend.generate()` does this. |
| UI cannot display backend details | `Backend.info()` returned incomplete or invalid `BackendInfo`. | Populate the required `name` and `model_id`, and set context and device fields where known. |

## Known limitations

- **`GenerationParams` does not forbid unknown keys.** Pydantic's default `extra="ignore"` applies, so a typo is silently dropped. That holds for `GenerationParams(temprature=0.2)`, for the `generation:` block of a YAML profile, and for `merged(temprature=0.2)`, which attaches the key to the object without dumping or using it. A Python caller's `agent.complete(msgs, temprature=0.2)` is affected too. Workaround: run `kumaru config -c <profile>` and check that your value shows up.
- **`merged()` does not validate.** Out-of-range or wrongly typed overrides, such as `temperature=5.0` or `max_tokens=0`, pass straight to the backend. Kumaru's own entry points validate first: config through `load_config`, CLI flags through Typer's `min`/`max`, HTTP through `ChatCompletionRequest`. Python callers must validate their own overrides. Workaround: `GenerationParams.model_validate(p.model_dump())` re-checks a merged result.
- **`top_k` and `repetition_penalty` always have a value.** Unlike `min_p` and `presence_penalty`, they cannot be `None`, so Kumaru's defaults of 50 and 1.05 override the engine's and the model's own defaults even when no config sets them. They are sent:
  - always by `llamacpp`;
  - by `openai_compat` and `llamacpp_server` unless `strict_openai` is set;
  - by `transformers` when it samples.

  Workaround: set both in the profile.
  - `top_k: 0` disables top-k in `llamacpp`. `openai_compat`, `llamacpp_server` and `transformers` omit it, so the engine's or the model's default applies.
  - `repetition_penalty: 1.0` is omitted by `openai_compat` and `llamacpp_server` and is a no-op elsewhere.

## Related files

- `src\kumaru\__init__.py` — re-exports the main public types.
- `src\kumaru\core\config.py` — embeds `GenerationParams` in `KumaruConfig` and defines the `ThinkingModeConfig` preset.
- `src\kumaru\agent\chat.py` — `resolve_params()` layers generation defaults, `thinking_mode`, and request overrides with `merged()`.
- `src\kumaru\backends\base.py` — `generate()` turns a stream into a `ChatResult`, joining reasoning; `count_tokens()` counts content only.
- `src\kumaru\server\schemas.py` — maps `Message.reasoning` to `reasoning_content` on the wire and drops it from incoming history.
- `src\kumaru\server\routes_chat.py` — streams `reasoning` as separate SSE `reasoning_content` deltas.
- `src\kumaru\cli.py` — prints reasoning dimmed and stores only the answer in the REPL history.
