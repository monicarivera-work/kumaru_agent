# `src\kumaru\core\types.py`

> Canonical Pydantic data models passed between Kumaru's UI, server, agent, backends, and data loaders.

**Read this when:** you add a chat schema field, implement a backend, process streaming output, or need to understand the exact shape of messages and generation results.

---

## What it does
This file defines Kumaru's shared message, generation, streaming, usage, result, and backend metadata types. The models validate data at API and disk boundaries while keeping the same shape across layers. Backends and agents use these models rather than backend-specific payloads.

## Why it exists
Every layer needs to agree on the same conversation and result shapes. Without these models, the server, UI, training data, and backends would each invent slightly different schemas, forcing adapters everywhere and making substitutions risky.

## Mental model
`Message` is the unit of conversation, `GenerationParams` is the normalised decoding contract, and `StreamChunk` or `ChatResult` is the generated output. Streaming chunks contain only the new `delta`; consumers append deltas. Backends translate these neutral models to their native wire format at the edge.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `FinishReason` | `Literal["stop", "length", "error", "cancelled"]` | Allowed generation termination reasons. |
| `Role` | `class Role(str, Enum)` | Identifies who produced a message. |
| `Message` | `Message(role: Role, content: str = "", name: str | None = None, tool_call_id: str | None = None, tool_calls: list[dict[str, Any]] | None = None)` | One conversation turn. |
| `Message.to_openai` | `to_openai(self) -> dict[str, Any]` | Render a message in OpenAI-compatible chat format. |
| `Message.system` | `system(content: str) -> Message` | Build a system message. |
| `Message.user` | `user(content: str) -> Message` | Build a user message. |
| `Message.assistant` | `assistant(content: str) -> Message` | Build an assistant message. |
| `GenerationParams` | `GenerationParams(max_tokens=1024, temperature=0.7, top_p=0.95, top_k=50, repetition_penalty=1.05, stop=[], seed=None)` | Backend-neutral decoding parameters. |
| `GenerationParams.merged` | `merged(self, **overrides: Any) -> GenerationParams` | Return a copy with non-`None` overrides applied. |
| `Usage` | `Usage(prompt_tokens: int = 0, completion_tokens: int = 0)` | Token accounting. |
| `Usage.total_tokens` | `property -> int` | Sum of prompt and completion tokens. |
| `StreamChunk` | `StreamChunk(delta: str = "", finish_reason: FinishReason | None = None, usage: Usage | None = None)` | One incremental streaming response piece. |
| `ChatResult` | `ChatResult(id=..., created=..., model="kumaru", message: Message, finish_reason="stop", usage=Usage(), latency_ms=0.0, time_to_first_token_ms=None)` | Completed non-streaming generation result. |
| `ChatResult.tokens_per_second` | `property -> float` | Completion throughput derived from usage and latency. |
| `BackendInfo` | `BackendInfo(name: str, model_id: str, device="unknown", dtype="unknown", context_length=0, supports_streaming=True, quantization=None, extra={})` | Loaded backend metadata shown by API and UI. |

## How to use it

Build a conversation:

```python
from kumaru.core.types import Message

messages = [
    Message.system("You are a concise operator assistant."),
    Message.user("What is the current backend?"),
]
```

Convert to OpenAI-compatible wire format:

```python
payload = [message.to_openai() for message in messages]
```

Apply request-level generation overrides without mutating defaults:

```python
from kumaru.core.types import GenerationParams

defaults = GenerationParams(max_tokens=1024, temperature=0.7)
request_params = defaults.merged(max_tokens=128, temperature=None)

assert request_params.max_tokens == 128
assert request_params.temperature == 0.7
```

Consume streaming chunks correctly:

```python
from kumaru.core.types import StreamChunk

chunks = [
    StreamChunk(delta="Hel"),
    StreamChunk(delta="lo"),
    StreamChunk(finish_reason="stop"),
]

text = "".join(chunk.delta for chunk in chunks)
```

Create backend metadata:

```python
from kumaru.core.types import BackendInfo

info = BackendInfo(
    name="echo",
    model_id="echo",
    device="cpu",
    dtype="none",
    context_length=8192,
)
```

## How to extend it

Add fields only when the shape is a cross-layer contract. Prefer optional fields or default values when a change must be backward-compatible for API clients and JSONL data.

For example, to expose a backend capability in metadata:

```python
class BackendInfo(BaseModel):
    name: str
    model_id: str
    device: str = "unknown"
    dtype: str = "unknown"
    context_length: int = 0
    supports_streaming: bool = True
    quantization: str | None = None
    supports_tools: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)
```

Then update every backend's `info()` method to set the field where needed. Do not add backend-specific generation knobs to `Message`; put backend-specific data in backend config or `BackendInfo.extra`.

## Failure modes
| Symptom | Likely cause | Fix |
|---|---|---|
| Pydantic validation fails for `GenerationParams` | A value is outside the declared bounds, such as `temperature=3.0` or `max_tokens=0`. | Use values within the field constraints. |
| Streamed text repeats itself | A consumer treats `delta` as accumulated text. | Append each chunk's `delta`; do not replace with or re-emit the full buffer. |
| Assistant tool-call messages fail because `content` is `null` | Some clients send `null` content for tool-call turns. | Use `Message`; its validator coerces `None` content to `""`. |
| Throughput shows `0.0` | `ChatResult.latency_ms` is zero or negative. | Ensure generation code records elapsed wall-clock time. |
| UI cannot display backend details | `Backend.info()` returned incomplete or invalid `BackendInfo`. | Populate required `name` and `model_id`, and set context and device fields where known. |

## Related files
- `src\kumaru\__init__.py` — re-exports the main public types.
- `src\kumaru\core\config.py` — embeds `GenerationParams` in `KumaruConfig`.
- `src\kumaru\backends\base.py` — uses `Message`, `GenerationParams`, `StreamChunk`, `ChatResult`, `Usage`, and `BackendInfo`.
- `src\kumaru\agent\chat.py` — prepares `Message` lists and returns `ChatResult` or `StreamChunk`.
- `src\kumaru\cli.py` — uses `Message` in the terminal chat loop.
