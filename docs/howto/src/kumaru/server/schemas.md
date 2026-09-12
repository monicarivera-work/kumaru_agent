# `src\kumaru\server\schemas.py`

> Pydantic wire models for Kumaru's OpenAI-compatible HTTP API.

**Read this when:** you are changing request/response JSON, debugging client compatibility, or updating the browser streaming parser.

---

## What it does
This file defines the public HTTP contract separately from internal domain models. Inbound chat messages are converted to `kumaru.core.types.Message`; outbound completions and SSE chunks mirror OpenAI's chat-completions shapes. It also defines health, model-list, and error response bodies.

## Why it exists
Core types are internal and can be refactored, but HTTP schemas are a published contract. If the server reused internal models directly, internal renames would become breaking API changes. These models keep compatibility with OpenAI clients while allowing Kumaru-specific additive metadata under `kumaru`.

## Mental model
- Request fields default to `None` when absence must preserve server config.
- Unknown extra request fields are ignored for OpenAI-client tolerance.
- Streaming chunks carry deltas; clients append `choices[0].delta.content`.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `ChatMessageIn` | `class ChatMessageIn(BaseModel)` | One inbound message with `role`, optional `content`, `name`, and `tool_call_id`. |
| `ChatMessageIn.to_domain` | `to_domain() -> Message` | Converts wire role/content fields to internal `Message`. |
| `ChatCompletionRequest` | `class ChatCompletionRequest(BaseModel)` | Body for `POST /v1/chat/completions`. |
| `ChatCompletionRequest.generation_overrides` | `generation_overrides() -> dict[str, Any]` | Returns request overrides for generation settings. |
| `ChatCompletionRequest.to_domain` | `to_domain() -> list[Message]` | Converts every inbound message to internal messages. |
| `UsageOut` | `class UsageOut(BaseModel)` | Token usage body with prompt, completion, and total tokens. |
| `ChoiceOut` | `class ChoiceOut(BaseModel)` | Non-streaming choice containing assistant message and finish reason. |
| `ChatCompletionResponse` | `class ChatCompletionResponse(BaseModel)` | Non-streaming OpenAI-style completion response plus `kumaru` metrics. |
| `ChatCompletionResponse.from_result` | `from_result(result: ChatResult) -> ChatCompletionResponse` | Converts a domain `ChatResult` to HTTP response JSON. |
| `sse_chunk` | `sse_chunk(*, completion_id: str, model: str, delta: str | None = None, role: str | None = None, finish_reason: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]` | Builds one `chat.completion.chunk` event payload. |
| `new_completion_id` | `new_completion_id() -> str` | Creates `chatcmpl-` IDs for streaming completions. |
| `ModelCard` | `class ModelCard(BaseModel)` | One model item for `/v1/models`. |
| `ModelList` | `class ModelList(BaseModel)` | OpenAI-style list wrapper for model cards. |
| `HealthResponse` | `class HealthResponse(BaseModel)` | `/healthz` response body. |
| `ErrorBody` | `class ErrorBody(BaseModel)` | Nested error details: message, type, optional hint. |
| `ErrorResponse` | `class ErrorResponse(BaseModel)` | Top-level error response wrapper. |

## How to use it

### Parse an OpenAI-style chat request

```python
from kumaru.server.schemas import ChatCompletionRequest

body = ChatCompletionRequest(
    messages=[{"role": "user", "content": "Hello"}],
    stream=False,
    max_completion_tokens=64,
)
messages = body.to_domain()
overrides = body.generation_overrides()
assert messages[0].content == "Hello"
assert overrides["max_tokens"] == 64
```

### Convert a domain result to HTTP JSON

```python
from kumaru.core.types import ChatResult, Message, Usage
from kumaru.server.schemas import ChatCompletionResponse

result = ChatResult(
    model="echo",
    message=Message.assistant("hi"),
    usage=Usage(prompt_tokens=3, completion_tokens=1),
    latency_ms=10.0,
)
response = ChatCompletionResponse.from_result(result)
print(response.model_dump())
```

### Build SSE frames that match the UI parser

```python
import json
from kumaru.server.schemas import new_completion_id, sse_chunk

cid = new_completion_id()
payload = sse_chunk(completion_id=cid, model="echo", delta="hello")
frame = f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
```

The browser in `ui\app.js` reads each frame, parses JSON after `data:`, and appends `evt.choices?.[0]?.delta?.content`.

### Call the API with httpx

```python
import httpx

res = httpx.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
    timeout=600,
)
res.raise_for_status()
print(res.json()["choices"][0]["message"]["content"])
```

## How to extend it
Add fields in a backward-compatible way: optional request fields default to `None`, and optional response metadata should be additive. Keep `ConfigDict(extra="ignore")` on OpenAI-compatible input models unless you intentionally want to reject clients.

Worked example: add an optional request `user` field for auditing without changing generation behavior.

```python
class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessageIn]
    stream: bool = False
    user: str | None = None
    # existing generation fields...
```

Then read `body.user` in `routes_chat.py` for logging. Do not pass unknown fields into `GenerationParams.merged()` unless they exist on that model.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| HTTP 422 validation error | `messages` missing, `role` not one of `system/user/assistant/tool`, or numeric field out of range. | Match `ChatCompletionRequest` field names and constraints. |
| Request `temperature=1.0` overrides server defaults unexpectedly | Client explicitly sent it; only `None` is ignored. | Omit the field to use config defaults. |
| `max_completion_tokens` ignored | Both token fields are absent or `max_tokens` was set and takes precedence through `self.max_tokens or ...`. | Send only the intended token field. |
| Streaming UI shows no text | SSE chunk lacks `choices[0].delta.content`. | Use `sse_chunk(delta=...)` for text deltas. |
| Client fails on `kumaru` field | Client is overly strict about OpenAI responses. | Configure client to ignore unknown fields or use only standard keys. |
| `content: null` becomes empty string | `ChatMessageIn.to_domain()` coerces `None` to `""`. | Send actual content for user/assistant turns. |

## Related files

- `src\kumaru\server\routes_chat.py` — consumes request schemas and returns response schemas.
- `src\kumaru\core\types.py` — internal domain models converted to/from wire shapes.
- `src\kumaru\agent\chat.py` — returns `ChatResult` and `StreamChunk` consumed here.
- `ui\app.js` — parses SSE chunks and `kumaru` streaming metrics.
- `src\kumaru\core\config.py` — generation defaults are preserved when request fields are absent.

