# `src\kumaru\server\schemas.py`

> Pydantic wire models for Kumaru's OpenAI-compatible HTTP API: requests, including the three ways of switching thinking on; responses with `reasoning_content`; SSE chunk bodies; and the health and model list.

**Read this when:**

- you change request or response JSON;
- you debug why a client's thinking switch or sampling field had no effect;
- you update a streaming parser, such as `ui\app.js`, for the reasoning channel.

---

## What it does

This file defines the **published** HTTP contract, separately from the internal types in `src\kumaru\core\types.py`:

- inbound messages become `Message` objects (`to_domain()`);
- request fields become generation overrides (`generation_overrides()`);
- a `ChatResult` becomes an OpenAI `chat.completion` body (`from_result()`);
- `sse_chunk()` builds each `chat.completion.chunk` event.

v0.2 adds:

- request fields `min_p`, `presence_penalty`, `thinking`, `thinking_budget`, `chat_template_kwargs` and `reasoning_effort`;
- `resolved_thinking()`;
- `reasoning_content` in non-streaming messages;
- `sse_chunk(reasoning=...)` for reasoning deltas.

## Why it exists

Core types are internal vocabulary and can be refactored freely; these models are a contract that external clients depend on. Sharing one set of models would turn every internal rename into a breaking API change.

The shapes mirror OpenAI's chat-completions API, so stock clients work unmodified. Kumaru-specific data is **additive**: a top-level `kumaru` object, and `reasoning_content`, which is the field name vLLM, llama.cpp and DeepSeek use. Clients that do not know these fields ignore them.

## Mental model

**`None` means "not sent".**

- Every generation field defaults to `None`.
- `generation_overrides()` returns all 11 keys, including the `None` ones.
- `GenerationParams.merged()` skips `None`, so only fields the client set override the server.

This matters: an SDK that always sends `temperature=1.0` *does* override your tuned default, and the only fix is on the client side.

**Thinking has three accepted spellings**, because clients differ. `resolved_thinking()` checks them in this order and stops at the first one present:

1. `thinking` (Kumaru) — `true`/`false`.
2. `chat_template_kwargs.enable_thinking` (vLLM/SGLang) — a JSON boolean is used as is. Some clients send a string instead; it means on only if it is `"true"`, `"1"`, `"yes"` or `"on"` (case- and whitespace-insensitive), so `"false"` means off. `null` counts as absent.
3. `reasoning_effort` (OpenAI) — `"none"` (case- and whitespace-insensitive) means off; an empty or whitespace-only string counts as absent; **any other string means on**.
4. None of them — `None`, which uses `generation.thinking` from the config.

Only the on/off decision is taken from `chat_template_kwargs`; its other keys are ignored. Likewise `reasoning_effort` does not set a budget; use `thinking_budget`. The server-side precedence of the resolved value (`generation` < `thinking_mode` < request) is in `docs\howto\src\kumaru\agent\chat.md`.

**Inbound messages are lenient but lossy.** `ChatMessageIn` has `extra="ignore"`, so `reasoning_content`, `tool_calls` and anything else a client echoes back are **dropped**. That is the intended behaviour for reasoning: thoughts are not fed back to the model (see `docs\howto\src\kumaru\core\types.md`). `content: null` becomes `""`.

**Reasoning is a separate channel on the way out.**

- **Non-streaming:** `message.reasoning_content` is present only when the model produced reasoning.
- **Streaming:** reasoning travels in `delta.reasoning_content` frames, which precede the `delta.content` frames.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `ChatMessageIn` | `ChatMessageIn(role: Literal["system", "user", "assistant", "tool"], content: str \| None = "", name: str \| None = None, tool_call_id: str \| None = None)` | One inbound message; `extra="ignore"`. |
| `ChatMessageIn.to_domain` | `to_domain(self) -> Message` | `Message(role=Role(role), content=content or "", name=..., tool_call_id=...)`. Never sets `reasoning`. |
| `ChatCompletionRequest` | `ChatCompletionRequest(model: str \| None = None, messages: list[ChatMessageIn], stream: bool = False, max_tokens: int \| None = None, max_completion_tokens: int \| None = None, temperature: float \| None = None, top_p: float \| None = None, top_k: int \| None = None, min_p: float \| None = None, presence_penalty: float \| None = None, repetition_penalty: float \| None = None, stop: list[str] \| str \| None = None, seed: int \| None = None, thinking: bool \| None = None, thinking_budget: int \| None = None, chat_template_kwargs: dict[str, Any] \| None = None, reasoning_effort: str \| None = None)` | `POST /v1/chat/completions` body; `extra="ignore"`. Bounds: `max_tokens` and `max_completion_tokens` ≥ 1 (**no upper bound**), `temperature` 0–2, `top_p` 0–1, `top_k` ≥ 0, `min_p` 0–1, `presence_penalty` −2–2, `repetition_penalty` 0.5–2, `thinking_budget` ≥ 0. `model` is accepted and ignored: the server answers with whatever is loaded. |
| `ChatCompletionRequest.resolved_thinking` | `resolved_thinking(self) -> bool \| None` | `thinking`, then `chat_template_kwargs["enable_thinking"]`, then `reasoning_effort`, then `None`. |
| `ChatCompletionRequest.generation_overrides` | `generation_overrides(self) -> dict[str, Any]` | Keys `max_tokens` (`max_tokens or max_completion_tokens`), `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`, `stop` (a `str` becomes `[str]`), `seed`, `thinking` (`resolved_thinking()`), `thinking_budget`. |
| `ChatCompletionRequest.to_domain` | `to_domain(self) -> list[Message]` | Converts every message. |
| `UsageOut` | `UsageOut(prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0)` | Usage block. |
| `ChoiceOut` | `ChoiceOut(index: int = 0, message: dict[str, Any], finish_reason: str = "stop")` | One non-streaming choice. |
| `ChatCompletionResponse` | `ChatCompletionResponse(id: str, object: str = "chat.completion", created: int, model: str, choices: list[ChoiceOut], usage: UsageOut, kumaru: dict[str, Any] = {})` | Non-streaming body. |
| `ChatCompletionResponse.from_result` | `from_result(cls, result: ChatResult) -> ChatCompletionResponse` (classmethod) | `message = {"role": "assistant", "content": ...}`, plus `reasoning_content` when it is non-empty. `kumaru` = `latency_ms`, `time_to_first_token_ms` (`None` if 0 or missing), `tokens_per_second`. |
| `sse_chunk` | `sse_chunk(*, completion_id: str, model: str, delta: str \| None = None, reasoning: str \| None = None, role: str \| None = None, finish_reason: str \| None = None, extra: dict[str, Any] \| None = None) -> dict[str, Any]` | One `chat.completion.chunk` body; framing as `data: ...` is the route's job. With `role` set, the delta is `{"role": role}` **only**. Otherwise it contains `reasoning_content` if `reasoning` is truthy and `content` if `delta is not None`, so `""` is emitted. `extra` becomes a top-level `kumaru` key. `created` is taken per call. |
| `new_completion_id` | `new_completion_id() -> str` | `"chatcmpl-" + uuid4().hex[:24]`; one id per stream. |
| `ModelCard` | `ModelCard(id: str, object: str = "model", created: int = <now>, owned_by: str = "kumaru")` | One `/v1/models` entry. |
| `ModelList` | `ModelList(object: str = "list", data: list[ModelCard])` | `/v1/models` body. |
| `HealthResponse` | `HealthResponse(status: Literal["ok", "loading", "error"], version: str, backend: str, model: str, loaded: bool, detail: str \| None = None)` | `/healthz` body. |
| `ErrorBody` / `ErrorResponse` | `ErrorBody(message: str, type: str, hint: str \| None = None)`; `ErrorResponse(error: ErrorBody)` | Documents the error shape. Not referenced by the server: `app.py`'s handler builds the same dict directly. |

## How to use it

### Parse a request, including a thinking switch

```python
from kumaru.server.schemas import ChatCompletionRequest

body = ChatCompletionRequest(
    messages=[{"role": "user", "content": "Prove sqrt(2) is irrational."}],
    max_completion_tokens=8000,
    chat_template_kwargs={"enable_thinking": True},   # vLLM-style switch
    thinking_budget=2048,
)
ov = body.generation_overrides()
assert ov["thinking"] is True and ov["max_tokens"] == 8000 and ov["thinking_budget"] == 2048
assert ov["temperature"] is None          # not sent, so the server's value is kept
messages = body.to_domain()
```

Equivalent switches: `{"thinking": true}`, `{"reasoning_effort": "high"}`. To force thinking off: `{"thinking": false}` or `{"reasoning_effort": "none"}`.

### Convert a result, with reasoning, to the response body

```python
from kumaru.core.types import ChatResult, Message, Usage
from kumaru.server.schemas import ChatCompletionResponse

result = ChatResult(
    model="echo",
    message=Message.assistant("42", reasoning="6 * 7 = 42"),
    usage=Usage(prompt_tokens=30, completion_tokens=12),
    latency_ms=250.0,
)
body = ChatCompletionResponse.from_result(result).model_dump()
assert body["choices"][0]["message"] == {"role": "assistant", "content": "42", "reasoning_content": "6 * 7 = 42"}
```

### Build SSE frames in the order the route sends them

```python
import json
from kumaru.server.schemas import new_completion_id, sse_chunk

cid = new_completion_id()

def frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

frames = [
    frame(sse_chunk(completion_id=cid, model="echo", role="assistant")),          # {"role":"assistant"}
    frame(sse_chunk(completion_id=cid, model="echo", reasoning="Let me think")),  # {"reasoning_content":...}
    frame(sse_chunk(completion_id=cid, model="echo", delta="Answer.")),           # {"content":...}
    frame(sse_chunk(completion_id=cid, model="echo", finish_reason="stop",
                    extra={"latency_ms": 812.0, "thinking_ms": 640.2})),          # {} + top-level "kumaru"
    "data: [DONE]\n\n",
]
```

A real transcript captured from the server is in `docs\howto\src\kumaru\server\routes_chat.md`.

### Consume the two channels as a client

`ui\app.js` does the following for each `data:` line:

1. Skip `[DONE]`.
2. Throw on `evt.error`.
3. Append `delta.reasoning_content` to the thought panel and `delta.content` to the answer.
4. Record `finish_reason`.
5. Read the timings from `evt.kumaru`, including `thinking_ms`.

## How to extend it

**Add request fields compatibly.** Make them optional, default them to `None`, keep `extra="ignore"`, and bound them with `Field(...)`. To make a new field a generation override:

1. Add it to `GenerationParams` (`docs\howto\src\kumaru\core\types.md`).
2. Add it here with the same bounds.
3. Add it to `generation_overrides()`.

For example, a `frequency_penalty`:

```python
class ChatCompletionRequest(BaseModel):
    ...
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)

    def generation_overrides(self) -> dict[str, Any]:
        stop = [self.stop] if isinstance(self.stop, str) else self.stop
        return {
            ...,
            "frequency_penalty": self.frequency_penalty,
        }
```

Every key returned by `generation_overrides()` must be a `GenerationParams` field. `merged()` runs no validation, so a stray key is silently attached to the params object, left out of `model_dump()`, and ignored by every backend.

**Add response metadata additively.** Put it in the `kumaru` dict, or in `extra` for the final SSE frame. Never rename or remove an OpenAI field.

**Accept another thinking spelling.** Add a branch to `resolved_thinking()` at the right precedence, and add a test in `tests\test_server.py`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| HTTP 422 `literal_error` at `["body","messages",0,"role"]` | The role is not `system`/`user`/`assistant`/`tool` (for example `developer` or `function`). | Map the role on the client. |
| HTTP 422 `string_type` at `["body","messages",i,"content"]` | OpenAI content-parts arrays (`[{"type":"text","text":...}]`) are not supported. | Send `content` as a string. |
| HTTP 422 `less_than_equal`, `greater_than_equal`, etc. | A sampling field is out of bounds (for example `temperature` 3, or `thinking_budget` −1). | Respect the bounds in the table. |
| A huge `max_tokens` (200000) is accepted | There is no upper bound here. `ChatAgent` trims it to the room the prompt leaves in the context window before calling the backend (`docs\howto\src\kumaru\agent\chat.md`). | Nothing breaks, but the reply may run until the window is full. Send a realistic cap to bound latency. |
| Server defaults are "ignored" | The client sends explicit values; SDKs often send `temperature`/`top_p` by default. | Omit the fields, or configure the SDK not to send them. |
| `max_completion_tokens` ignored | `max_tokens` is also set and wins. | Send one of them. |
| Thinking stays **off** with `"enable_thinking": "enabled"` | Only the strings `"true"`, `"1"`, `"yes"` and `"on"` mean on. | Send a JSON boolean. |
| Thinking turned **on** with `reasoning_effort: "minimal"` or `"low"` | Every value except `"none"` means on; only an empty string falls through to the server default. | Use `"none"` or `thinking: false`. |
| `reasoning_content` sent back in history has no effect | `ChatMessageIn` ignores it, by design. | Nothing to fix; old thoughts are deliberately not replayed. |
| The streaming client shows no thoughts | It reads only `delta.content`. | Also read `delta.reasoning_content`. |
| A strict client rejects the response | It does not tolerate the additive `kumaru` or `reasoning_content` fields. | Configure it to ignore unknown fields. |

## Known limitations

- **`max_tokens` has no upper bound.** `max_tokens` and `max_completion_tokens` are only `ge=1`, and `merged()` does not re-validate, so the 131,072 ceiling on `GenerationParams.max_tokens` never applies to HTTP requests. The only cap is `ChatAgent`'s trim to the room the prompt leaves in the context window. One client can therefore hold a generation slot (`server.max_concurrent_requests`) until the window is full. Workaround: enforce a ceiling in your clients or a reverse proxy. A smaller `backend.max_context_length` lowers the worst case.

## Related files

- `src\kumaru\server\routes_chat.py` — uses these models; owns SSE framing, error frames and auth.
- `src\kumaru\core\types.py` — `Message` (`reasoning`, alias `reasoning_content`), `GenerationParams`, `ChatResult`.
- `src\kumaru\agent\chat.py` — `resolve_params()` consumes `generation_overrides()`.
- `src\kumaru\server\app.py` — the error handler that emits the `ErrorBody` shape.
- `ui\app.js` — reference streaming client for both channels.
- `tests\test_server.py` — wire-contract tests, including reasoning frames.
