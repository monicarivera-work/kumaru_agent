# `src\kumaru\server\routes_chat.py`

> The HTTP endpoints: OpenAI-compatible chat completions (JSON or SSE, with thoughts in a separate `reasoning_content` channel), plus model listing, status, hot-swap, health and metrics.

**Read this when:** you call the API from a client, change an endpoint, debug streaming or thinking output, or need to know which HTTP status a failure produces.

---

## What it does

`create_app()` in `src\kumaru\server\app.py` mounts this file's `router`.

| Method | Path | Auth | Purpose | Statuses |
|---|---|---|---|---|
| `POST` | `/v1/chat/completions` | Yes, when `server.api_key` is set | One chat turn. Body: `ChatCompletionRequest` (`docs\howto\src\kumaru\server\schemas.md`). Returns a `chat.completion` JSON, or with `"stream": true` an SSE stream of `chat.completion.chunk` frames. | 200, 400, 401, 413, 422, 500, 503 |
| `GET` | `/v1/models` | No | `{"object":"list","data":[{"id":<backend.model>,"object":"model","created":<now>,"owned_by":"kumaru"}]}`; always exactly one model. | 200 |
| `GET` | `/v1/info` | No | The UI's status panel in one round trip: readiness, registered backends and tools, safe config, metrics, and the effective agent. | 200 |
| `POST` | `/v1/backend` | Yes, when set | Hot-swap the backend or model without restarting. Body: any of `name`, `model`, `device`, `dtype`, `load_in_4bit`, `adapter_path`, `base_url`, `model_path`. Only the `transformers` backend can attach an `adapter_path`; every other backend refuses one with a 503 (see Failure modes). | 200, 400, 401, 422, 503 |
| `GET` | `/healthz` | No | Liveness and readiness: `HealthResponse` with `status` `ok`, `loading` or `error`. | 200 (`ok`, `loading`), 503 (`error`) |
| `GET` | `/metrics` | No | In-process counters: `uptime_s`, `requests`, `errors`, `prompt_tokens`, `completion_tokens`, `avg_latency_ms`, `avg_tokens_per_second`. | 200 |

`app.py`, not this file, serves the UI at `/`, `/static/*`, and FastAPI's `/docs` and `/openapi.json`.

## Why it exists

- **OpenAI's wire format.** Every OpenAI SDK, LangChain-style client and eval harness works when pointed at `http://<host>:<port>/v1`. Kumaru-only data is additive: the top-level `kumaru` stats object and `reasoning_content`.
- **SSE, not WebSockets.** Generation is one-directional. SSE works with plain `fetch`, survives proxies, and needs no connection state.
- **Thoughts in their own channel.** Reasoning travels in `reasoning_content`, the field vLLM, llama.cpp and DeepSeek use, never inside `content`.
  - Clients that don't know the field see a clean answer.
  - The UI can show thoughts in a collapsible panel.
  - Most importantly, a client that saves `content` into history never replays old thoughts to the model. Reasoning models are trained on histories that contain final answers only, and stale thoughts would waste context.
- **Admission control.** Every generation runs inside `state.semaphore` (`server.max_concurrent_requests`). Overload becomes a queue instead of a GPU out-of-memory crash or a thrashing KV cache.
- **`/healthz` returns 200 while loading.** Loading a 35B model can take minutes, and an orchestrator must not kill the process for being slow; the `status` field distinguishes the states.
- **Stats in the final SSE frame.** The UI gets time-to-first-token, thinking time and throughput without a second request.

## Mental model

### The request pipeline (`POST /v1/chat/completions`)

1. **Auth** (`require_auth`). It runs before body-schema errors are reported, so a bad body without a key gets 401. Unparseable JSON is the exception: it fails earlier, with 422.
2. **Schema validation.** FastAPI returns 422 with `{"detail": [...]}`, not Kumaru's error envelope.
3. **`state.ensure_ready()`.**
   - While the model is still loading, the request **waits**, for up to 600 s.
   - After that wait, or if the load failed, it gets a 503.
4. **Empty `messages`** → 400. This check comes *after* step 3, so during loading even an invalid request waits for the model first.
5. **Translation.** `body.generation_overrides()` and `body.to_domain()`. The `model` field is ignored: the server answers with whatever is loaded.
6. **Non-streaming path.** `async with state.semaphore` → `agent.complete()` → `metrics.record()` → `ChatCompletionResponse`. A `KumaruError` maps to its class's status.
7. **Streaming path.** The `StreamingResponse` is returned at once, so **HTTP 200 and the headers are committed before any work starts**. `_stream_sse` then:
   1. acquires the semaphore. A queued client sees headers but no frames.
   2. sends the role frame.
   3. iterates `agent.stream()`, where the context-budget check runs.
   4. emits the reasoning and content frames, then the final frame, and records metrics.
   5. sends `[DONE]`.

   From step 7 on, every failure becomes an **error frame inside a 200 response**.

### Stream grammar

```
data: {...,"delta":{"role":"assistant"},...}            exactly one, sent when a slot is acquired
data: {...,"delta":{"reasoning_content":"..."},...}     zero or more, while the model thinks
data: {...,"delta":{"content":"..."},...}               zero or more, the answer
data: {...,"delta":{},"finish_reason":"stop"|"length"|...,"kumaru":{stats}}
data: [DONE]
```

- Frames carry the backend's chunks in order; the route never reorders them. Reasoning precedes content because models think first.
- A chunk with both fields becomes one frame with both keys.
- Empty chunks are not forwarded.
- **On an error**, the remaining frames are replaced by `data: {"error":{"message","type","hint"}}`, then `data: [DONE]`. The `hint` key is present only for `KumaruError`s. Errors count in `metrics.errors`.
- **On a client disconnect**, generation stops at the next chunk boundary.
  - Observed under uvicorn: no further frames, no `streamed:` log line, nothing added to `/metrics`.
  - The per-chunk `request.is_disconnected()` check, which logs `client disconnected; aborting generation`, is a fallback for servers that keep iterating a dead stream.
- **The semaphore slot is held until the stream ends.** A long thinking stream occupies a slot for its full duration.

### The `kumaru` stats

| Field | Final SSE frame | Non-streaming body |
|---|---|---|
| `latency_ms` | From the start of the stream generator to `finish_reason`. **Includes the semaphore wait and `prepare()`.** | Measured inside `backend.generate()`. Excludes the queue wait. |
| `time_to_first_token_ms` | To the first reasoning **or** content token, from the same start. `null` if 0. | Same definition, backend-measured. `null` if 0 or missing. |
| `thinking_ms` | First reasoning chunk to first content chunk, or to the finish if the model never answered. `null` without reasoning. | Not reported. |
| `prompt_tokens`, `completion_tokens` | From the backend's last `usage` chunk; **0** if it never sent one. | In `usage`. If the backend sent none, `generate()` estimates (characters ÷ 4). |
| `tokens_per_second` | `completion_tokens / latency`, end to end, so queueing, prefill and thinking all lower it. | `ChatResult.tokens_per_second`. |

Built-in backends count thought tokens as completion tokens, because they are generated tokens.

### Thinking resolution

The request's `thinking`, `chat_template_kwargs.enable_thinking` or `reasoning_effort` resolve to one `thinking` override (`docs\howto\src\kumaru\server\schemas.md`). `ChatAgent.resolve_params` then layers `generation` < `thinking_mode` (only when thinking resolves on) < request fields (`docs\howto\src\kumaru\agent\chat.md`).

Captured with the echo backend, which reports the `max_tokens` and `temperature` it received. The config was `generation: {max_tokens: 256, temperature: 0.7, thinking: false}` and `thinking_mode: {max_tokens: 4000, temperature: 1.0}`.

| Request adds | Echo reports | Why |
|---|---|---|
| nothing | `max_tokens=256, temperature=0.7` | `generation` only. |
| `"thinking": true` | `max_tokens=4000, temperature=1.0` | The `thinking_mode` preset applies. |
| `"thinking": true, "temperature": 0.2` | `max_tokens=4000, temperature=0.2` | A request field beats the preset. |
| `"reasoning_effort": "high"` | `max_tokens=4000, temperature=1.0` | Any effort except `"none"` means on. |
| `"chat_template_kwargs": {"enable_thinking": true}, "max_tokens": 50` | `max_tokens=50, temperature=1.0` | A request field beats the preset. |

## Public API

The public surface is the HTTP contract above. In Python:

| Name | Signature | What it's for |
|---|---|---|
| `router` | `router = APIRouter()` | Holds the six routes; included by `create_app()`. |
| `get_state` | `get_state(request: Request) -> AppState` | `request.app.state.kumaru`. Use it in every route. |
| `require_auth` | `async require_auth(request: Request, authorization: str \| None = Header(default=None)) -> None` | Bearer gate, attached as `dependencies=[Depends(require_auth)]`. A no-op when `server.api_key` is empty. Otherwise it strips a leading `"Bearer "` (case-sensitive) and compares; a mismatch raises `AuthError` (401) with hint ``Send 'Authorization: Bearer <key>'.``. |
| `chat_completions` | `async chat_completions(body: ChatCompletionRequest, request: Request)` | `POST /v1/chat/completions`. Returns `ChatCompletionResponse` or a `StreamingResponse` with `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`. |
| `_stream_sse` | `async _stream_sse(state: AppState, agent: Any, messages: list, overrides: dict, request: Request) -> AsyncIterator[str]` | Private. Renders `agent.stream()` as SSE text; owns the slot, the stats, metrics recording, error frames and `[DONE]`. |
| `list_models` | `async list_models(request: Request) -> ModelList` | `GET /v1/models`. |
| `info` | `async info(request: Request) -> dict` | `GET /v1/info`. Keys: `version`, `ready`, `loading`, `error`, `available_backends` (`BACKENDS.names()`), `available_tools` (`TOOLS.names()`), `config` (`backend` without `api_key`; `generation`; `thinking_mode` with `None` values kept; `agent`), `metrics`, and `agent` (`ChatAgent.describe()`, only once loaded). The `server` section is never included. |
| `swap_backend` | `async swap_backend(payload: dict, request: Request) -> dict` | `POST /v1/backend`. Keeps allowed keys with non-`None` values, logs `swapping backend: ...`, awaits `AppState.swap_backend(**overrides)`, then returns `{"ok": true, "backend": <backend config without api_key>}`. |
| `healthz` | `async healthz(request: Request) -> JSONResponse` | `GET /healthz`. `backend` and `model` are the configured values, so they are available while loading. |
| `metrics` | `async metrics(request: Request) -> dict` | `GET /metrics`: `state.metrics.snapshot()`. |

## How to use it

Examples assume `kumaru serve` on the default `127.0.0.1:8080`. Omit the `Authorization` header when `server.api_key` is unset.

### Non-streaming (PowerShell)

```powershell
$body = @{ messages = @(@{ role = "user"; content = "Say hello" }); max_tokens = 64 } | ConvertTo-Json -Depth 10
$r = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/chat/completions `
  -ContentType "application/json" -Body $body -Headers @{ Authorization = "Bearer <api_key>" }
$r.choices[0].message.content
```

`-Depth 10` matters: `ConvertTo-Json` defaults to depth 2 and would flatten the nested message objects to strings.

### Streaming with thinking (`curl.exe`)

Write the body to a file. Inline JSON quoting differs between Windows PowerShell 5.1 and PowerShell 7, and `"@body.json"` must be quoted because a bare `@` is PowerShell's splatting operator.

```powershell
'{"messages":[{"role":"user","content":"hi"}],"stream":true,"thinking":true,"thinking_budget":2}' |
  Set-Content -NoNewline -Encoding ascii body.json
curl.exe -sN http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" `
  -H "Authorization: Bearer <api_key>" --data-binary "@body.json"
```

Real output from `kumaru serve -c test` (echo backend), with 15 content frames elided. `thinking_budget: 2` keeps two of echo's eight-character thought pieces. Echo timings are meaningless.

```text
data: {"id":"chatcmpl-96a530952cd54503b5355b1b","object":"chat.completion.chunk","created":1791658943,"model":"echo","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-96a530952cd54503b5355b1b","object":"chat.completion.chunk","created":1791658943,"model":"echo","choices":[{"index":0,"delta":{"reasoning_content":"The user"},"finish_reason":null}]}

data: {"id":"chatcmpl-96a530952cd54503b5355b1b","object":"chat.completion.chunk","created":1791658943,"model":"echo","choices":[{"index":0,"delta":{"reasoning_content":" asked: "},"finish_reason":null}]}

data: {"id":"chatcmpl-96a530952cd54503b5355b1b","object":"chat.completion.chunk","created":1791658943,"model":"echo","choices":[{"index":0,"delta":{"content":"[echo:"},"finish_reason":null}]}

data: {"id":"chatcmpl-96a530952cd54503b5355b1b","object":"chat.completion.chunk","created":1791658943,"model":"echo","choices":[{"index":0,"delta":{"content":"8f4343"},"finish_reason":null}]}

  ... 15 more content frames ...

data: {"id":"chatcmpl-96a530952cd54503b5355b1b","object":"chat.completion.chunk","created":1791658943,"model":"echo","choices":[{"index":0,"delta":{"content":"0."},"finish_reason":null}]}

data: {"id":"chatcmpl-96a530952cd54503b5355b1b","object":"chat.completion.chunk","created":1791658943,"model":"echo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"kumaru":{"latency_ms":5.6,"time_to_first_token_ms":0.7,"thinking_ms":0.6,"prompt_tokens":218,"completion_tokens":20,"tokens_per_second":3566.27}}

data: [DONE]
```

### Non-streaming with thinking

Real response, pretty-printed. The request was the same as above with `"stream": false`:

```json
{
  "id": "chatcmpl-8e5bc974c4e341eebb253353",
  "object": "chat.completion",
  "created": 1791658559,
  "model": "echo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "[echo:8f434346] You said: hi\nContext: 2 message(s), ~218 prompt tokens, max_tokens=256, temperature=0.0.",
        "reasoning_content": "The user asked: "
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 218, "completion_tokens": 20, "total_tokens": 238},
  "kumaru": {"latency_ms": 0.1, "time_to_first_token_ms": 0.1, "tokens_per_second": 176522.52}
}
```

Without thinking, `reasoning_content` is absent rather than empty.

### Consume both channels (Python)

```python
import json
import httpx

def chat_stream(messages, *, base="http://127.0.0.1:8080", api_key=None, **params):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {"messages": messages, "stream": True, **params}
    with httpx.stream("POST", f"{base}/v1/chat/completions", json=body, headers=headers, timeout=600) as res:
        res.raise_for_status()                      # 4xx/5xx: failed before the stream started
        for line in res.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                return
            evt = json.loads(data)
            if "error" in evt:                      # failed after the 200 was sent
                raise RuntimeError(f"{evt['error']['type']}: {evt['error']['message']}")
            choice = evt["choices"][0]
            if text := choice["delta"].get("reasoning_content"):
                yield "thought", text
            if text := choice["delta"].get("content"):
                yield "answer", text
            if choice["finish_reason"]:
                yield "done", {"finish_reason": choice["finish_reason"], **evt["kumaru"]}

for kind, value in chat_stream([{"role": "user", "content": "hi"}], thinking=True):
    print(kind, value)
```

Store only the `answer` text in your history; the server ignores `reasoning_content` in incoming messages anyway.

### With the OpenAI Python SDK

The SDK is not a Kumaru dependency (`pip install openai`). It requires a non-empty `api_key`, so pass any string when the server has none. `model` is required by the SDK and ignored by Kumaru. Send Kumaru-only fields through `extra_body`; the response object exposes unknown fields as attributes.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="<api_key>")
resp = client.chat.completions.create(
    model="kumaru",
    messages=[{"role": "user", "content": "Is 221 prime?"}],
    extra_body={"thinking": True, "thinking_budget": 1024},
)
print(resp.choices[0].message.reasoning_content)
print(resp.choices[0].message.content)
```

### Error responses

| Status | Body | When |
|---|---|---|
| 200 + error frame | `data: {"error": {...}}` then `data: [DONE]` | Any failure after a stream started, including a context overflow. |
| 400 | `{"error": {"message", "type": "RequestError", "hint"}}` | Empty `messages`. `/v1/backend` with no allowed, non-null field. |
| 401 | `type: "AuthError"` | `server.api_key` set and the header does not match. |
| 413 | `type: "ContextLengthExceededError"` | Non-streaming only: `agent.memory_policy: none` and history over budget. |
| 422 | FastAPI `{"detail": [...]}` | Schema violation or unparseable JSON. Not counted in `metrics.errors`. |
| 500 | Plain text `Internal Server Error` | An unexpected exception in a non-streaming request; the traceback is in the server log. |
| 503 | `type: "BackendError"` | The load failed; still loading after 600 s; a hot-swap failed; or the backend failed during non-streaming generation. |

Every `KumaruError` maps through its class's `http_status`; the full table is in `docs\howto\src\kumaru\core\errors.md`. Real bodies:

```text
401  {"error":{"message":"Invalid or missing API key.","type":"AuthError","hint":"Send 'Authorization: Bearer <key>'."}}
400  {"error":{"message":"`messages` must contain at least one message.","type":"RequestError","hint":null}}
400  {"error":{"message":"No valid fields supplied.","type":"RequestError","hint":"Allowed: ['adapter_path', 'base_url', 'device', 'dtype', 'load_in_4bit', 'model', 'model_path', 'name']"}}
413  {"error":{"message":"Conversation needs 218 tokens but the budget is 64.","type":"ContextLengthExceededError","hint":"Switch agent.memory_policy to 'sliding_window', or raise the budget."}}
```

The same overflow with `"stream": true` gives HTTP 200 and:

```text
data: {"id":"chatcmpl-45632996624040fdad99bb41","object":"chat.completion.chunk","created":1791659164,"model":"echo","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"error":{"message":"Conversation needs 218 tokens but the budget is 64.","type":"ContextLengthExceededError","hint":"Switch agent.memory_policy to 'sliding_window', or raise the budget."}}

data: [DONE]
```

### Hot-swap, status, health

```powershell
# Swap the model; the old backend is closed first to free VRAM. Blocks until the new one is loaded.
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/backend -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer <api_key>" } -Body '{"name":"echo","model":"echo"}'

Invoke-RestMethod http://127.0.0.1:8080/healthz   # status ok | loading | error
Invoke-RestMethod http://127.0.0.1:8080/metrics
(Invoke-RestMethod http://127.0.0.1:8080/v1/info).agent.thinking_mode
```

A real `/healthz` body: `{"status":"ok","version":"0.2.0","backend":"echo","model":"echo","loaded":true,"detail":null}`.

Real `/v1/info` from `-c test` after one request, with large objects elided as `{...}`:

```json
{
  "version": "0.2.0", "ready": true, "loading": false, "error": null,
  "available_backends": ["echo", "llamacpp", "llamacpp_server", "openai_compat", "transformers"],
  "available_tools": ["current_time"],
  "config": {
    "backend": {...}, "generation": {...},
    "thinking_mode": {"max_tokens": null, "temperature": null, "top_p": null, "top_k": null, "min_p": null,
                      "presence_penalty": null, "repetition_penalty": null, "thinking_budget": null},
    "agent": {...}
  },
  "metrics": {"uptime_s": 0.4, "requests": 1, "errors": 0, "prompt_tokens": 218, "completion_tokens": 18,
              "avg_latency_ms": 273.6, "avg_tokens_per_second": 65.79},
  "agent": {
    "backend": {"name": "echo", "model_id": "echo", "device": "cpu", "dtype": "none", "context_length": 8192,
                "supports_streaming": true, "quantization": null, "extra": {"purpose": "testing and install verification"}},
    "memory_policy": "sliding_window", "max_history_tokens": 2048, "tools": [], "generation": {...}, "thinking_mode": {}
  }
}
```

Lifecycle details (the loading state, the swap semantics, and a `/healthz` wait loop) are in `docs\howto\src\kumaru\server\app.md`.

## How to extend it

**Add an endpoint.**

- Decorate it with `@router.get` or `@router.post`.
- Read shared state with `get_state(request)`.
- Raise `KumaruError` subclasses for operator errors; `app.py` renders them.
- Add `dependencies=[Depends(require_auth)]` if the route spends compute, mutates state or reveals anything sensitive.

```python
@router.get("/v1/readiness")
async def readiness(request: Request) -> dict:
    state = get_state(request)
    return {
        "ready": state.ready,
        "loading": not state.ready and state.load_error is None,
        "error": state.load_error,
        "model": state.config.backend.model,
    }
```

**Add a streaming endpoint.** Copy `_stream_sse`'s shape:

- acquire `state.semaphore` *inside* the generator;
- re-raise `asyncio.CancelledError`;
- turn exceptions into an error frame;
- always end with `data: [DONE]`;
- send the same three headers.

The `X-Accel-Buffering: no` header stops nginx from buffering the whole response.

**Add a stat to the final frame.** Compute it in `_stream_sse` and add it to the `extra` dict. Add the non-streaming equivalent in `ChatCompletionResponse.from_result()`, and render it in `ui\app.js` if the UI should show it.

**Add a request field.** See `docs\howto\src\kumaru\server\schemas.md`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| 401 with a correct key | The prefix is `bearer` (lowercase) or there is extra text. Only the exact `Bearer ` prefix is stripped; a raw key with no prefix *is* accepted. | Send `Authorization: Bearer <api_key>`. |
| 401 although the body is also invalid | Auth is checked before schema errors are reported. | Fix the auth first. Unparseable JSON still gets 422. |
| 422 with `{"detail": [...]}` instead of `{"error": ...}` | Schema validation by FastAPI, not a Kumaru error. | Read `detail[].loc` and `msg`. See the schemas failure table. |
| A request hangs for minutes, then returns 503 `Backend is still loading.` | The model was still loading; requests wait up to 600 s. | Poll `/healthz` until `ok` before sending traffic. |
| An empty-`messages` request returns 503 instead of 400 | `ensure_ready()` runs before the empty check. | Fix the load error first. |
| Every chat returns 503 after a `/v1/backend` call failed | The swap closes the old backend *before* loading the new one (to free VRAM), so a failed swap leaves nothing loaded. The next chat retries the same bad config once. | POST a working config to `/v1/backend`. |
| `/v1/backend` returns 503 `cannot attach a LoRA adapter`, and later swaps fail the same way | `adapter_path` is set for a backend other than `transformers`. The overrides stay in the live config even when the load fails, and `null` values are dropped, so `"adapter_path": null` cannot clear it (400). | Send `"adapter_path": ""` (an empty string clears it) with the rest of the swap. To use the adapter, swap to `"name": "transformers"` with the adapter's base model. |
| 413 for non-streaming, but HTTP 200 + error frame for the same streaming request | The status is committed before the context check runs inside the stream. | Check for `error` frames in every streaming client. |
| A stream ends with `[DONE]` and no final frame | An error frame came first, or a custom backend never yielded a chunk with `finish_reason`. Nothing is recorded in `/metrics`. | Read the error frame. Make the backend end with `StreamChunk(finish_reason=..., usage=...)`. |
| An empty answer with `finish_reason: "length"` and only thoughts | Thinking used the whole `max_tokens`. The UI says "No answer: the max-tokens cap ran out while the model was still thinking." | Raise `max_tokens` (or `thinking_mode.max_tokens`), or set `thinking_budget`. |
| Thoughts arrive inside `content`, for example as `<think>…</think>` | The backend yields thoughts as `delta`; a custom backend that doesn't split them. | Yield `StreamChunk(reasoning=...)`, or use `ReasoningSplitter` from `src\kumaru\backends\base.py`. |
| `completion_tokens: 0` in the final frame | The backend sent no `usage`. The streaming path does not estimate; `generate()` does. | Emit `usage` on the final chunk. |
| Streaming `latency_ms` and `tokens_per_second` look worse than non-streaming | The streaming timer starts before the semaphore wait and `prepare()`, and its tokens/s is end to end. | Expected. Compare like with like. |
| Clients queue, and the first frame arrives late | All `server.max_concurrent_requests` slots are busy; streams hold their slot to the end. | Raise the limit to what the hardware batches (with llama-server, match `backend.llama_server.parallel`), or add replicas. |
| Streaming arrives as one blob | A proxy buffers SSE. | Keep `X-Accel-Buffering: no` for nginx; disable buffering in other proxies. |
| Request `model` has no effect | It is ignored; the loaded model answers. | Use `/v1/backend` to switch. `/v1/models` shows what is loaded. |
| `/v1/backend` returns 200 but nothing changed as intended | Unknown keys are dropped silently, and values are not validated (for example `"load_in_4bit": "yes"` stays a string). | Check the `backend` block in the response or in `/v1/info`. |
| `/v1/info` lacks an installed plugin backend | `BACKENDS.names()` does not scan entry points; plugins appear only after a lookup miss. | See `docs\howto\src\kumaru\core\registry.md`. |
| The system prompt or model paths are readable without a key | `/v1/info`, `/metrics`, `/healthz` and `/v1/models` are unauthenticated by design. `/v1/info` includes `config.agent.system_prompt` and the backend config, but never `api_key` or the `server` section. | Don't expose the port publicly; put an authenticating proxy in front. |
| Generation continues after the user closes the tab | A backend that blocks between chunks. Cancellation only takes effect at a chunk boundary. | Yield regularly; use `iterate_in_thread()` for blocking generators. |

## Known limitations

- **Streaming reports 0 tok/s when the backend sends no usage.** `_stream_sse` takes `completion_tokens` from the last `usage` the backend yielded and, unlike `Backend.generate()`, does not estimate. With no usage, or all-zero usage, the final frame says `completion_tokens: 0` and `tokens_per_second: 0.0`, and the UI shows `0 tok/s`. `/metrics` still adds the request's latency but counts 0 completion tokens, which drags `avg_tokens_per_second` down.
  - `echo`, `transformers` and `llamacpp` count tokens themselves.
  - `openai_compat` and `llamacpp_server` relay the upstream's usage, and send zeros when the upstream sends none (for example, a server that ignores `stream_options.include_usage`).

  Workaround: measure with non-streaming requests, or make a custom backend put `usage` on its final chunk.

## Related files

- `src\kumaru\server\app.py` — `AppState` (semaphore, `ensure_ready`, `swap_backend`, `Metrics`), the error handler, mounting.
- `src\kumaru\server\schemas.py` — request, response and SSE chunk shapes; `resolved_thinking()`.
- `src\kumaru\agent\chat.py` — `complete()`, `stream()`, `resolve_params()`.
- `src\kumaru\core\errors.py` — status codes per exception.
- `src\kumaru\core\registry.py` — `available_backends` and `available_tools`.
- `ui\app.js` — the reference client: SSE parsing, the thought panel, stats, the empty-answer message.
- `tests\test_server.py` — endpoint, streaming, reasoning and error-status tests.
