# `src\kumaru\server\routes_chat.py`

> OpenAI-compatible chat, streaming, health, metrics, model, info, and backend-control endpoints.

**Read this when:** you are changing HTTP behavior, debugging streaming, wiring a new client, adding an endpoint, or investigating API-key and readiness behavior.

---

## What it does
This file defines the routes mounted by `create_app()`. It serves OpenAI-style chat completions, wraps streaming output as Server-Sent Events, exposes health and metrics, lists the configured model, reports UI status, and hot-swaps the backend. Streaming uses `fetch`-compatible SSE frames: `data: {json}\n\n` chunks followed by `data: [DONE]`.

## Why it exists
Kumaru wants compatibility with existing OpenAI clients and a simple browser UI. SSE is enough because generation is one-way; WebSockets would add stateful connection complexity without benefit. The route layer also enforces bounded admission by acquiring the app semaphore around generation, so overload becomes queueing instead of process-killing GPU exhaustion.

## Mental model
- Requests are stateless: every chat request must send the full conversation.
- `/healthz` returns 200 while loading and 503 only when loading failed.
- Streaming errors are sent as SSE `{"error": ...}` frames, then `[DONE]`.

## Public API

| Method | Path | Request shape | Response shape | Status codes |
|---|---|---|---|---|
| `POST` | `/v1/chat/completions` | `ChatCompletionRequest`: `messages`, optional `stream`, `max_tokens`, `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `stop`, `seed`. Requires `Authorization: Bearer <server.api_key>` only when configured. | If `stream=false`, `ChatCompletionResponse`. If `stream=true`, `text/event-stream` with OpenAI-style `chat.completion.chunk` frames and final `data: [DONE]`. | `200` success, `422` validation, `413` context length, `503` backend load failure, current generic `KumaruError` cases use their class `http_status` or default `500`. |
| `GET` | `/v1/models` | None. | `ModelList` with one `ModelCard` for `state.config.backend.model`. | `200`. |
| `GET` | `/v1/info` | None. | Dict with `version`, `ready`, `loading`, `error`, `available_backends`, `available_tools`, safe `config`, `metrics`, and optional `agent`. | `200`. |
| `POST` | `/v1/backend` | Dict with any of `name`, `model`, `device`, `dtype`, `load_in_4bit`, `adapter_path`, `base_url`, `model_path`. Requires API key when configured. | `{"ok": true, "backend": ...}` with `api_key` excluded. | `200` success, `503` load failure, `500` no valid fields or invalid API key in current code, `422` malformed JSON. |
| `GET` | `/healthz` | None. | `HealthResponse`: `status`, `version`, `backend`, `model`, `loaded`, `detail`. | `200` for `ok` or `loading`, `503` for `error`. |
| `GET` | `/metrics` | None. | Metrics snapshot: uptime, request/error counts, token totals, averages. | `200`. |

## How to use it

### Non-streaming chat with curl

```powershell
curl.exe -X POST http://127.0.0.1:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}],\"stream\":false,\"max_tokens\":64}"
```

### Streaming chat with Python `httpx`

```python
import json
import httpx

with httpx.stream(
    "POST",
    "http://127.0.0.1:8080/v1/chat/completions",
    json={"messages": [{"role": "user", "content": "Count to three"}], "stream": True},
    timeout=600,
) as res:
    res.raise_for_status()
    for line in res.iter_lines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        evt = json.loads(data)
        print(evt.get("choices", [{}])[0].get("delta", {}).get("content", ""), end="")
```

### Use API-key auth when configured

```powershell
curl.exe -X POST http://127.0.0.1:8080/v1/backend `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer YOUR_KEY" `
  -d "{\"name\":\"echo\",\"model\":\"echo\"}"
```

### Poll readiness and UI info

```python
import httpx

health = httpx.get("http://127.0.0.1:8080/healthz").json()
info = httpx.get("http://127.0.0.1:8080/v1/info").json()
print(health["status"], info["loading"], info["metrics"]["requests"])
```

## How to extend it
Add a route to `router`, use `get_state(request)` for shared state, and return Pydantic-compatible JSON. Apply `Depends(require_auth)` if the route can spend compute or mutate backend state.

Worked example: add a backend readiness detail endpoint.

```python
from fastapi import Request

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

For streaming endpoints, set `media_type="text/event-stream"`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no` so proxies do not buffer the response.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| HTTP 503 from `/healthz` | Backend failed during load. | Read `detail`, fix dependency/model config, or try the `echo` backend. |
| Chat waits before responding | Model is still loading or semaphore queue is full. | Check `/v1/info`; tune `server.max_concurrent_requests` to hardware. |
| Streaming arrives as one big blob | Reverse proxy buffered SSE. | Preserve `X-Accel-Buffering: no`; disable proxy buffering for this path. |
| Browser stop button does not stop GPU work immediately | Disconnect is checked between emitted chunks. | Ensure backend yields chunks regularly and honors cancellation where possible. |
| Invalid API key returns generic server error | `require_auth()` raises base `KumaruError`, whose current `http_status` is 500. | Send `Authorization: Bearer <key>`; consider introducing an auth-specific 401 error. |
| UI shows request failed while stream is open | `_stream_sse()` emitted an `error` frame. | Inspect the frame's `error.message` and server logs. |
| HTTP 422 on chat | Request does not match `ChatCompletionRequest`. | Send `messages` as a non-empty list with valid roles and bounded params. |

## Related files

- `src\kumaru\server\app.py` — creates `AppState`, semaphore, error handler, and mounts this router.
- `src\kumaru\server\schemas.py` — defines request/response/SSE shapes.
- `src\kumaru\agent\chat.py` — performs `complete()` and `stream()`.
- `src\kumaru\core\errors.py` — maps domain failures to HTTP statuses.
- `src\kumaru\core\registry.py` — supplies available backend and tool names for `/v1/info`.
- `ui\app.js` — POSTs streaming chat requests and parses SSE frames with `fetch`.

