# `src/kumaru/server/routes_chat.py`

> FastAPI chat, streaming, history, and clear-history endpoints.

**Read this when:** integrating API clients or debugging streamed responses.

---

## What it does
It defines `/api/chat`, `/api/chat/stream`, `/api/history/{session_id}`, and `DELETE /api/history/{session_id}`. Streaming uses SSE frames containing JSON payloads for deltas, final usage, or errors.

## Why it exists
SSE is simple, proxy-friendly, and works with browser `fetch` streams for POST bodies. Routes keep the HTTP protocol separate from `ChatAgent` orchestration.

## Mental model
Routes retrieve the single process-wide agent from `request.app.state.agent`. Streaming catches `KumaruError` and emits an error frame instead of leaking tracebacks.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `router` | `APIRouter` | Router mounted under `/api`. |
| `get_agent` | `get_agent(request: Request) -> ChatAgent` | Return app-state agent. |
| `chat` | `chat(body: ChatRequestModel, request: Request) -> ChatResponseModel` | Non-streamed chat endpoint. |
| `chat_stream` | `chat_stream(body: ChatRequestModel, request: Request) -> StreamingResponse` | SSE streamed chat endpoint. |
| `history` | `history(session_id: str, request: Request) -> HistoryResponseModel` | Return session history. |
| `clear_history` | `clear_history(session_id: str, request: Request) -> dict[str, str]` | Clear one session. |

## How to use it
```bash
curl -s http://127.0.0.1:8000/api/health
```

```bash
curl -s -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","session_id":"docs"}'
```

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","session_id":"docs"}'
```

```bash
curl -X DELETE http://127.0.0.1:8000/api/history/docs
```

## How to extend it
Add new chat-related endpoints to this router and keep request/response models in `schemas.py`. For another streaming event type, emit it through `_sse({...})` and update `ui/app.js` parsing.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Stream arrives all at once | Proxy/server buffering. | Preserve `X-Accel-Buffering: no` and disable buffering upstream. |
| Mid-stream error shown as event | Backend raised `KumaruError`. | Read `message`/`hint`; server remains alive. |
| `AttributeError: state.agent` | App lifespan did not run or route used outside app factory. | Use `create_app()` under an ASGI server/test client with lifespan support. |

## Related files
- [Server app](./app.md)
- [Schemas](./schemas.md)
- [Chat agent](../agent/chat.md)
- [UI app](../../../ui/app.js.md)
