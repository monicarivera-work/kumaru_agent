# `src/kumaru/server/schemas.py`

> Pydantic wire models for the Kumaru HTTP API.

**Read this when:** building clients, changing API fields, or understanding validation errors.

---

## What it does
It defines request, response, history, health, and error schemas. It also translates between wire `MessageModel` objects and internal `CoreMessage` values.

## Why it exists
Internal dataclasses can evolve without silently breaking browser clients. Field constraints reject bad requests before they reach a model backend.

## Mental model
Wire models are the API contract; core types are internal. Convert explicitly at the boundary.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `MessageModel` | `MessageModel(role: Literal["system", "user", "assistant"], content: str)` | One wire message. |
| `MessageModel.to_core` | `to_core(self) -> CoreMessage` | Convert to internal message. |
| `MessageModel.from_core` | `from_core(cls, message: CoreMessage) -> MessageModel` | Convert internal message to wire model. |
| `ChatRequestModel` | `ChatRequestModel(message: str, session_id: str = "default")` | Body for chat endpoints; validates length. |
| `UsageModel` | `UsageModel(prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0)` | Usage response model. |
| `ChatResponseModel` | `ChatResponseModel(reply: str, session_id: str, usage: UsageModel = UsageModel())` | Non-streamed chat response. |
| `HistoryResponseModel` | `HistoryResponseModel(session_id: str, messages: list[MessageModel] = ...)` | Conversation history response. |
| `HealthModel` | `HealthModel(status: Literal["ok", "degraded"] = "ok", version: str, backend: str, model: str | None = None, ready: bool = True, detail: dict[str, object] = ...)` | Health response. |
| `ErrorModel` | `ErrorModel(error: str, message: str, hint: str | None = None)` | API error response. |

## How to use it
```python
from kumaru.server.schemas import ChatRequestModel
body = ChatRequestModel(message="hello", session_id="s1")
```

```python
from kumaru.server.schemas import MessageModel
wire = MessageModel(role="user", content="hi")
core = wire.to_core()
```

```python
from kumaru.core.types import Message
from kumaru.server.schemas import MessageModel
wire = MessageModel.from_core(Message.assistant("ok"))
```

```python
from kumaru.server.schemas import UsageModel
usage = UsageModel(prompt_tokens=1, completion_tokens=2, total_tokens=3)
```

## How to extend it
Add fields in a backwards-compatible way where possible: give new response fields defaults and keep request constraints explicit. If an internal type changes, update conversion methods here.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| HTTP 422 response | Pydantic validation failed, often message/session length. | Send non-empty `message` and a 1-128 char `session_id`. |
| Role validation fails | Role is not one of the three literals. | Use `system`, `user`, or `assistant`. |
| Browser/client mismatch | Schema changed without route/UI update. | Update `routes_chat.py` and `ui/app.js` together. |

## Related files
- [Chat routes](./routes_chat.md)
- [Server app](./app.md)
- [Core types](../core/types.md)
