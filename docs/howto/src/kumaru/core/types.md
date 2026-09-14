# `src/kumaru/core/types.py`

> Immutable value types shared across CLI, agent, backends, and server translations.

**Read this when:** constructing messages, backend requests, streamed chunks, or usage reports.

---

## What it does
It defines roles, messages, usage accounting, stream chunks, and chat requests. These types are SDK-agnostic, so every layer can exchange the same objects.

## Why it exists
Backends should not care whether input came from the browser, CLI, or tests. Plain dataclasses and a string enum keep the wire format simple while protecting internal code from mutation surprises.

## Mental model
`Message` objects are immutable and normalize string roles to `Role`. `ChatRequest` carries per-request sampling so one backend instance can serve different generation settings.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `Role` | `class Role(str, Enum)` | `system`, `user`, and `assistant` roles. |
| `Message` | `Message(role: Role, content: str)` | One immutable conversation turn. |
| `Message.to_dict` | `to_dict(self) -> dict[str, str]` | OpenAI-style dict. |
| `Message.from_dict` | `from_dict(cls, data: dict[str, Any]) -> Message` | Build from OpenAI-style dict. |
| `Message.system` | `system(cls, content: str) -> Message` | Create system message. |
| `Message.user` | `user(cls, content: str) -> Message` | Create user message. |
| `Message.assistant` | `assistant(cls, content: str) -> Message` | Create assistant message. |
| `Usage` | `Usage(prompt_tokens: int = 0, completion_tokens: int = 0)` | Token accounting. |
| `Usage.total_tokens` | `total_tokens(self) -> int` | Prompt plus completion tokens. |
| `Usage.to_dict` | `to_dict(self) -> dict[str, int]` | JSON-safe usage dict. |
| `StreamChunk` | `StreamChunk(delta: str = "", done: bool = False, usage: Usage | None = None)` | One streamed reply event. |
| `ChatRequest` | `ChatRequest(messages: list[Message] = ..., max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95, stop: list[str] = ...)` | Backend generation request. |
| `ChatRequest.as_dicts` | `as_dicts(self) -> list[dict[str, str]]` | Messages in OpenAI wire format. |

## How to use it
```python
from kumaru.core.types import Message
msg = Message.user("Explain SSE")
print(msg.to_dict())
```

```python
from kumaru.core.types import Message
msg = Message.from_dict({"role": "assistant", "content": "ok"})
print(msg.role.value)
```

```python
from kumaru.core.types import ChatRequest, Message
req = ChatRequest(messages=[Message.system("Be brief"), Message.user("Hi")], stop=["END"])
print(req.as_dicts())
```

```python
from kumaru.core.types import StreamChunk, Usage
chunks = [StreamChunk(delta="hi"), StreamChunk(done=True, usage=Usage(2, 1))]
```

## How to extend it
Add fields only when all backends can safely ignore or consume them. For example, a future `tools` field belongs on `ChatRequest` if it is per-request and backend-facing.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| `ValueError: ... is not a valid Role` | Role string is not `system`, `user`, or `assistant`. | Validate/translate external input before constructing `Message`. |
| Cannot assign `msg.content` | `Message` is frozen. | Create a new `Message` with changed content. |
| Usage shows zero tokens | Backend did not report usage. | Treat zero as unknown unless the backend documents exact counts. |

## Related files
- [Prompt assembly](../agent/prompt.md)
- [Backend base](../backends/base.md)
- [Server schemas](../server/schemas.md)
