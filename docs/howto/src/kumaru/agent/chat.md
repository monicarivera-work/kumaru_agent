# `src/kumaru/agent/chat.py`

> Stateful chat orchestrator that connects memory, prompt assembly, and a backend.

**Read this when:** calling Kumaru from Python, injecting a test backend, or understanding session behavior.

---

## What it does
It converts a user turn into a `ChatRequest`, calls the selected backend, writes successful exchanges to memory, streams chunks when requested, exposes health, and closes backend resources.

## Why it exists
CLI and HTTP code need one object that owns the loop from history to prompt to backend to memory. Keeping tools, retrieval, and planning out of v1 prevents the central class from becoming a catch-all.

## Mental model
Memory is updated only after successful full generation or after a stream iterator is consumed to completion. Aborted or failed generations do not poison history.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `DEFAULT_SESSION` | `str` | Default session id, `default`. |
| `ChatAgent` | `ChatAgent(config: KumaruConfig, *, backend: Backend | None = None, memory: ConversationMemory | None = None)` | Create agent with loaded or injected backend/memory. |
| `ChatAgent.ask_with_usage` | `ask_with_usage(self, user_input: str, *, session_id: str = DEFAULT_SESSION) -> tuple[str, Usage]` | Send one turn and return reply plus usage. |
| `ChatAgent.ask` | `ask(self, user_input: str, *, session_id: str = DEFAULT_SESSION) -> str` | Send one turn and return text. |
| `ChatAgent.stream` | `stream(self, user_input: str, *, session_id: str = DEFAULT_SESSION) -> Iterator[StreamChunk]` | Yield streamed reply chunks. |
| `ChatAgent.reset` | `reset(self, session_id: str | None = None) -> None` | Clear one session or all sessions. |
| `ChatAgent.history` | `history(self, session_id: str = DEFAULT_SESSION) -> list[Message]` | Return session history. |
| `ChatAgent.health` | `health(self) -> dict[str, object]` | Backend health plus session count. |
| `ChatAgent.close` | `close(self) -> None` | Close backend resources. |

## How to use it
```python
from kumaru.agent.chat import ChatAgent
from kumaru.core.config import load_config
with ChatAgent(load_config()) as agent:
    print(agent.ask("hello"))
```

```python
reply, usage = agent.ask_with_usage("hello", session_id="cli")
print(reply, usage.total_tokens)
```

```python
for chunk in agent.stream("hello", session_id="web"):
    print(chunk.delta, end="")
```

```python
agent.reset("web")
print(agent.history("web"))
```

## How to extend it
Wrap or subclass `ChatAgent` for tools/retrieval. Example: override `_request()` to prepend retrieved context to `user_input`, then call `super()._request(session_id, augmented_input)`.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| History did not update after streaming | Caller stopped before iterator completed. | Always consume `stream()` to the final `done` chunk for committed turns. |
| Retry repeats same prior context after failure | Failures intentionally leave memory untouched. | Fix backend issue and retry the same input. |
| Backend loaded unexpectedly in tests | No backend was injected. | Pass `backend=` and optionally `memory=` to `ChatAgent`. |

## Related files
- [Prompt](./prompt.md)
- [Memory](./memory.md)
- [Backends](../backends/__init__.md)
- [Routes](../server/routes_chat.md)
