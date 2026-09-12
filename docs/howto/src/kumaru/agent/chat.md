# `src\kumaru\agent\chat.py`

> Turn orchestrator that prepares messages, calls the backend, and records generation behavior.

**Read this when:** you are changing chat behavior, debugging prompt contents, adding request-level generation overrides, or reasoning about stateless scaling.

---

## What it does
`ChatAgent` turns a caller-supplied message list into a model response. It builds the system prompt, injects tool descriptions, trims history to fit the context budget, calls the backend, and logs timing/token metrics. It has both non-streaming and streaming paths.

## Why it exists
Backends should implement inference mechanics, not product policy. Without this layer, every backend would need to duplicate prompt normalization, history trimming, config overrides, and telemetry. The class is deliberately stateless: callers send the full conversation each turn, so any server replica can handle any request without sticky sessions.

## Mental model
- The caller owns conversation persistence; `ChatAgent` stores no history.
- `prepare()` is pure and testable; `complete()` and `stream()` perform I/O.
- The backend context window is split between prompt history and reserved completion tokens.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `ChatAgent` | `class ChatAgent(backend: Backend, config: KumaruConfig)` | Owns prompt, memory, tool, and generation policy for one request. |
| `ChatAgent.prepare` | `prepare(messages: Sequence[Message], *, extra_context: str | None = None) -> list[Message]` | Returns normalized, budgeted messages with exactly one system message at index 0. |
| `ChatAgent.complete` | `async complete(messages: Sequence[Message], *, extra_context: str | None = None, **overrides) -> ChatResult` | Produces a full assistant message by calling `backend.generate`. |
| `ChatAgent.stream` | `async stream(messages: Sequence[Message], *, extra_context: str | None = None, **overrides) -> AsyncIterator[StreamChunk]` | Yields incremental response chunks from `backend.stream`. |
| `ChatAgent.describe` | `describe() -> dict` | Returns effective backend, memory, tools, and generation settings for `/v1/info`. |

## How to use it

### Prepare messages without running a model

```python
from kumaru.agent.chat import ChatAgent
from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS
from kumaru.core.types import Message

config = load_config("test")
backend = BACKENDS.create(config.backend.name, config.backend)
agent = ChatAgent(backend, config)

prepared = agent.prepare([Message.user("What can you do?")])
assert prepared[0].role.value == "system"
print(prepared[-1].content)
```

### Run a non-streaming completion

```python
from kumaru.core.types import Message

result = await agent.complete(
    [Message.user("Give me one operational risk of unlimited concurrency.")],
    temperature=0.2,
    max_tokens=128,
)
print(result.message.content)
print(result.usage.total_tokens, result.tokens_per_second)
```

### Stream a reply

```python
from kumaru.core.types import Message

parts = []
async for chunk in agent.stream([Message.user("Count to five.")], max_tokens=32):
    if chunk.delta:
        parts.append(chunk.delta)
    if chunk.finish_reason:
        print("finished:", chunk.finish_reason)

print("".join(parts))
```

### Add retrieved context for one turn

```python
messages = [Message.user("Summarize the deployment note.")]
extra_context = "Deployment note: bind to 127.0.0.1 unless server.api_key is set."
result = await agent.complete(messages, extra_context=extra_context)
```

## How to extend it
The intended extension point is to keep orchestration here and put replaceable policies in their own modules.

Worked example: expose tool specs to a future backend that supports structured tool calls.

```python
prepared = self.prepare(messages, extra_context=extra_context)
params = self._params(**overrides)

if self.tools:
    tool_specs = self.tools.specs()
    # Pass tool_specs through a backend method or GenerationParams extension.

result = await self.backend.generate(prepared, params)
```

When adding a loop for tools, enforce `config.agent.max_tool_iterations` and use `ToolBelt.invoke()` so tool timeouts and error-to-text conversion remain centralized.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ContextLengthExceededError` | `memory_policy` is `none` and the message list exceeds the computed prompt budget. | Use `sliding_window` or lower `generation.max_tokens` so more room remains for history. |
| Responses truncate immediately | `generation.max_tokens` reserves too much or too little relative to backend `context_length`. | Set realistic `max_tokens` and `agent.max_history_tokens` for the model. |
| UI shows slow first token | Backend `stream()` blocks before yielding the first `StreamChunk`. | Use `iterate_in_thread()` in blocking backends and avoid expensive work on the event loop. |
| Tool names appear in the prompt but no tool runs | v0.1 has only the tool seam; `ChatAgent` does not yet implement a tool-call loop. | Add a bounded loop around backend calls before relying on tool execution. |
| Server replicas lose history | A client expected server-side sessions. | Send the full `messages` list on every request; persist history in the client or external store. |

## Related files

- `src\kumaru\agent\prompt.py` — builds and normalizes the system message.
- `src\kumaru\agent\memory.py` — supplies the selected `MemoryPolicy`.
- `src\kumaru\agent\tools.py` — resolves enabled tool names and descriptions.
- `src\kumaru\backends\base.py` — defines `Backend.generate`, `stream`, `count_tokens`, and `context_length`.
- `src\kumaru\core\config.py` — defines `KumaruConfig`, generation defaults, and agent settings.
- `src\kumaru\server\routes_chat.py` — calls `complete()` and `stream()` for HTTP requests.

