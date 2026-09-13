# `src/kumaru/backends/base.py`

> Abstract backend contract for engines that turn chat requests into text.

**Read this when:** implementing a new generation backend or calling a backend directly.

---

## What it does
It defines `GenerationResult`, the abstract `Backend.generate()` method, default non-incremental streaming, health reporting, context-manager cleanup, and rough token estimation.

## Why it exists
The agent should not know whether text comes from a GPU model, GGUF file, or HTTP server. A small backend contract isolates orchestration from engine details.

## Mental model
Implement `generate()` first; inherit `stream()` until the engine supports real streaming. `close()` must be safe to call more than once.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `GenerationResult` | `GenerationResult(text: str, usage: Usage = Usage())` | Complete non-streamed reply. |
| `Backend` | `Backend(config: BackendConfig)` | Base class for generation engines. |
| `Backend.name` | `str` | Registry/logging/health backend name. |
| `Backend.generate` | `generate(self, request: ChatRequest) -> GenerationResult` | Required full generation method. |
| `Backend.stream` | `stream(self, request: ChatRequest) -> Iterator[StreamChunk]` | Optional incremental generation. |
| `Backend.health` | `health(self) -> dict[str, object]` | Liveness and identity details. |
| `Backend.close` | `close(self) -> None` | Release resources idempotently. |
| `estimate_tokens` | `estimate_tokens(text: str) -> int` | Approximate text tokens as chars/4. |

## How to use it
```python
from kumaru.backends.base import estimate_tokens
print(estimate_tokens("hello world"))
```

```python
from kumaru.backends.echo import EchoBackend
from kumaru.core.config import BackendConfig
from kumaru.core.types import ChatRequest, Message
backend = EchoBackend(BackendConfig())
result = backend.generate(ChatRequest(messages=[Message.user("hi")]))
print(result.text)
```

```python
from kumaru.backends.echo import EchoBackend
from kumaru.core.config import BackendConfig
from kumaru.core.types import ChatRequest, Message
with EchoBackend(BackendConfig(), delay_s=0) as backend:
    for chunk in backend.stream(ChatRequest(messages=[Message.user("hi")])):
        print(chunk.delta, chunk.done)
```

## How to extend it
Create a subclass with `name` and `generate()`. Example: `class ReverseBackend(Backend): name = "reverse"; def generate(self, request): return GenerationResult(request.messages[-1].content[::-1])`.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Agent cannot instantiate subclass | `generate()` was not implemented. | Implement the abstract method. |
| Streaming appears as one chunk | Subclass inherits default `stream()`. | Override `stream()` for true incremental output. |
| Token counts look approximate | Backend used `estimate_tokens`. | Use engine-reported token counts when available. |

## Related files
- [Backend registry](./__init__.md)
- [Echo backend](./echo.md)
- [Core types](../core/types.md)
