# `src/kumaru/backends/echo.py`

> Dependency-free deterministic backend that echoes the last user message.

**Read this when:** smoke-testing Kumaru without model weights or isolating backend problems.

---

## What it does
It returns a banner explaining that no real model is active and includes the last user message when present. It streams in small chunks with an optional delay to exercise UI streaming.

## Why it exists
Fresh clones and tests need a backend that always works without network, GPU, or downloaded weights. Echo proves the CLI, agent, API, and UI paths independently of real model behavior.

## Mental model
`EchoBackend` is not intelligent; it is an operational diagnostic. It still respects `max_tokens` approximately so truncation paths can be tested.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `EchoBackend` | `EchoBackend(config: BackendConfig, *, delay_s: float = 0.01)` | Echo backend instance. |
| `EchoBackend.name` | `str` | Always `echo`. |
| `EchoBackend.generate` | `generate(self, request: ChatRequest) -> GenerationResult` | Return full echo/banner result. |
| `EchoBackend.stream` | `stream(self, request: ChatRequest) -> Iterator[StreamChunk]` | Yield 8-character chunks and final usage. |
| `EchoBackend.health` | `health(self) -> dict[str, object]` | Report ready echo backend. |

## How to use it
```python
from kumaru.backends.echo import EchoBackend
from kumaru.core.config import BackendConfig
backend = EchoBackend(BackendConfig(), delay_s=0)
```

```python
from kumaru.core.types import ChatRequest, Message
result = backend.generate(ChatRequest(messages=[Message.user("ping")], max_tokens=20))
print(result.text)
```

```python
for chunk in backend.stream(ChatRequest(messages=[Message.user("ping")])):
    if chunk.delta:
        print(chunk.delta, end="")
```

```bash
kumaru ask --backend echo "ping"
```

## How to extend it
Extend it for tests by subclassing or injecting a different `delay_s`. For example, use `EchoBackend(BackendConfig(), delay_s=0)` in test fixtures to avoid sleeps.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Answer says Kumaru cannot answer | The `echo` backend is selected. | Switch `backend.name` to `openai_compat`, `transformers`, or `llamacpp`. |
| UI streaming seems slow | Default `delay_s` sleeps between chunks. | Use `delay_s=0` in tests. |
| Reply is cut off | `request.max_tokens * 4` truncation applied. | Increase `agent.max_tokens` for diagnostics. |

## Related files
- [Backend base](./base.md)
- [Backend registry](./__init__.md)
- [Test config](../../../configs/test.yaml.md)
