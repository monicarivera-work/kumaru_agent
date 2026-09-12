# `src\kumaru\backends\echo.py`

> Implements a deterministic, dependency-free backend for tests, CI, and install checks.

**Read this when:** you need to verify Kumaru without model weights, isolate app-layer failures from model failures, or write tests for streaming behavior.

---

## What it does

`EchoBackend` returns a structured response based on the last user message. It never loads a model, never uses a GPU, and streams small chunks with an optional artificial delay. The response includes a stable SHA-256 prefix, message count, approximate prompt tokens, `max_tokens`, and `temperature`.

## Why it exists

Most bugs above inference are in HTTP routing, SSE framing, memory trimming, terminal output, UI rendering, and error handling. Those layers do not need a 15 GB model or CUDA device. Echo gives operators a fast bisection: if `kumaru chat --backend echo` works, the problem is probably in the model backend or its dependencies.

## Mental model

1. Echo is fake inference but a real `Backend` implementation.
2. The output is deterministic for a given last user message because it uses a stable hash.
3. `backend.extra.delay_s` simulates inter-token delay; set it to `0` for instant tests.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `EchoBackend` | `class EchoBackend(config: BackendConfig)` | Dependency-free backend registered as `echo`. |
| `EchoBackend.name` | `str = "echo"` | Registry and introspection name. |
| `EchoBackend._load` | `async def _load(self) -> None` | No-op load hook. |
| `EchoBackend.info` | `def info(self) -> BackendInfo` | Reports `model_id="echo"`, `device="cpu"`, `dtype="none"`, context length, and streaming support. |
| `EchoBackend.stream` | `async def stream(self, messages: Sequence[Message], params: GenerationParams) -> AsyncIterator[StreamChunk]` | Streams deterministic chunks and a final usage chunk. |

## How to use it

### Run one prompt with no model dependencies

```powershell
kumaru chat --backend echo -q "hello"
kumaru chat -c test -q "hello"
```

### Start the HTTP server with the test profile

```powershell
kumaru serve -c test
```

`configs\test.yaml` uses:

```yaml
backend:
  name: echo
  model: echo
  max_context_length: 8192
  extra:
    delay_s: 0.01

generation:
  max_tokens: 256
  temperature: 0.0
```

### Instantiate directly in a test

```python
from kumaru.backends.echo import EchoBackend
from kumaru.core.config import BackendConfig
from kumaru.core.types import GenerationParams, Message

backend = EchoBackend(BackendConfig(name="echo", model="echo", extra={"delay_s": 0}))
await backend.load()
parts = []
async for chunk in backend.stream([Message.user("ping")], GenerationParams(max_tokens=32)):
    if chunk.delta:
        parts.append(chunk.delta)
print("".join(parts))
```

### Bisect an operator issue

```powershell
kumaru chat --backend echo -q "smoke test"
kumaru chat -c local-gpu -q "smoke test"
```

If echo works and the real profile fails, inspect optional dependencies, CUDA, VRAM, model ids, adapter paths, or upstream servers.

## How to extend it

Use echo as the template for other test-only backends: keep construction cheap, no-op `_load()` if there are no resources, return accurate `BackendInfo`, stream `StreamChunk(delta=...)`, and emit a final chunk with `finish_reason` and `Usage`.

```python
from kumaru.backends.base import Backend, chunks_from_text
from kumaru.core.registry import BACKENDS
from kumaru.core.types import BackendInfo, StreamChunk, Usage

@BACKENDS.register("fixed")
class FixedBackend(Backend):
    name = "fixed"

    async def _load(self) -> None:
        return None

    def info(self) -> BackendInfo:
        return BackendInfo(name=self.name, model_id="fixed", device="cpu", dtype="none")

    async def stream(self, messages, params):
        for piece in chunks_from_text("fixed response", size=6):
            yield StreamChunk(delta=piece)
        yield StreamChunk(finish_reason="stop", usage=Usage(prompt_tokens=0, completion_tokens=2))
```

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Tests are slower than expected | `backend.extra.delay_s` defaults to `0.01`. | Set `extra.delay_s: 0`. |
| Reply says `You said: (nothing)` | No user message, or empty user content. | Send `Message.user(...)` or CLI `-q "..."`. |
| Token counts are approximate | Echo inherits the base heuristic counter. | Use a tokenizer-backed backend for exact counts. |
| `Unknown backend: 'echo'` | Registry lazy registration is missing. | Restore `BACKENDS.register_lazy("echo", "kumaru.backends.echo:EchoBackend")`. |
| Echo works but transformers fails | Echo has no optional ML dependencies. | Install `kumaru[torch]`, check CUDA/device/dtype, or use a smaller profile. |

## Related files

- `src\kumaru\backends\base.py` — provides `Backend`, `chunks_from_text`, and default token counting.
- `src\kumaru\core\registry.py` — lazily registers `echo`.
- `src\kumaru\core\types.py` — defines message, params, chunk, usage, and info models.
- `configs\test.yaml` — ready-to-run echo profile.
- `src\kumaru\cli.py` — `chat` and `serve` instantiate this backend through the registry.
