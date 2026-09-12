# `src\kumaru\backends\__init__.py`

> Re-exports the lightweight backend primitives that backend authors can import from `kumaru.backends`.

**Read this when:** you need the supported backend import surface, want to avoid importing heavyweight model engines, or are checking why concrete backends are resolved through the registry instead of this package.

---

## What it does

This package initializer imports `Backend` and `iterate_in_thread` from `kumaru.backends.base`. It exposes only those names through `__all__`. It does not import concrete backends such as transformers or llama.cpp, so basic CLI commands do not pay for optional ML imports.

## Why it exists

Kumaru needs a stable import path for backend implementers without forcing torch, llama.cpp, or model weights into every process. Without this small seam, commands such as `kumaru config`, `kumaru info`, and `kumaru serve --backend echo` could become slow or fragile because package import would trigger heavyweight backend modules.

## Mental model

1. `kumaru.backends` is a convenience import for backend primitives.
2. `kumaru.core.registry.BACKENDS` is where concrete backend names are resolved.
3. Keep this file lightweight; register expensive concrete engines lazily.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `Backend` | `class Backend(config: BackendConfig)` | Abstract inference backend contract, re-exported from `kumaru.backends.base`. |
| `iterate_in_thread` | `async def iterate_in_thread(make_iterator: Any, *, queue_size: int = 256) -> AsyncIterator[Any]` | Runs a blocking iterator on a worker thread and yields items to async callers. |
| `__all__` | `['Backend', 'iterate_in_thread']` | Declares the supported wildcard import surface. |

## How to use it

### Import the base class from the public package

```python
from kumaru.backends import Backend
from kumaru.core.types import BackendInfo, GenerationParams, Message, StreamChunk

class MyBackend(Backend):
    name = "my_backend"

    async def _load(self) -> None:
        return None

    def info(self) -> BackendInfo:
        return BackendInfo(name=self.name, model_id=self.config.model)

    async def stream(self, messages: list[Message], params: GenerationParams):
        yield StreamChunk(delta="hello")
        yield StreamChunk(finish_reason="stop")
```

### Use the thread bridge without importing `base.py` directly

```python
from kumaru.backends import iterate_in_thread

async def stream_sync_engine(engine, prompt: str):
    def produce():
        yield from engine.generate_tokens(prompt)

    async for item in iterate_in_thread(produce):
        yield item
```

### Resolve a concrete backend by name

```python
from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS

cfg = load_config("test")
backend = BACKENDS.create(cfg.backend.name, cfg.backend)
await backend.load()
```

### Check backend names from the CLI

```powershell
kumaru info
kumaru config -c default
kumaru chat --backend echo -q "smoke test"
```

## How to extend it

Do not add heavyweight concrete backend imports here. Add a new backend in its own module and register it:

```python
from kumaru.backends import Backend
from kumaru.core.registry import BACKENDS

@BACKENDS.register("toy")
class ToyBackend(Backend):
    ...
```

If importing the backend pulls in torch, CUDA, llama.cpp, or another expensive optional dependency, register it lazily in `src\kumaru\core\registry.py`:

```python
BACKENDS.register_lazy("toy", "kumaru.backends.toy_backend:ToyBackend")
```

Only add symbols to `__all__` when they are lightweight, stable, and useful to backend authors.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'TransformersBackend' from 'kumaru.backends'` | Concrete backends are not re-exported here. | Use `BACKENDS.create("transformers", cfg.backend)` or import the concrete module explicitly. |
| CLI startup becomes slow | A heavyweight import was added to this file. | Remove it and use `BACKENDS.register_lazy(...)`. |
| Custom backend is not listed by `kumaru info` | The class was written but never registered. | Add `@BACKENDS.register(...)`, `register_lazy(...)`, or a `kumaru.backends` package entry point. |
| Wildcard import omits a helper | The helper is not in `__all__`. | Add only lightweight public helpers to `__all__`. |

## Related files

- `src\kumaru\backends\base.py` — defines the exported primitives.
- `src\kumaru\core\registry.py` — owns concrete backend registration and lazy imports.
- `src\kumaru\cli.py` — instantiates selected backends for `chat` and `serve`.
- `configs\default.yaml` — documents backend names operators choose in configuration.
