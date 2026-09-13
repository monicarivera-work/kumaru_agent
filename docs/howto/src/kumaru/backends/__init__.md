# `src/kumaru/backends/__init__.py`

> Backend registry and lazy loader for selectable generation engines.

**Read this when:** selecting, listing, or adding a backend.

---

## What it does
It lists built-in backend import paths, exposes a runtime registry, and instantiates the backend named by `BackendConfig.name`. Built-ins are imported only when selected.

## Why it exists
Optional dependencies like torch should not be required for echo or HTTP use. Lazy loading prevents one backend's dependency stack from blocking all others.

## Mental model
Built-ins live in `BACKEND_PATHS`; runtime plugins live in `backend_registry`. `load_backend()` checks runtime registrations first, then lazy built-ins.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `BACKEND_PATHS` | `dict[str, str]` | Built-in backend name to `module:ClassName` map. |
| `backend_registry` | `Registry[Backend]` | Runtime backend registrations. |
| `available_backends` | `available_backends() -> list[str]` | Sorted selectable names. |
| `load_backend` | `load_backend(config: BackendConfig) -> Backend` | Instantiate selected backend. |
| `Backend` | `class Backend` | Base backend contract. |
| `GenerationResult` | `class GenerationResult` | Non-streamed backend result. |
| `estimate_tokens` | `estimate_tokens(text: str) -> int` | Rough token count helper. |

## How to use it
```python
from kumaru.backends import available_backends
print(available_backends())
```

```python
from kumaru.backends import load_backend
from kumaru.core.config import BackendConfig
backend = load_backend(BackendConfig(name="echo"))
print(backend.health())
```

```python
from kumaru.backends import backend_registry
from kumaru.backends.echo import EchoBackend
backend_registry.register("my_echo", EchoBackend)
```

## How to extend it
To publish a built-in backend, add a `"name": "module:ClassName"` entry to `BACKEND_PATHS`. For application-local plugins, register a factory with `backend_registry.register("name", Factory)` before calling `load_backend()`.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| `unknown backend` | Name is absent from runtime registry and `BACKEND_PATHS`. | Register it or fix `backend.name`. |
| Import error mentions optional extra | Selected backend dependency is not installed. | Install the backend extra named in the hint. |
| Runtime backend not used | Registration happened after `load_backend()`. | Register before constructing `ChatAgent`. |

## Related files
- [Backend base](./base.md)
- [Echo backend](./echo.md)
- [OpenAI-compatible backend](./openai_compat.md)
- [Config](../core/config.md)
