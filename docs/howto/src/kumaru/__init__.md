# `src/kumaru/__init__.py`

> Package entry point that re-exports Kumaru's stable public types and helpers.

**Read this when:** importing Kumaru from another Python program or checking the installed version.

---

## What it does
This file exposes configuration loading, core chat value types, deliberate exceptions, and `__version__`. It avoids forcing callers to know the internal layer layout for common integrations.

## Why it exists
It gives integrations a stable facade. Without it, every caller would import deep paths and small internal refactors would become breaking changes.

## Mental model
Use `kumaru` for stable library imports; use deeper modules only for layer-specific operations. `__all__` is the top-level compatibility boundary.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `__version__` | `str` | Current package version. |
| `KumaruConfig` | `class KumaruConfig` | Root config dataclass. |
| `load_config` | `load_config(path: str | Path | None = None, *, env: dict[str, str] | None = None) -> KumaruConfig` | Load defaults, YAML, and env overrides. |
| `ChatRequest` | `class ChatRequest` | Backend request payload. |
| `Message` | `class Message` | Immutable chat message. |
| `Role` | `class Role(str, Enum)` | Message role values. |
| `Usage` | `class Usage` | Token accounting. |
| `KumaruError` | `class KumaruError(Exception)` | Base deliberate failure. |
| `ConfigError` | `class ConfigError(KumaruError)` | Configuration failure. |
| `BackendError` | `class BackendError(KumaruError)` | Generation/backend failure. |
| `ModelNotAvailableError` | `class ModelNotAvailableError(BackendError)` | Missing model or dependency. |

## How to use it
```python
from kumaru import __version__
print(__version__)
```

```python
from kumaru import load_config
config = load_config("configs/default.yaml")
print(config.backend.name)
```

```python
from kumaru import ChatRequest, Message
req = ChatRequest(messages=[Message.user("hello")], max_tokens=32)
print(req.as_dicts())
```

## How to extend it
Only re-export names that are meant for external callers. For example, to make streaming chunks top-level, import `StreamChunk` from `kumaru.core.types` and add it to `__all__`.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError` for a backend-specific class | The facade intentionally omits backend implementations. | Import backend classes from `kumaru.backends.*` or use `load_backend`. |
| Version differs from package metadata | `__version__` was not updated with release metadata. | Update release process to keep both in sync. |

## Related files
- [CLI](./cli.md)
- [Core package](./core/__init__.md)
- [Configuration](./core/config.md)
