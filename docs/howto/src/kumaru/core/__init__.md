# `src/kumaru/core/__init__.py`

> Core-layer facade for configuration, errors, logging, registry, and value types.

**Read this when:** working inside Kumaru internals and needing dependency-safe imports.

---

## What it does
It re-exports the vocabulary that all upper layers can depend on. The module keeps `core` independent from backends, agent orchestration, and HTTP serving.

## Why it exists
The one-way dependency rule prevents cycles and keeps upper layers independently testable. Without it, importing a type could accidentally load web or model dependencies.

## Mental model
`kumaru.core` is the bottom layer. Anything here may be imported upward, but it must not import from `kumaru.backends`, `kumaru.agent`, or `kumaru.server`.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `AgentConfig` | `class AgentConfig` | Agent prompt/sampling/memory settings. |
| `BackendConfig` | `class BackendConfig` | Backend engine settings. |
| `ServerConfig` | `class ServerConfig` | HTTP server settings. |
| `KumaruConfig` | `class KumaruConfig` | Full config tree. |
| `load_config` | `load_config(path: str | Path | None = None, *, env: dict[str, str] | None = None) -> KumaruConfig` | Load validated config. |
| `KumaruError` | `class KumaruError(Exception)` | Base deliberate error. |
| `Registry` | `class Registry(Generic[T])` | Name-to-factory registry. |
| `Message` | `class Message` | Conversation value type. |
| `Role` | `class Role(str, Enum)` | Message role enum. |
| `StreamChunk` | `class StreamChunk` | Streaming reply piece. |
| `Usage` | `class Usage` | Token usage. |
| `get_logger` | `get_logger(name: str) -> logging.Logger` | Namespaced logger. |
| `setup_logging` | `setup_logging(level: str = "INFO", *, json_format: bool = False) -> None` | Configure root logging. |

## How to use it
```python
from kumaru.core import Message, Role
msg = Message(Role.USER, "hello")
```

```python
from kumaru.core import load_config
config = load_config()
```

```python
from kumaru.core import Registry
registry = Registry[int]("number")
registry.register("one", lambda: 1)
print(registry.create("one"))
```

## How to extend it
Add new foundational types here only if every upper layer may depend on them. Keep imports acyclic: a new `core` module must not import backend, agent, or server symbols.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Import cycle | A core module imported an upper layer. | Move shared code down into core or inject it from the upper layer. |
| Optional dependency imports during startup | A core re-export imports a backend module. | Re-export only core-owned symbols here. |

## Related files
- [Types](./types.md)
- [Config](./config.md)
- [Errors](./errors.md)
- [Registry](./registry.md)
- [Logging](./logging.md)
