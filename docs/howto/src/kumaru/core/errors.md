# `src/kumaru/core/errors.py`

> Deliberate exception hierarchy with HTTP status metadata.

**Read this when:** raising operator-facing failures or mapping errors to API responses.

---

## What it does
It defines a base `KumaruError` plus specific config, backend, model-availability, and registry errors. Each error can serialize itself to JSON and carries an HTTP status.

## Why it exists
Routes need to distinguish expected operational failures from bugs without parsing strings. Without this hierarchy, the API would either leak tracebacks or flatten useful hints into generic 500s.

## Mental model
Raise `KumaruError` subclasses for failures an operator can understand. Let ordinary Python exceptions escape for programming bugs.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `KumaruError` | `KumaruError(message: str, *, hint: str | None = None)` | Base deliberate Kumaru failure. |
| `KumaruError.http_status` | `int` | Default HTTP status for the error class. |
| `KumaruError.to_dict` | `to_dict(self) -> dict[str, str]` | JSON-safe error payload. |
| `ConfigError` | `class ConfigError(KumaruError)` | Invalid startup configuration. |
| `BackendError` | `class BackendError(KumaruError)` | Generation/network/OOM backend failure. |
| `ModelNotAvailableError` | `class ModelNotAvailableError(BackendError)` | Missing model or optional dependency. |
| `RegistryError` | `class RegistryError(KumaruError)` | Unknown or duplicate registry name. |

## How to use it
```python
from kumaru.core.errors import ConfigError
raise ConfigError("agent.max_tokens must be > 0")
```

```python
from kumaru.core.errors import ModelNotAvailableError
err = ModelNotAvailableError("model missing", hint="download weights")
print(err.http_status, err.to_dict())
```

```python
from kumaru.core.errors import KumaruError
try:
    raise KumaruError("planned failure")
except KumaruError as exc:
    print(exc.message)
```

## How to extend it
Create a subclass when a distinct handler or HTTP status is useful. Example: `class RateLimitError(BackendError): http_status = 429` keeps routing code generic.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| API returns 500 for expected failure | Code raised a plain exception. | Raise a `KumaruError` subclass with a helpful hint. |
| UI lacks remediation text | `hint` was omitted. | Pass `hint=` when the operator can take a concrete action. |
| Config error appears mid-request | Validation was deferred. | Move checks into config loading/startup. |

## Related files
- [Config](./config.md)
- [Registry](./registry.md)
- [Chat routes](../server/routes_chat.md)
