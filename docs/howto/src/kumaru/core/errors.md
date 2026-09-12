# `src\kumaru\core\errors.py`

> Exception hierarchy for deliberate Kumaru failures with HTTP status and operator hints.

**Read this when:** startup fails with a config or registry error, you add a new domain failure, or you need the CLI and server to handle an error by type.

---

## What it does
This file defines one base exception, `KumaruError`, and specific subclasses for configuration, registry, backend, dependency, context length, and tool failures. Each deliberate error can carry a human-readable message, optional hint, and HTTP status code. The CLI uses these fields to print actionable output.

## Why it exists
Kumaru needs errors that can be mapped to CLI messages and HTTP responses without string matching. Without this hierarchy, callers would have to inspect exception text, leading to brittle handling and incorrect status codes.

## Mental model
Raise a specific `KumaruError` subclass for failures users or operators can act on. Use `message` for what happened and `hint` for what to do next. Keep unexpected programmer bugs as normal exceptions so they are not mistaken for handled domain failures.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `KumaruError` | `KumaruError(message: str, *, hint: str | None = None)` | Base class for deliberate Kumaru errors; default `http_status = 500`. |
| `KumaruError.http_status` | `int` | Status code server layers can use for HTTP responses. |
| `KumaruError.hint` | `str | None` | Optional operator guidance. |
| `KumaruError.message` | `str` | Error summary without the hint. |
| `ConfigError` | `ConfigError(message: str, *, hint: str | None = None)` | Missing, malformed, or inconsistent configuration; `http_status = 400`. |
| `RegistryError` | `RegistryError(message: str, *, hint: str | None = None)` | Unknown or duplicate registered component; `http_status = 400`. |
| `BackendError` | `BackendError(message: str, *, hint: str | None = None)` | Backend load or generation failure; `http_status = 503`. |
| `DependencyMissingError` | `DependencyMissingError(package: str, *, extra: str | None = None)` | Optional dependency missing; builds an install hint; `http_status = 501`. |
| `ContextLengthExceededError` | `ContextLengthExceededError(message: str, *, hint: str | None = None)` | Prompt does not fit the model context window; `http_status = 413`. |
| `ToolError` | `ToolError(message: str, *, hint: str | None = None)` | Registered tool failed while executing; `http_status = 500`. |

## How to use it

Raise a configuration error with a hint:

```python
from kumaru.core.errors import ConfigError

raise ConfigError(
    "Config file not found: local-gpu",
    hint="Run kumaru info to list installed profiles.",
)
```

Catch all deliberate Kumaru errors:

```python
from kumaru.core.errors import KumaruError

try:
    run_operation()
except KumaruError as exc:
    print(f"error: {exc.message}")
    if exc.hint:
        print(f"hint: {exc.hint}")
```

Report a missing optional package:

```python
from kumaru.core.errors import DependencyMissingError

raise DependencyMissingError("llama-cpp-python", extra="llamacpp")
```

Map to an HTTP response:

```python
from kumaru.core.errors import KumaruError

def error_payload(exc: KumaruError) -> tuple[int, dict[str, str | None]]:
    return exc.http_status, {"error": exc.message, "hint": exc.hint}
```

## How to extend it

Add a subclass when callers need to branch on a distinct failure type or return a distinct status code.

For example, if model downloads become a first-class operator concern:

```python
class ModelDownloadError(BackendError):
    """Model weights could not be downloaded or found in cache."""

    http_status = 503
```

Use it where the failure is detected:

```python
raise ModelDownloadError(
    "Could not download model weights.",
    hint="Check network access and the model id in backend.model.",
)
```

Do not create subclasses for cosmetic wording differences. If the handling and status code are the same, use the existing class with a clearer message and hint.

## Failure modes
| Symptom | Likely cause | Fix |
|---|---|---|
| CLI prints a Python traceback for an operator error | Code raised a generic exception instead of `KumaruError`. | Raise the appropriate subclass with a message and hint. |
| Server returns `500` for bad user config | The wrong error type was raised. | Use `ConfigError` for invalid configuration or `RegistryError` for unknown components. |
| User sees no install command for an optional backend | Code raised `BackendError` instead of `DependencyMissingError`. | Raise `DependencyMissingError(package, extra="...")`. |
| Error handling branches on message text | A distinct error type is missing. | Add or use a specific subclass and branch on type. |
| `str(exc)` duplicates or hides hints | Caller formats `str(exc)` when it needs separate fields. | Use `exc.message` and `exc.hint` for CLI/API output. |

## Related files
- `src\kumaru\cli.py` — catches `KumaruError` in `main()` and prints `message` plus `hint`.
- `src\kumaru\core\config.py` — raises `ConfigError` for missing files and invalid merged config.
- `src\kumaru\core\registry.py` — raises `RegistryError` for duplicate or unknown components.
- `src\kumaru\backends\base.py` — backend implementations should use `BackendError` and `ContextLengthExceededError` when appropriate.
- `src\kumaru\agent\chat.py` — lets deliberate core/backend errors propagate to callers.
