# `src/kumaru/core/logging.py`

> Central logging setup and Kumaru-namespaced logger helper.

**Read this when:** configuring process logs or adding logging to a module.

---

## What it does
It configures the root logger once, optionally with JSON formatting, and returns namespaced loggers. Extra logging fields are preserved in JSON output.

## Why it exists
Library modules must not call `logging.basicConfig`. Central setup prevents duplicate handlers and makes structured logs available without touching every call site later.

## Mental model
Application entry points call `setup_logging()`. All other code calls `get_logger("area")` and inherits operator settings.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `JsonFormatter` | `class JsonFormatter(logging.Formatter)` | Render log records as JSON objects. |
| `JsonFormatter.format` | `format(self, record: logging.LogRecord) -> str` | Format one record. |
| `setup_logging` | `setup_logging(level: str = "INFO", *, json_format: bool = False) -> None` | Configure root logging once; later calls adjust level. |
| `get_logger` | `get_logger(name: str) -> logging.Logger` | Return `kumaru.<name>` logger unless already namespaced. |

## How to use it
```python
from kumaru.core.logging import setup_logging
setup_logging("DEBUG", json_format=True)
```

```python
from kumaru.core.logging import get_logger
log = get_logger("agent.chat")
log.info("chat turn", extra={"session": "default"})
```

```python
from kumaru.core.logging import get_logger
assert get_logger("kumaru.server").name == "kumaru.server"
```

## How to extend it
If adding a new process entry point, call `setup_logging(config.log_level, json_format=config.log_json)` once near startup. Inside modules, add `log = get_logger("module.name")` and use `extra={...}` for fields.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Duplicate log lines | Another entry point configured handlers separately. | Use `setup_logging` and avoid `basicConfig` elsewhere. |
| JSON logs missing custom field | Field name collides with reserved `LogRecord` attributes. | Choose a non-reserved key in `extra`. |
| Later `json_format=True` has no effect | Formatter is chosen only on first setup. | Configure JSON on the first call at process start. |

## Related files
- [CLI](../cli.md)
- [Server app](../server/app.md)
