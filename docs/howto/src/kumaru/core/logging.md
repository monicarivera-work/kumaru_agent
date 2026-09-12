# `src\kumaru\core\logging.py`

> Shared logging setup for colourised local logs or JSON service logs.

**Read this when:** logs are duplicated, you need JSON logs in production, you add structured log fields, or you want a Kumaru namespaced logger.

---

## What it does
This file configures logging for the `kumaru` logger namespace. It provides a console formatter for human-readable local output and a JSON formatter for log aggregation. `get_logger()` configures logging on first use, so modules can safely request a logger without explicit startup wiring.

## Why it exists
Kumaru runs both as a local CLI and as a service. Without a single logging setup, local output would be noisy, production logs would be hard to parse, and uvicorn/root logger propagation could print duplicate lines.

## Mental model
Kumaru owns the `kumaru` logger tree. `setup_logging()` clears and replaces handlers for that tree, disables propagation, and quiets noisy third-party libraries. Use `extra={...}` on log calls when fields should become queryable JSON keys.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `ConsoleFormatter` | `ConsoleFormatter(*, color: bool = True)` | Compact single-line formatter for local logs. |
| `ConsoleFormatter.format` | `format(self, record: logging.LogRecord) -> str` | Render time, level, shortened logger name, message, and exception. |
| `JsonFormatter` | `JsonFormatter()` | One-JSON-object-per-line formatter for services. |
| `JsonFormatter.format` | `format(self, record: logging.LogRecord) -> str` | Render log records and non-standard record fields as JSON. |
| `setup_logging` | `setup_logging(level: str = "INFO", *, fmt: str | None = None) -> None` | Configure the `kumaru` logger namespace. |
| `get_logger` | `get_logger(name: str) -> logging.Logger` | Return a Kumaru namespaced logger, configuring logging if needed. |

Module constants such as `_COLORS` and `_CONFIGURED` are private implementation details.

## How to use it

Get a module logger:

```python
from kumaru.core.logging import get_logger

logger = get_logger(__name__)
logger.info("backend ready")
```

Configure JSON logs explicitly:

```python
from kumaru.core.logging import setup_logging

setup_logging("INFO", fmt="json")
```

Configure with environment variables:

```shell
set KUMARU_LOG_LEVEL=DEBUG
set KUMARU_LOG_FORMAT=json
kumaru serve
```

Attach structured fields:

```python
from kumaru.core.logging import get_logger

logger = get_logger("kumaru.server")
logger.info("request completed", extra={"tokens": 128, "latency_ms": 42.5})
```

With JSON formatting, `tokens` and `latency_ms` become top-level JSON fields.

## How to extend it

Add formatter behaviour only when both CLI and service use cases remain clear. For example, to include milliseconds in JSON timestamps, change `JsonFormatter.format()`:

```python
payload: dict[str, Any] = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
    "ts_ms": int(record.created * 1000),
    "level": record.levelname,
    "logger": record.name,
    "msg": record.getMessage(),
}
```

When adding new structured fields, prefer the standard logging `extra` argument at call sites:

```python
logger.info("streamed", extra={"chunks": emitted, "ttft_ms": ttft_ms})
```

Do not create per-module handlers. Use `get_logger()` and let the central setup control formatting and propagation.

## Failure modes
| Symptom | Likely cause | Fix |
|---|---|---|
| Each log line appears twice | Records are propagating to the root logger or another handler was added. | Use `setup_logging()`; it sets `root.propagate = False` for `kumaru`. |
| JSON logs are not emitted | `fmt` is not `"json"` and `KUMARU_LOG_FORMAT` is unset or different. | Call `setup_logging(..., fmt="json")` or set `KUMARU_LOG_FORMAT=json`. |
| Debug logs do not appear | Logger level defaults to `INFO`. | Set `KUMARU_LOG_LEVEL=DEBUG` or call `setup_logging("DEBUG")`. |
| Logs from `httpx` or `urllib3` are too noisy | Third-party logger level was changed elsewhere. | Reset logging with `setup_logging()` or set those loggers back to `WARNING`. |
| Structured fields missing from console logs | `ConsoleFormatter` only prints the message line and exception. | Use JSON format when structured fields must be queryable. |

## Related files
- `src\kumaru\core\registry.py` — uses `get_logger()` for plugin discovery messages.
- `src\kumaru\backends\base.py` — logs backend load completion.
- `src\kumaru\agent\chat.py` — logs generation and streaming timing.
- `src\kumaru\core\config.py` — `TelemetryConfig` carries `log_level` and `log_format`.
- `src\kumaru\cli.py` — exposes `--log-level` for `serve`.
