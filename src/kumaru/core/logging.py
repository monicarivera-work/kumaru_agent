"""Logging setup - one function, called once, at process start.

Rationale
---------
Library code must never call ``logging.basicConfig``; only the application
entry point (``kumaru.cli``) does, via :func:`setup_logging`. Everything else
calls :func:`get_logger` and inherits whatever the operator configured.

JSON output is available because the moment you run Kumaru on more than one
machine you want to grep structured logs, and retrofitting that later means
touching every call site.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_CONFIGURED = False

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Anything passed via logger.info(..., extra={...}) is preserved.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", *, json_format: bool = False) -> None:
    """Configure the root logger. Safe to call more than once (later calls
    only adjust the level, so tests and reloads do not stack handlers)."""
    global _CONFIGURED
    root = logging.getLogger()
    resolved = getattr(logging, str(level).upper(), logging.INFO)

    if _CONFIGURED:
        root.setLevel(resolved)
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root.handlers[:] = [handler]
    root.setLevel(resolved)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (``kumaru.<name>``)."""
    if not name.startswith("kumaru"):
        name = f"kumaru.{name}"
    return logging.getLogger(name)
