"""
kumaru/logger.py
----------------
Structured logging for the Kumaru agent.

Why structured logging?
  In production you need to be able to search and filter logs.  Emitting
  JSON-lines (one JSON object per line) makes that trivial with any log
  aggregation tool (Datadog, CloudWatch, Splunk, …).

  During local development we use a human-readable format instead so that
  reading the agent's reasoning is not painful.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Each record contains:
      timestamp – ISO-8601 UTC
      level     – DEBUG / INFO / WARNING / ERROR
      logger    – the logger name (usually the module)
      message   – the human-readable message
      **extra   – any additional key=value context passed to the logger
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Copy any extra fields the caller attached via the `extra` kwarg.
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            } and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload)


def get_logger(name: str, *, verbose: bool = False) -> logging.Logger:
    """Return a logger configured for the Kumaru agent.

    Args:
        name:    Usually ``__name__`` of the calling module.
        verbose: When *True* the logger emits DEBUG-level messages in a
                 human-friendly format (good for local development).
                 When *False* it emits INFO+ as structured JSON.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured (e.g. called twice in the same process).
        return logger

    handler = logging.StreamHandler(sys.stdout)

    if verbose:
        fmt = "%(asctime)s [%(levelname)s] %(name)s – %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.setLevel(logging.DEBUG)
    else:
        handler.setFormatter(StructuredFormatter())
        logger.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.propagate = False
    return logger
