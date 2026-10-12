"""Structured logging setup.

Two modes, one switch:

* **console** (default) — colorized, human-readable, for local development.
* **json** — one JSON object per line, for when you run Kumaru as a service and
  something else (Loki, CloudWatch, Datadog) has to parse it.

The reason this matters for scalability: the moment you run more than one
replica, grepping human-readable logs stops working. Having the JSON switch from
day one means the transition costs one environment variable, not a refactor.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from typing import Any

_CONFIGURED = False

_RESET = "\x1b[0m"
_COLORS = {
    "DEBUG": "\x1b[38;5;244m",
    "INFO": "\x1b[38;5;39m",
    "WARNING": "\x1b[38;5;214m",
    "ERROR": "\x1b[38;5;196m",
    "CRITICAL": "\x1b[48;5;196m\x1b[38;5;231m",
}


def ensure_utf8_streams() -> None:
    """Force stdout/stderr to UTF-8.

    Windows consoles default to a legacy code page (cp1252 or cp437), so any
    non-ASCII character in a log line renders as a replacement glyph. Since the
    output includes em-dashes, arrows, and box-drawing characters, that turns
    readable diagnostics into noise. Reconfiguring the streams fixes it
    globally; `errors="replace"` guarantees we can never crash on an
    unencodable character, which would be a far worse failure than a wrong glyph.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # Detached or redirected streams may refuse reconfiguration; a wrong
        # glyph is a far better outcome than a crash at startup.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


class ConsoleFormatter(logging.Formatter):
    """Compact, aligned, optionally colorized single-line records."""

    def __init__(self, *, color: bool = True) -> None:
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = record.levelname
        prefix = f"{_COLORS.get(level, '')}{level:<8}{_RESET}" if self.color else f"{level:<8}"
        name = record.name.replace("kumaru.", "")
        msg = record.getMessage()
        line = f"{ts} {prefix} {name:<26} {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(logging.Formatter):
    """One JSON object per line — machine-parseable, aggregation-friendly."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Anything attached via `logger.info(..., extra={"tokens": 42})` is
        # promoted to a top-level field so it is queryable downstream.
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord("", 0, "", 0, "", (), None).__dict__ and key != "extra":
                payload.setdefault(key, value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", *, fmt: str | None = None) -> None:
    """Configure the root logger once. Safe to call repeatedly.

    Parameters
    ----------
    level: ``DEBUG`` | ``INFO`` | ``WARNING`` | ``ERROR``
    fmt:   ``console`` | ``json``. Defaults to ``$KUMARU_LOG_FORMAT`` or console.
    """
    global _CONFIGURED

    ensure_utf8_streams()
    fmt = fmt or os.environ.get("KUMARU_LOG_FORMAT", "console")
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(ConsoleFormatter(color=sys.stderr.isatty()))

    root = logging.getLogger("kumaru")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Do not let records bubble to the root logger, which uvicorn also
    # configures — otherwise every line prints twice.
    root.propagate = False

    # These are chatty and rarely useful at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use."""
    if not _CONFIGURED:
        setup_logging(os.environ.get("KUMARU_LOG_LEVEL", "INFO"))
    return logging.getLogger(name if name.startswith("kumaru") else f"kumaru.{name}")
