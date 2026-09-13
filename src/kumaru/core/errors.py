"""The exception hierarchy for Kumaru.

Why a hierarchy at all? Because the HTTP layer must translate failures into
status codes without inspecting strings. Every error raised on purpose inherits
from :class:`KumaruError`, so ``routes_chat`` can catch exactly that and return
a clean 4xx/5xx, while a genuine bug (``TypeError``, ``KeyError``) still
escapes loudly as a 500 with a traceback in the logs.

Each error carries an ``http_status`` so the mapping lives with the error, not
scattered through the routing code.
"""

from __future__ import annotations


class KumaruError(Exception):
    """Base class for every deliberate Kumaru failure."""

    http_status: int = 500

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, str]:
        """JSON-safe form, used by the API error handler."""
        payload = {"error": type(self).__name__, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        return payload


class ConfigError(KumaruError):
    """Configuration is missing, malformed, or internally inconsistent.

    Raised at startup, never mid-request: an invalid config should stop the
    process before it accepts traffic.
    """

    http_status = 500


class BackendError(KumaruError):
    """A backend failed while generating (network, OOM, bad response)."""

    http_status = 502


class ModelNotAvailableError(BackendError):
    """The requested model or its optional dependency is not installed.

    Separate from :class:`BackendError` because it is almost always fixed by an
    action on the operator's machine (``pip install``, download weights),
    so the UI can show an actionable hint instead of "backend failed".
    """

    http_status = 503


class RegistryError(KumaruError):
    """A plugin name was requested that is not registered, or double-registered."""

    http_status = 500
