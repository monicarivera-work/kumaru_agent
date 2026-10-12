"""Kumaru's exception hierarchy.

Why a hierarchy instead of raising `ValueError` everywhere: the server must map
failures onto HTTP status codes, and the CLI must print an actionable message. A
flat exception type forces string-matching on error text, which breaks silently.
One base class plus specific subclasses lets both callers branch on *type* and
stay correct forever.
"""

from __future__ import annotations

import os


class KumaruError(Exception):
    """Base class for every error Kumaru raises deliberately.

    `http_status` lets the server translate domain errors into responses without
    the server layer needing to know about every error type.
    """

    http_status: int = 500
    hint: str | None = None

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if hint is not None:
            self.hint = hint

    def __str__(self) -> str:
        return f"{self.message} ({self.hint})" if self.hint else self.message


class ConfigError(KumaruError):
    """Configuration is missing, malformed, or internally inconsistent."""

    http_status = 400


class RegistryError(KumaruError):
    """A component was requested by name but is not registered."""

    http_status = 400


class RequestError(KumaruError):
    """An API request is well-formed JSON but asks for something invalid."""

    http_status = 400


class AuthError(KumaruError):
    """An API request carried a missing or wrong API key.

    401, not 500: HTTP clients (the OpenAI SDK included) retry 5xx responses,
    so a mistyped key would otherwise cost several slow, misleading retries.
    """

    http_status = 401


class BackendError(KumaruError):
    """A backend failed to load, or failed during generation."""

    http_status = 503


_SOURCE_INSTALL = 'pip install -e ".[{extra}]" in the Kumaru folder'
# Kumaru installs from its source folder, not from PyPI, so a hint must never
# say `pip install "kumaru[...]"`: that asks the package index for a project
# named kumaru, which is not this one. PyTorch comes first through setup.ps1
# because the PyPI wheel for Windows is CPU-only.
_SETUP_INSTALL = {
    "torch": r'.\scripts\setup.ps1 -Gpu (a CUDA build of PyTorch, then pip install -e ".[train]")',
    "train": r'.\scripts\setup.ps1 -Gpu (a CUDA build of PyTorch, then pip install -e ".[train]")',
    "quant": r".\scripts\setup.ps1 -Quant, or " + _SOURCE_INSTALL.format(extra="quant"),
    "gguf": r".\scripts\setup.ps1 -Gguf, or " + _SOURCE_INSTALL.format(extra="gguf"),
}


def install_command(extra: str, *, windows: bool = os.name == "nt") -> str:
    """How to install one of Kumaru's optional extras (see pyproject.toml)."""
    if not windows:
        # setup.ps1 is Windows-only, and Linux PyPI wheels of PyTorch include CUDA.
        return _SOURCE_INSTALL.format(extra=extra)
    return _SETUP_INSTALL.get(extra, _SOURCE_INSTALL.format(extra=extra))


class DependencyMissingError(BackendError):
    """An optional dependency is required for the selected component.

    Carries the exact install command so the user is never left guessing.
    """

    http_status = 501

    def __init__(self, package: str, *, extra: str | None = None) -> None:
        install = install_command(extra) if extra else f"pip install {package}"
        super().__init__(
            f"Optional dependency '{package}' is not installed.",
            hint=f"Install it with: {install}",
        )
        self.package = package


class ContextLengthExceededError(KumaruError):
    """The prompt does not fit in the model's context window."""

    http_status = 413


class ToolError(KumaruError):
    """A registered tool failed while executing."""

    http_status = 500
