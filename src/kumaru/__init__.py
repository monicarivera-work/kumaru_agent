"""Kumaru — a local-first, scalable LLM agent.

Public surface is intentionally tiny. Everything else is reachable through the
registry so that new components can be added without editing this file.
"""

from kumaru.core.types import (
    ChatResult,
    GenerationParams,
    Message,
    Role,
    StreamChunk,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "ChatResult",
    "GenerationParams",
    "Message",
    "Role",
    "StreamChunk",
    "Usage",
    "__version__",
]
