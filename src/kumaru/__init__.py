"""Kumaru - a local-first, scalable LLM chat agent.

This package is deliberately layered so each layer can be swapped or scaled
independently:

    core/      value types, config, errors, logging, plugin registry
    backends/  anything that can turn a prompt into tokens
    agent/     conversation orchestration (prompting + memory)
    server/    HTTP surface (FastAPI) and the browser UI it serves

The only public entry points most callers need are re-exported here.
"""

from kumaru.core.config import KumaruConfig, load_config
from kumaru.core.errors import (
    BackendError,
    ConfigError,
    KumaruError,
    ModelNotAvailableError,
)
from kumaru.core.types import ChatRequest, Message, Role, Usage

__version__ = "0.1.0"

__all__ = [
    "BackendError",
    "ChatRequest",
    "ConfigError",
    "KumaruConfig",
    "KumaruError",
    "Message",
    "ModelNotAvailableError",
    "Role",
    "Usage",
    "__version__",
    "load_config",
]
