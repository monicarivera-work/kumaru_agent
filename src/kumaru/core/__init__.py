"""Core layer: the vocabulary every other layer depends on.

Nothing in ``kumaru.core`` may import from ``kumaru.backends``,
``kumaru.agent`` or ``kumaru.server``. That one-way rule is what keeps the
dependency graph acyclic and makes each upper layer independently testable.
"""

from kumaru.core.config import (
    AgentConfig,
    BackendConfig,
    KumaruConfig,
    ServerConfig,
    load_config,
)
from kumaru.core.errors import (
    BackendError,
    ConfigError,
    KumaruError,
    ModelNotAvailableError,
)
from kumaru.core.logging import get_logger, setup_logging
from kumaru.core.registry import Registry
from kumaru.core.types import ChatRequest, Message, Role, StreamChunk, Usage

__all__ = [
    "AgentConfig",
    "BackendConfig",
    "BackendError",
    "ChatRequest",
    "ConfigError",
    "KumaruConfig",
    "KumaruError",
    "Message",
    "ModelNotAvailableError",
    "Registry",
    "Role",
    "ServerConfig",
    "StreamChunk",
    "Usage",
    "get_logger",
    "load_config",
    "setup_logging",
]
