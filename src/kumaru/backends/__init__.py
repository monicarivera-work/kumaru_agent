"""Backend registry and lazy loader.

Backends are registered by *name* with an import path, and the module is only
imported when that name is selected. That is what lets someone run Kumaru
against a llama.cpp server without ever installing torch.
"""

from __future__ import annotations

from importlib import import_module

from kumaru.backends.base import Backend, GenerationResult, estimate_tokens
from kumaru.core.config import BackendConfig
from kumaru.core.errors import ModelNotAvailableError, RegistryError
from kumaru.core.logging import get_logger
from kumaru.core.registry import Registry

log = get_logger("backends")

#: name -> "module:ClassName". Add a line here to publish a new backend.
BACKEND_PATHS: dict[str, str] = {
    "echo": "kumaru.backends.echo:EchoBackend",
    "openai_compat": "kumaru.backends.openai_compat:OpenAICompatBackend",
    "transformers": "kumaru.backends.transformers_backend:TransformersBackend",
    "llamacpp": "kumaru.backends.llamacpp_backend:LlamaCppBackend",
}

#: Exposed for callers that want to register a backend at runtime.
backend_registry: Registry[Backend] = Registry("backend")


def available_backends() -> list[str]:
    """Names that can be selected, including runtime registrations."""
    return sorted(set(BACKEND_PATHS) | set(backend_registry.names()))


def _resolve(name: str) -> type[Backend]:
    path = BACKEND_PATHS[name]
    module_name, class_name = path.split(":")
    try:
        module = import_module(module_name)
    except ImportError as exc:  # optional dependency missing
        raise ModelNotAvailableError(
            f"backend '{name}' could not be imported: {exc}",
            hint=f"install its extra, e.g. pip install -e '.[{name}]'",
        ) from exc
    return getattr(module, class_name)


def load_backend(config: BackendConfig) -> Backend:
    """Instantiate the backend named by ``config.name``."""
    name = config.name.strip().lower()
    if name in backend_registry:
        return backend_registry.create(name, config)
    if name not in BACKEND_PATHS:
        raise RegistryError(
            f"unknown backend '{config.name}'",
            hint=f"available: {', '.join(available_backends())}",
        )
    log.info("loading backend", extra={"backend": name, "model": config.model})
    return _resolve(name)(config)


__all__ = [
    "BACKEND_PATHS",
    "Backend",
    "GenerationResult",
    "available_backends",
    "backend_registry",
    "estimate_tokens",
    "load_backend",
]
