"""Typed configuration, loaded from YAML with environment overrides.

Precedence (lowest to highest)
------------------------------
1. the dataclass defaults below - Kumaru must run with *no* config file
2. the YAML file passed to :func:`load_config`
3. ``KUMARU_*`` environment variables (see :data:`ENV_OVERRIDES`)

Why dataclasses instead of raw dicts: a typo in a dict key fails silently at
2am; a typo here raises :class:`ConfigError` at startup with the offending key
named. Secrets (``api_key``) are read from the environment only, never stored
in a committed YAML file.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from kumaru.core.errors import ConfigError

DEFAULT_SYSTEM_PROMPT = (
    "You are Kumaru, a helpful local assistant. "
    "Answer accurately and concisely. "
    "If you are not sure about something, say so instead of guessing."
)


@dataclass(slots=True)
class BackendConfig:
    """Which engine generates tokens, and how."""

    name: str = "echo"
    model: str = ""
    # HTTP backends (openai_compat: llama.cpp server, Ollama, vLLM, LM Studio).
    base_url: str = "http://localhost:11434/v1"
    api_key_env: str = "KUMARU_API_KEY"
    timeout_s: float = 120.0
    # In-process backends (transformers / llama.cpp).
    device: str = "auto"
    dtype: str = "auto"
    context_length: int = 4096
    n_threads: int = 0  # 0 = let the runtime decide
    gpu_layers: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        """Secret pulled from the environment at call time, never persisted."""
        return os.environ.get(self.api_key_env, "")


@dataclass(slots=True)
class AgentConfig:
    """Conversation behaviour: prompt, sampling, and memory budget."""

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    # Memory is bounded by turns *and* characters so one huge paste cannot
    # silently blow the context window.
    max_history_turns: int = 12
    max_history_chars: int = 12000
    stop: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ServerConfig:
    """Where the HTTP server listens and who may call it."""

    host: str = "127.0.0.1"
    port: int = 8000
    # Same-origin by default: the UI is served by this app, so no CORS needed.
    cors_origins: list[str] = field(default_factory=list)
    serve_ui: bool = True


@dataclass(slots=True)
class KumaruConfig:
    """The whole configuration tree."""

    backend: BackendConfig = field(default_factory=BackendConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    log_level: str = "INFO"
    log_json: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, safe to log or return from ``/api/config``."""
        return asdict(self)

    def validate(self) -> None:
        """Fail fast on values that would only break later, mid-request."""
        if not self.backend.name:
            raise ConfigError("backend.name must not be empty")
        if self.agent.max_tokens <= 0:
            raise ConfigError("agent.max_tokens must be > 0")
        if not 0.0 <= self.agent.temperature <= 2.0:
            raise ConfigError("agent.temperature must be between 0.0 and 2.0")
        if not 0.0 < self.agent.top_p <= 1.0:
            raise ConfigError("agent.top_p must be in (0.0, 1.0]")
        if self.agent.max_history_turns < 1:
            raise ConfigError("agent.max_history_turns must be >= 1")
        if self.agent.max_history_chars < 1:
            raise ConfigError("agent.max_history_chars must be >= 1")
        if not 1 <= self.server.port <= 65535:
            raise ConfigError("server.port must be between 1 and 65535")
        if self.backend.timeout_s <= 0:
            raise ConfigError("backend.timeout_s must be > 0")


#: ``environment variable -> dotted config path``.
ENV_OVERRIDES: dict[str, str] = {
    "KUMARU_BACKEND": "backend.name",
    "KUMARU_MODEL": "backend.model",
    "KUMARU_BASE_URL": "backend.base_url",
    "KUMARU_DEVICE": "backend.device",
    "KUMARU_CONTEXT_LENGTH": "backend.context_length",
    "KUMARU_GPU_LAYERS": "backend.gpu_layers",
    "KUMARU_SYSTEM_PROMPT": "agent.system_prompt",
    "KUMARU_MAX_TOKENS": "agent.max_tokens",
    "KUMARU_TEMPERATURE": "agent.temperature",
    "KUMARU_HOST": "server.host",
    "KUMARU_PORT": "server.port",
    "KUMARU_LOG_LEVEL": "log_level",
}

_SECTIONS = {"backend": BackendConfig, "agent": AgentConfig, "server": ServerConfig}


def _coerce(current: Any, raw: str) -> Any:
    """Convert an env string to the type of the value it replaces."""
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        try:
            return int(raw)
        except ValueError:
            raise ConfigError(f"expected an integer, got {raw!r}") from None
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError:
            raise ConfigError(f"expected a number, got {raw!r}") from None
    if isinstance(current, list):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return raw


def _build_section(name: str, cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"unknown key(s) in '{name}': {', '.join(sorted(unknown))}",
            hint=f"valid keys: {', '.join(sorted(known))}",
        )
    return cls(**data)


def from_dict(data: dict[str, Any]) -> KumaruConfig:
    """Build a config tree from a nested dict (the parsed YAML)."""
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")

    kwargs: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        section = data.get(name, {}) or {}
        if not isinstance(section, dict):
            raise ConfigError(f"config section '{name}' must be a mapping")
        kwargs[name] = _build_section(name, cls, dict(section))

    top_level = {f.name for f in fields(KumaruConfig)} - set(_SECTIONS)
    unknown = set(data) - set(_SECTIONS) - top_level
    if unknown:
        raise ConfigError(f"unknown top-level key(s): {', '.join(sorted(unknown))}")
    for key in top_level:
        if key in data:
            kwargs[key] = data[key]

    return KumaruConfig(**kwargs)


def apply_env_overrides(
    config: KumaruConfig, env: dict[str, str] | None = None
) -> KumaruConfig:
    """Apply ``KUMARU_*`` variables on top of ``config`` (mutates and returns it)."""
    env = dict(os.environ if env is None else env)
    for var, path in ENV_OVERRIDES.items():
        raw = env.get(var)
        if raw is None or raw == "":
            continue
        target: Any = config
        *parents, leaf = path.split(".")
        for part in parents:
            target = getattr(target, part)
        try:
            setattr(target, leaf, _coerce(getattr(target, leaf), raw))
        except ConfigError as exc:
            raise ConfigError(f"{var}: {exc.message}") from None
    return config


def load_config(
    path: str | Path | None = None, *, env: dict[str, str] | None = None
) -> KumaruConfig:
    """Load configuration from ``path`` (if given), then apply env overrides.

    Passing no path returns the built-in defaults, which is why a fresh clone
    runs with ``kumaru chat`` and no setup at all.
    """
    data: dict[str, Any] = {}
    if path is not None:
        file_path = Path(path)
        if not file_path.is_file():
            raise ConfigError(
                f"config file not found: {file_path}",
                hint="pass --config with a path to a YAML file under configs/",
            )
        try:
            loaded = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"could not parse {file_path}: {exc}") from None
        data = loaded or {}

    config = apply_env_overrides(from_dict(data), env)
    config.validate()
    return config
