"""Configuration: one typed object assembled from four ordered sources.

Precedence (later wins)
-----------------------
1. **Field defaults** in this file — always valid, so Kumaru runs with zero setup.
2. **YAML file** (``configs/*.yaml``) — the reproducible, commit-to-git layer.
3. **Environment variables** (``KUMARU_BACKEND__MODEL=...``) — the deployment
   layer; how you change a container's behaviour without rebuilding it.
4. **Explicit overrides** (CLI flags) — the "just this once" layer.

Why implement the merge by hand instead of leaning on a settings framework:
precedence bugs in configuration are miserable to debug, and the rule above is
about forty lines of obvious code. Being able to *read* the rule is worth more
than the abstraction.

Everything is validated by Pydantic, so a typo in a YAML key fails loudly at
startup rather than silently doing the wrong thing 40 minutes into a run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from kumaru.core.errors import ConfigError
from kumaru.core.types import GenerationParams

ENV_PREFIX = "KUMARU_"
ENV_NESTED_DELIMITER = "__"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


class _Base(BaseModel):
    # `extra="forbid"` converts a silent typo into a startup error. This single
    # setting removes an entire category of "why is my config being ignored" bugs.
    model_config = ConfigDict(extra="forbid")


class BackendConfig(_Base):
    """Which inference engine to load, and how."""

    name: str = "transformers"
    model: str = "Qwen/Qwen2.5-7B-Instruct"

    # -- local hardware placement (transformers / llamacpp) ------------------
    device: str = "auto"  # auto | cuda | cuda:0 | cpu | mps
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    load_in_4bit: bool = False  # QLoRA-style NF4 quantization; needs [quant]
    load_in_8bit: bool = False
    attn_implementation: str | None = None  # e.g. "flash_attention_2" when available
    max_context_length: int = 8192
    trust_remote_code: bool = False

    # LoRA adapters produced by `kumaru train`. Loading an adapter is how you
    # deploy a fine-tune without touching the multi-gigabyte base weights.
    adapter_path: str | None = None

    # -- remote backends (openai_compat: vLLM, Ollama, TGI, hosted APIs) -----
    base_url: str | None = None
    api_key: str | None = None
    timeout_s: float = 300.0

    # -- GGUF / llama.cpp ----------------------------------------------------
    model_path: str | None = None
    n_gpu_layers: int = -1  # -1 = offload every layer it can fit
    n_threads: int | None = None

    extra: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(_Base):
    """How the conversation itself is managed."""

    system_prompt: str = (
        "You are Kumaru, a precise and helpful assistant. "
        "Answer directly and concisely. Lead with the answer, then justify it. "
        "If you are uncertain, say so plainly and state what would resolve the "
        "uncertainty. Never invent citations, APIs, file paths, or numbers."
    )
    # Budget for conversation history, in tokens. When exceeded, the memory
    # policy decides what to drop or compress.
    max_history_tokens: int = 6144
    memory_policy: str = "sliding_window"  # sliding_window | drop_oldest | none
    # Reserved for the tool layer; empty means "pure chat", which is v0.1.
    enabled_tools: list[str] = Field(default_factory=list)
    max_tool_iterations: int = 4


class ServerConfig(_Base):
    """HTTP surface."""

    host: str = "127.0.0.1"
    port: int = 8080
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    # Guards against one huge request starving every other caller.
    max_concurrent_requests: int = 8
    request_timeout_s: float = 600.0
    serve_ui: bool = True
    api_key: str | None = None  # when set, requests must send a matching bearer token


class TelemetryConfig(_Base):
    """Observability. Off-box telemetry is opt-in and off by default."""

    log_level: str = "INFO"
    log_format: str = "console"  # console | json
    metrics_enabled: bool = True  # in-process counters exposed at /metrics
    trace_file: str | None = None  # JSONL of every request, for offline analysis


class KumaruConfig(_Base):
    """The single object every subsystem reads from."""

    backend: BackendConfig = Field(default_factory=BackendConfig)
    generation: GenerationParams = Field(default_factory=GenerationParams)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    def dump_yaml(self) -> str:
        """Serialize the *fully resolved* config.

        Print this when reporting a bug or recording an experiment: it captures
        every layer already merged, which makes a run reproducible.
        """
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, width=100)


# ---------------------------------------------------------------------------
# Source merging
# ---------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `overlay` into `base`, returning a new dict.

    Nested dicts merge key-by-key; every other type (including lists) is
    replaced wholesale. Replacing lists is deliberate — "append or replace?" is
    ambiguous, and silent appending produces duplicated stop-sequences and
    duplicated CORS origins that take an hour to track down.
    """
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(raw: str) -> Any:
    """Parse an env var string into a Python scalar.

    YAML's scalar rules give us ``true``/``false``, ints, floats, and quoted
    strings for free, and fall back to the raw string when parsing fails.
    """
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    return raw if parsed is None and raw != "null" else parsed


def env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Build a nested dict from ``KUMARU_SECTION__FIELD`` variables.

    ``KUMARU_BACKEND__MODEL=Qwen/Qwen2.5-1.5B-Instruct``
        -> ``{"backend": {"model": "Qwen/Qwen2.5-1.5B-Instruct"}}``
    """
    environ = os.environ if environ is None else environ
    known_sections = set(KumaruConfig.model_fields)
    out: dict[str, Any] = {}

    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower().split(ENV_NESTED_DELIMITER)
        # Skip bare knobs like KUMARU_LOG_LEVEL that are not section paths.
        if len(path) < 2 or path[0] not in known_sections:
            continue
        cursor = out
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _coerce(raw)
    return out


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    use_env: bool = True,
) -> KumaruConfig:
    """Assemble the final config from all four sources.

    Parameters
    ----------
    path:
        YAML file, or a bare profile name resolved against ``configs/``
        (``load_config("local-gpu")`` -> ``configs/local-gpu.yaml``).
    overrides:
        Highest-precedence nested dict, normally built from CLI flags.
    use_env:
        Set ``False`` in tests so the ambient environment cannot leak in.
    """
    data: dict[str, Any] = {}

    if path is not None:
        resolved = Path(path)
        if not resolved.exists() and resolved.suffix == "":
            resolved = CONFIG_DIR / f"{path}.yaml"
        if not resolved.exists():
            raise ConfigError(
                f"Config file not found: {path}",
                hint=f"Available profiles: {', '.join(list_profiles()) or '(none)'}",
            )
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"Config file {resolved} must contain a YAML mapping at the top level")
        data = deep_merge(data, loaded)

    if use_env:
        data = deep_merge(data, env_overrides())

    if overrides:
        data = deep_merge(data, overrides)

    try:
        return KumaruConfig(**data)
    except Exception as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc


def list_profiles() -> list[str]:
    """Names of the YAML profiles shipped in ``configs/``."""
    if not CONFIG_DIR.exists():
        return []
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
