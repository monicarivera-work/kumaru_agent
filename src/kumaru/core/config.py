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
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


def default_models_dir() -> str:
    """Where downloaded GGUF weights live unless `backend.models_dir` says otherwise."""
    return str(Path.home() / ".cache" / "kumaru" / "models")


class LlamaServerOptions(_Base):
    """Knobs for the managed `llama-server` process (backend `llamacpp_server`).

    Kumaru starts llama.cpp's own HTTP server as a child process and talks to it
    over the OpenAI protocol. llama.cpp ships new model architectures and kernels
    weeks before Python bindings do, so this is how Kumaru stays current: update
    one binary, not your code.
    """

    # Path to llama-server(.exe). None = tools/llama.cpp/ (installed by
    # scripts/get-llama-server.ps1), then whatever is on PATH.
    binary: str | None = None
    host: str = "127.0.0.1"
    port: int | None = None  # None = pick a free port at startup
    # Concurrent sequences ("slots"). Continuous batching interleaves them in
    # one forward pass, so 4 parallel requests cost far less than 4x the time.
    parallel: int = Field(default=4, ge=1, le=64)
    # One shared KV-cache pool for all slots (instead of ctx/parallel each), so a
    # single long conversation can still use the whole context window.
    kv_unified: bool = True
    flash_attn: Literal["on", "off", "auto"] = "on"
    # KV-cache precision: f16 (default) | q8_0 | q4_0. q8_0 halves KV memory at a
    # small quality cost; useful for very long contexts on small GPUs.
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    # Mixture-of-Experts offload: keep the expert weights of the first N layers
    # in system RAM. The lever that fits a 35B MoE on a 12-16 GB GPU.
    n_cpu_moe: int | None = Field(default=None, ge=0)
    batch_size: int | None = Field(default=None, ge=1)  # -b, logical batch for prompts
    ubatch_size: int | None = Field(default=None, ge=1)  # -ub, physical batch
    # Text injected when a thinking budget runs out, nudging the model to answer.
    reasoning_budget_message: str = (
        "\n\nI have used my thinking budget, so I will now give my best final answer.\n"
    )
    startup_timeout_s: float = Field(default=900.0, gt=0)
    log_file: str | None = None  # None = runs/logs/llama-server-<port>.log
    extra_args: list[str] = Field(default_factory=list)  # appended verbatim

    @field_validator("flash_attn", mode="before")
    @classmethod
    def _flash_attn_from_bool(cls, value: Any) -> Any:
        # YAML 1.1, which also parses env overrides, reads a bare on/off as a bool.
        if isinstance(value, bool):
            return "on" if value else "off"
        return value


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
    # transformers only. Greedy (temperature 0) requests that arrive together
    # are decoded as one batch of up to this many; 1 turns batching off. A
    # decoding step reads every weight once for the whole batch, so a batch of
    # 32 runs nearly as fast as one request: what makes a full benchmark of a
    # small model take minutes, not hours. Sampled requests never batch.
    max_batch_size: int = Field(default=1, ge=1, le=512)

    # LoRA adapters produced by `kumaru train`. Loading an adapter is how you
    # deploy a fine-tune without touching the multi-gigabyte base weights.
    adapter_path: str | None = None

    # -- remote backends (openai_compat: vLLM, Ollama, TGI, hosted APIs) -----
    base_url: str | None = None
    api_key: str | None = None
    timeout_s: float = 300.0

    # -- GGUF / llama.cpp ----------------------------------------------------
    model_path: str | None = None
    n_gpu_layers: int = -1  # -1 = all (llamacpp_server: as many as fit); see default.yaml
    n_threads: int | None = None
    # Download/cache directory for GGUF weights resolved from "org/repo:QUANT"
    # specs. Point it at your biggest, fastest disk (KUMARU_BACKEND__MODELS_DIR).
    models_dir: str = Field(default_factory=default_models_dir)
    llama_server: LlamaServerOptions = Field(default_factory=LlamaServerOptions)

    extra: dict[str, Any] = Field(default_factory=dict)


class ThinkingModeConfig(_Base):
    """Sampling preset applied *only* when thinking is on.

    Reasoning models are tuned with different decoding settings for their two
    modes (Qwen3.6: temperature 1.0 when thinking, 0.7 when not). Precedence:
    ``generation`` < ``thinking_mode`` (if thinking) < per-request overrides.
    Every field is optional; unset fields fall through to ``generation``.
    """

    max_tokens: int | None = Field(default=None, ge=1, le=131_072)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    repetition_penalty: float | None = Field(default=None, ge=0.5, le=2.0)
    thinking_budget: int | None = Field(default=None, ge=0)


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
    # Reserved for the tool layer; empty means "pure chat", the only mode so far.
    enabled_tools: list[str] = Field(default_factory=list)
    max_tool_iterations: int = 4
    expertise: list[Literal["coding", "playwright", "windows", "apis"]] = Field(default_factory=list)
    knowledge_files: list[str] = Field(default_factory=list)
    max_knowledge_chars: int = Field(default=6000, ge=0, le=24000)
    # Documentation indexes built by `kumaru learn TOPIC --source FOLDER`. The
    # passages matching each question are added to it, within this budget.
    knowledge_indexes: list[str] = Field(default_factory=list)
    max_retrieved_chars: int = Field(default=6000, ge=0, le=24000)


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

    # The levels that both Python logging and uvicorn understand.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: str = "console"  # console | json
    metrics_enabled: bool = True  # in-process counters exposed at /metrics
    trace_file: str | None = None  # JSONL of every request, for offline analysis

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: Any) -> Any:
        # Accept "warn", " Info " and similar; reject typos at load time rather
        # than as a KeyError deep inside uvicorn when the server starts.
        if isinstance(value, str):
            level = value.strip().upper()
            return {"WARN": "WARNING", "FATAL": "CRITICAL"}.get(level, level)
        return value


class KumaruConfig(_Base):
    """The single object every subsystem reads from."""

    backend: BackendConfig = Field(default_factory=BackendConfig)
    generation: GenerationParams = Field(default_factory=GenerationParams)
    thinking_mode: ThinkingModeConfig = Field(default_factory=ThinkingModeConfig)
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
