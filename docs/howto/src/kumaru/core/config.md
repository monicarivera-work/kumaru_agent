# `src\kumaru\core\config.py`

> Typed configuration: one `KumaruConfig` assembled from field defaults, a YAML profile, `KUMARU_*` environment variables, and CLI-flag overrides, validated with unknown keys rejected.

**Read this when:**

- you add a config field or a profile;
- you debug which setting won;
- you tune the managed `llama-server` (`backend.llama_server`) or the model cache (`backend.models_dir`);
- you set the thinking preset (`thinking_mode`);
- you wire a new CLI flag.

---

## What it does

This file defines the configuration models and the merge that builds one `KumaruConfig`. The top-level sections, in order, are:

`backend`, `generation`, `thinking_mode`, `agent`, `server`, `telemetry`

v0.2 adds:

- `backend.llama_server` (`LlamaServerOptions`), the knobs for the managed llama.cpp `llama-server` process.
- `backend.models_dir`, the GGUF download/cache directory. It defaults to `default_models_dir()`.
- The `thinking_mode` section (`ThinkingModeConfig`), a sampling preset applied only when thinking is on.
- New `generation` fields, defined on `GenerationParams` in `src\kumaru\core\types.py`: `min_p`, `presence_penalty`, `thinking`, `thinking_budget`.

## Why it exists

Precedence bugs in configuration are miserable to debug during serving and training. This module keeps the rule in about forty lines of plain code instead of a settings framework, so you can read it. Every model in this file sets `extra="forbid"`, so a typo in a YAML key, env var, or override fails at startup instead of being silently ignored for a 40-minute run. The exception is `generation`, which is `GenerationParams` from `src\kumaru\core\types.py` and ignores unknown keys.

## Mental model

Precedence, from lowest to highest (later wins). Verified with `load_config` and `kumaru config`.

1. **Field defaults** in this file. Always valid, so Kumaru runs with zero setup.
2. **YAML** — a file path, or a bare profile name resolved to `configs\<name>.yaml`. This is the reproducible, commit-to-git layer.
3. **Environment** — `KUMARU_<SECTION>__<FIELD>[__<SUBFIELD>...]`. The deployment layer.
4. **Explicit overrides** — a nested dict, normally built from CLI flags. The "just this once" layer.

Merge rules (`deep_merge`):

- Nested dicts merge key by key.
- Lists and scalars are **replaced**, never appended. Silent appending would produce duplicated stop sequences and CORS origins.

Pydantic validates the merged dict once, at the end.

### Environment variables

`env_overrides` builds the nested dict from the environment:

1. It keeps keys that start with `KUMARU_`.
2. It strips the prefix, **lowercases** the rest, and splits it on `__`.
3. It requires at least two parts, the first of which must be a known section. Anything else is skipped silently. That includes `KUMARU_LOG_LEVEL`/`KUMARU_LOG_FORMAT` (read by `src\kumaru\core\logging.py`) and a misspelled section. Nesting depth is unlimited.
4. It parses values with YAML scalar rules (`_coerce`):
   - `18` becomes an int, and `true` becomes a bool.
   - `["a", "b"]` becomes a list, and `null` becomes `None`.
   - An empty value stays `""`.
   - Anything unparsable stays the raw string.

The YAML parsing has a sharp edge: YAML 1.1 reads `on`/`off`/`yes`/`no` as booleans and digit strings as ints. String fields given such values fail validation, so quote them, for example `'"off"'`.

### Thinking preset

`thinking_mode` applies **only when thinking resolves to `True`**: the request's `thinking` if set, otherwise `generation.thinking`. Its non-null fields then override `generation`, and per-request values override both, so the order is `generation` < `thinking_mode` < request. `ChatAgent.resolve_params()` implements this.

It exists because reasoning models are tuned with different sampling per mode. Qwen3.6, for example, recommends temperature 1.0 when thinking and 0.7 when not.

With `generation.thinking: null` (the model's default) the preset is **not** applied, even if the model thinks by default. Profiles for hybrid models should therefore set `generation.thinking` explicitly; `qwen36` sets `false`.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `ENV_PREFIX` | `str = "KUMARU_"` | Environment-variable prefix. |
| `ENV_NESTED_DELIMITER` | `str = "__"` | Separator between path segments in env names. |
| `REPO_ROOT` | `Path` = `Path(__file__).resolve().parents[3]` | Repository root of a source checkout. |
| `CONFIG_DIR` | `Path` = `REPO_ROOT / "configs"` | Where bare profile names resolve. |
| `default_models_dir` | `default_models_dir() -> str` | `str(Path.home() / ".cache" / "kumaru" / "models")`; default for `backend.models_dir`. |
| `LlamaServerOptions` | `LlamaServerOptions(binary: str \| None = None, host: str = "127.0.0.1", port: int \| None = None, parallel: int = 4, kv_unified: bool = True, flash_attn: Literal["on", "off", "auto"] = "on", cache_type_k: str \| None = None, cache_type_v: str \| None = None, n_cpu_moe: int \| None = None, batch_size: int \| None = None, ubatch_size: int \| None = None, reasoning_budget_message: str = "\n\nI have used my thinking budget, so I will now give my best final answer.\n", startup_timeout_s: float = 900.0, log_file: str \| None = None, extra_args: list[str] = [])` | Knobs for the managed `llama-server` child (`backend.llama_server`). Bounds: `parallel` 1–64, `n_cpu_moe` ≥ 0, `batch_size`/`ubatch_size` ≥ 1, `startup_timeout_s` > 0. |
| `BackendConfig` | `BackendConfig(name: str = "transformers", model: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "auto", dtype: str = "auto", load_in_4bit: bool = False, load_in_8bit: bool = False, attn_implementation: str \| None = None, max_context_length: int = 8192, trust_remote_code: bool = False, max_batch_size: int = 1, adapter_path: str \| None = None, base_url: str \| None = None, api_key: str \| None = None, timeout_s: float = 300.0, model_path: str \| None = None, n_gpu_layers: int = -1, n_threads: int \| None = None, models_dir: str = default_models_dir(), llama_server: LlamaServerOptions = LlamaServerOptions(), extra: dict[str, Any] = {})` | Which inference engine to load, and how. Fields are shared by all backends, and each backend reads only its own. For example, `adapter_path` (a LoRA from `kumaru train`) is used by the `transformers` backend only; every other backend refuses to load while it is set. |
| `ThinkingModeConfig` | `ThinkingModeConfig(max_tokens: int \| None = None, temperature: float \| None = None, top_p: float \| None = None, top_k: int \| None = None, min_p: float \| None = None, presence_penalty: float \| None = None, repetition_penalty: float \| None = None, thinking_budget: int \| None = None)` | Sampling preset used only when thinking is on. Same bounds as `GenerationParams`: `max_tokens` 1–131072, `temperature` 0–2, `top_p` 0–1, `top_k` ≥ 0, `min_p` 0–1, `presence_penalty` −2–2, `repetition_penalty` 0.5–2, `thinking_budget` ≥ 0. |
| `AgentConfig` | `AgentConfig(system_prompt: str = "You are Kumaru, a precise and helpful assistant. …", max_history_tokens: int = 6144, memory_policy: str = "sliding_window", enabled_tools: list[str] = [], max_tool_iterations: int = 4)` | System prompt, history budget, memory policy, tools. |
| `ServerConfig` | `ServerConfig(host: str = "127.0.0.1", port: int = 8080, cors_origins: list[str] = ["*"], max_concurrent_requests: int = 8, request_timeout_s: float = 600.0, serve_ui: bool = True, api_key: str \| None = None)` | HTTP surface. |
| `TelemetryConfig` | `TelemetryConfig(log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO", log_format: str = "console", metrics_enabled: bool = True, trace_file: str \| None = None)` | Logging and observability. A `mode="before"` validator, `_normalize_log_level`, strips and upper-cases a string level and maps `WARN` to `WARNING` and `FATAL` to `CRITICAL`. Any other name fails at load, not later inside uvicorn. |
| `KumaruConfig` | `KumaruConfig(backend: BackendConfig = ..., generation: GenerationParams = ..., thinking_mode: ThinkingModeConfig = ..., agent: AgentConfig = ..., server: ServerConfig = ..., telemetry: TelemetryConfig = ...)` | The single object every subsystem reads; every section defaults to its own defaults. |
| `KumaruConfig.dump_yaml` | `dump_yaml(self) -> str` | The fully resolved config as YAML (`sort_keys=False`), for bug reports and run records. It does not redact anything: `backend.api_key` and `server.api_key` appear in plain text. |
| `deep_merge` | `deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]` | New dict; nested dicts merge, everything else (lists included) is replaced. |
| `env_overrides` | `env_overrides(environ: dict[str, str] \| None = None) -> dict[str, Any]` | `KUMARU_*` variables as a nested dict (`os.environ` when `environ` is `None`). |
| `load_config` | `load_config(path: str \| Path \| None = None, *, overrides: dict[str, Any] \| None = None, use_env: bool = True) -> KumaruConfig` | Build and validate the final config; raises `ConfigError`. |
| `list_profiles` | `list_profiles() -> list[str]` | Sorted stems of `configs\*.yaml`: `cpu-only`, `default`, `local-gpu`, `low-vram`, `qwen25-7b-gguf`, `qwen36`, `test`, `vllm`. |

### Field notes that are not obvious from the types

**`backend`**

- `models_dir` is where `org/repo:QUANT` specs are found or downloaded, as `<models_dir>\org\repo\`. `llamacpp_server` applies `expanduser()`, so the `~/.cache/kumaru/models` in `configs\default.yaml` works. Point it at your largest, fastest disk.
- `max_context_length` is also the `llama-server` `--ctx-size`. The KV cache it sizes is the main VRAM cost after the weights. The CLI's `--context` sets it.
- `max_batch_size` (1–512, default 1) is read by `transformers` only: greedy requests that arrive together are decoded as one batch of up to this many, and 1 turns batching off. See [`transformers_backend.md`](../backends/transformers_backend.md#mental-model).

**`backend.llama_server`**

| Field | Meaning |
|---|---|
| `binary` | `None` searches `tools\llama.cpp\` (installed by `scripts\get-llama-server.ps1`), then `PATH`. |
| `port` | `None` picks a free port at startup. |
| `parallel` | The number of concurrent slots; continuous batching serves them in one forward pass. |
| `kv_unified` | One shared KV pool, so a single long conversation can use the whole window. |
| `flash_attn` | `on` \| `off` \| `auto`; anything else fails validation at load (`literal_error`). YAML 1.1, which also parses env overrides, reads a bare `on`/`off` (and `yes`/`no`) as a boolean; the validator maps it back, so `KUMARU_BACKEND__LLAMA_SERVER__FLASH_ATTN=off` works. |
| `cache_type_k`, `cache_type_v` | `f16` \| `q8_0` \| `q4_0`. `q8_0` halves KV memory at a small quality cost. |
| `n_cpu_moe` | Keeps the expert weights of the first N layers in system RAM; the lever that fits a 35B mixture-of-experts model on a 12–16 GB GPU. |
| `batch_size`, `ubatch_size` | `-b`/`-ub`. |
| `reasoning_budget_message` | Injected when `thinking_budget` runs out. |
| `log_file` | `None` writes `runs\logs\llama-server-<port>.log`. |
| `extra_args` | Appended verbatim to the command line. |

The flag mapping is in `docs\howto\src\kumaru\backends\llamacpp_server.md`.

**`thinking_mode` and `generation`**

`thinking_mode.max_tokens` also sets how much context the agent reserves for the reply when thinking (see `src\kumaru\agent\chat.py`). Thinking needs a much larger cap, which is why `qwen36` sets 32768.

**Reserved fields, declared but not read by v0.2 code**

`server.request_timeout_s`, `telemetry.metrics_enabled`, `telemetry.trace_file`, and `agent.max_tool_iterations`. Setting them validates but changes nothing.

## How to use it

### Load a profile, apply overrides, print what won

```python
from kumaru.core.config import load_config

cfg = load_config(
    "qwen36",
    overrides={"backend": {"max_context_length": 32768}, "generation": {"thinking": True}},
)
print(cfg.dump_yaml())
```

From the shell, `kumaru config -c <profile> [-b BACKEND] [-m MODEL]` prints the same resolved YAML. It is the fastest way to answer "which value won?".

### Override from the environment (PowerShell)

```powershell
$env:KUMARU_BACKEND__LLAMA_SERVER__N_CPU_MOE = "18"     # nested: backend.llama_server.n_cpu_moe -> 18 (int)
$env:KUMARU_BACKEND__MODELS_DIR = "D:\kumaru\models"   # backend.models_dir
$env:KUMARU_THINKING_MODE__TEMPERATURE = "0.9"         # thinking_mode.temperature
$env:KUMARU_GENERATION__THINKING = "true"              # generation.thinking -> True (bool)
$env:KUMARU_SERVER__CORS_ORIGINS = '["http://localhost:3000", "http://127.0.0.1:3000"]'  # a list
$env:KUMARU_BACKEND__LLAMA_SERVER__FLASH_ATTN = '"off"' # quoted: bare off would become False
kumaru config -c qwen36
```

`$env:` lasts for the current shell. `setx KUMARU_BACKEND__MODELS_DIR D:\kumaru\models` persists the value for **new** shells only.

The nested form was checked in Python:

```python
import os
from kumaru.core.config import load_config

os.environ["KUMARU_BACKEND__LLAMA_SERVER__N_CPU_MOE"] = "18"
assert load_config("qwen36").backend.llama_server.n_cpu_moe == 18
```

### See precedence in action

```powershell
$env:KUMARU_BACKEND__MODEL = "env-model"
kumaru config -c test                    # backend.model: env-model   (env beats YAML)
kumaru config -c test -m flag-model      # backend.model: flag-model  (CLI flag beats env)
Remove-Item Env:KUMARU_BACKEND__MODEL
```

### Set a thinking preset in a profile

```yaml
generation:            # used when thinking is off
  temperature: 0.7
  top_p: 0.8
  thinking: false      # explicit, so the preset below is not skipped
thinking_mode:         # applied on top of `generation` only when thinking is on
  max_tokens: 32768
  temperature: 1.0
  top_p: 0.95
  thinking_budget: null   # null = unlimited thinking
```

`kumaru chat -c <profile> --think`, a request with `"thinking": true`, or the UI toggle then uses `temperature` 1.0 and `max_tokens` 32768. Request fields still win.

### Keep tests hermetic

```python
cfg = load_config("test", use_env=False, overrides={"backend": {"extra": {"delay_s": 0}}})
```

## How to extend it

Add a setting to the section model that owns it. Because of `extra="forbid"`, YAML, env, and overrides all reject the key until the field exists.

1. Add the field with a safe default and `Field(...)` bounds:

   ```python
   class ServerConfig(_Base):
       # ...existing fields...
       max_request_bytes: int = Field(default=2_000_000, ge=1)
   ```

2. Document it in `configs\default.yaml`, which is the annotated reference profile.
3. Read it where it is used. A field nobody reads is a trap: see the reserved fields above.
4. It is settable from the environment immediately: `$env:KUMARU_SERVER__MAX_REQUEST_BYTES = "1000000"`.
5. For a CLI flag, add the flag to the command in `src\kumaru\cli.py`. Then add a `"flag_name": ("server", "max_request_bytes")` entry to `_build_overrides`'s mapping.

To add a whole section:

- Add a `_Base` model.
- Add it as a field on `KumaruConfig`. That alone makes it a known env section, because `env_overrides` reads `KumaruConfig.model_fields`.

A new generation knob belongs on `GenerationParams`; see `docs\howto\src\kumaru\core\types.md`. If it is part of the thinking preset, mirror it on `ThinkingModeConfig`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Config file not found: <name>` with `hint: Available profiles: ...` | No such path, and no `configs\<name>.yaml`. | Use a listed profile or a real path; `kumaru info` lists profiles too. |
| `hint: Available profiles: (none)` | `CONFIG_DIR` is `REPO_ROOT\configs`, which exists only in a source checkout or editable install. | Pass a file path, or install with `pip install -e .`. |
| An API key shows up in a pasted `kumaru config` dump | `dump_yaml` serialises every field verbatim, including `backend.api_key` and `server.api_key`. (Bench run records are different: the runner redacts credential-like keys before saving the config.) | Redact `api_key:` lines before sharing a dump. |
| `must contain a YAML mapping at the top level` | The YAML's top level is a list or scalar. | Make it a mapping (`backend: ...`). An empty file is fine. |
| `Invalid configuration: ... Extra inputs are not permitted` | Misspelled or unmodelled key in YAML, env, or overrides. For example, `KUMARU_BACKEND__N_CPU_MOE` is missing the `LLAMA_SERVER__` level. | Fix the path, or add the field. |
| `Invalid configuration: ... telemetry.log_level ... Input should be 'DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'` | A level name outside the five (after stripping, upper-casing, and the `WARN`/`FATAL` aliases), from YAML, `KUMARU_TELEMETRY__LOG_LEVEL` or `serve --log-level`. | Use one of the five names, in any case. |
| `Invalid configuration: ... Input should be a valid string` (the full Pydantic message adds `input_type=bool` or `int`; the CLI drops that bracketed part) | An env value was parsed as YAML: `off` became `False`, and `12345` became an int (for example in `KUMARU_SERVER__API_KEY`). | Quote it inside the value: `$env:KUMARU_BACKEND__LLAMA_SERVER__FLASH_ATTN = '"off"'`. |
| An environment variable has no effect | Not shaped `KUMARU_<SECTION>__<FIELD>`, a single `_` where `__` belongs, or an unknown section. All of these are skipped silently. A `setx` value is also invisible to the current shell. | Check the spelling against the section names, open a new shell after `setx`, and confirm with `kumaru config`. |
| `thinking_mode` values are ignored | Thinking resolved to `None` (model default) or `False`; the preset applies only on `True`. | Set `generation.thinking`, pass `--think`, or send `"thinking": true`. |
| A list lost its earlier entries | Lists are replaced, not appended. | Put the complete list in the highest-precedence layer. |
| `The <name> backend cannot attach a LoRA adapter (backend.adapter_path is '<path>').` when the model loads | Only the `transformers` backend attaches a LoRA adapter (`supports_adapters = True`). `Backend.load()` refuses the field on every other backend rather than serve the base model under the adapter's name. | Serve the adapter with `backend.name: transformers` and its base model. To use it with a GGUF backend, merge it and convert the merged model to GGUF yourself; Kumaru has no step for that. |
| GGUF re-downloads, or lands on the wrong disk | `backend.models_dir` differs between runs (env vs YAML). | Pin it in one layer (`setx KUMARU_BACKEND__MODELS_DIR ...`) and confirm with `kumaru config`. |
| Tests vary by machine | Ambient `KUMARU_*` variables leak in. | `load_config(..., use_env=False)`. |

## Known limitations

- **Four fields are reserved and read by nothing** (see "Reserved fields" above): `server.request_timeout_s`, `telemetry.metrics_enabled`, `telemetry.trace_file` and `agent.max_tool_iterations`. `/metrics` is always served, and a request waits at most 600 s for a loading backend, whatever they say.
- **`cors_origins: ["*"]` is paired with `allow_credentials=True`** (`src\kumaru\server\app.py`). Starlette then reflects every `Origin`, so CORS lets any web page open in your browser call the local API and read the answers, `POST /v1/backend` included. Workaround: `cors_origins: []` (the bundled UI is same-origin and needs no entry) or your own origins. `server.api_key` also stops such pages, but the bundled UI sends no key.
- **`kumaru config` prints API keys.** `dump_yaml` redacts nothing. Workaround: delete the `api_key:` lines before sharing.
- **An all-digit API key from an environment variable becomes an int**, and validation then fails with `Input should be a valid string`. Workaround: quote it inside the value, `$env:KUMARU_SERVER__API_KEY = '"12345"'`.
- **`server.max_concurrent_requests` has no minimum.** `0` makes every generation wait forever, and no error is logged; a negative value fails at startup with `ValueError: Semaphore initial value must be >= 0`. Workaround: keep it ≥ 1, and ≥ `llama_server.parallel`.
- **`agent.memory_policy` is a free string.** An unknown name passes validation; `ChatAgent` then logs `unknown memory_policy '<name>'; using sliding_window` when it is built, after the model loads. Workaround: `sliding_window`, `drop_oldest` or `none`.

## Related files

- `configs\default.yaml` — annotated reference for every key, including `llama_server`, `models_dir`, and `thinking_mode`; `configs\qwen36.yaml` and `configs\qwen25-7b-gguf.yaml` are the v0.2 GGUF profiles.
- `src\kumaru\cli.py` — `_build_overrides` turns flags into the `overrides` dict.
- `src\kumaru\core\types.py` — `GenerationParams`, embedded as `generation`.
- `src\kumaru\agent\chat.py` — `resolve_params()` applies `generation` < `thinking_mode` < request.
- `src\kumaru\backends\llamacpp_server.py` — consumes `llama_server`, `models_dir`, `max_context_length`.
- `src\kumaru\core\logging.py` — reads `KUMARU_LOG_LEVEL`/`KUMARU_LOG_FORMAT`, which are outside this scheme.
- `src\kumaru\core\errors.py` — `ConfigError`.
