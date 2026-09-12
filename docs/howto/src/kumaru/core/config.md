# `src\kumaru\core\config.py`

> Typed configuration loader that merges defaults, YAML profiles, environment variables, and explicit overrides.

**Read this when:** you add a config field, debug which setting won, create a YAML profile, or wire a new CLI flag into runtime configuration.

---

## What it does
This file defines Kumaru's configuration models and the merge logic that builds one `KumaruConfig`. The merge order is defaults, YAML file or profile, `KUMARU_...` environment variables, then explicit overrides. Pydantic validates the final object and rejects unknown fields.

## Why it exists
Configuration precedence bugs are hard to diagnose during model serving and training. This module keeps the rule small and explicit so operators can reproduce a run and catch typos at startup rather than after a long generation or training job.

## Mental model
All subsystems read one resolved `KumaruConfig`. Nested environment variables use `KUMARU_SECTION__FIELD`, and CLI flags become the same nested dictionary shape. Lists replace lists during merge; they are not appended.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `ENV_PREFIX` | `str` | Environment variable prefix, `"KUMARU_"`. |
| `ENV_NESTED_DELIMITER` | `str` | Nested environment delimiter, `"__"`. |
| `REPO_ROOT` | `Path` | Repository root inferred from this file. |
| `CONFIG_DIR` | `Path` | Directory where bare profile names are resolved, `REPO_ROOT / "configs"`. |
| `BackendConfig` | `BackendConfig(...)` | Backend selection, model, hardware, adapter, remote, and GGUF options. |
| `AgentConfig` | `AgentConfig(...)` | System prompt, history budget, memory policy, and tool settings. |
| `ServerConfig` | `ServerConfig(...)` | HTTP host, port, CORS, concurrency, timeout, UI, and API key settings. |
| `TelemetryConfig` | `TelemetryConfig(...)` | Logging, metrics, and trace-file settings. |
| `KumaruConfig` | `KumaruConfig(backend=..., generation=..., agent=..., server=..., telemetry=...)` | Single typed config object every subsystem consumes. |
| `KumaruConfig.dump_yaml` | `dump_yaml(self) -> str` | Serialise the fully resolved config to YAML. |
| `deep_merge` | `deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]` | Recursively merge nested dictionaries with overlay precedence. |
| `env_overrides` | `env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]` | Convert `KUMARU_SECTION__FIELD` environment variables into nested overrides. |
| `load_config` | `load_config(path: str | Path | None = None, *, overrides: dict[str, Any] | None = None, use_env: bool = True) -> KumaruConfig` | Build and validate the final config. |
| `list_profiles` | `list_profiles() -> list[str]` | Return YAML profile names from `configs\`. |

## How to use it

Load defaults only:

```python
from kumaru.core.config import load_config

cfg = load_config(use_env=False)
print(cfg.backend.name)
```

Load a profile by bare name:

```python
from kumaru.core.config import load_config

cfg = load_config("local-gpu")
```

Apply highest-precedence overrides:

```python
from kumaru.core.config import load_config

cfg = load_config(
    "local-gpu",
    overrides={
        "backend": {"model": "Qwen/Qwen2.5-1.5B-Instruct"},
        "generation": {"temperature": 0.2, "max_tokens": 256},
    },
)
```

Build overrides from environment variables:

```python
from kumaru.core.config import env_overrides

overrides = env_overrides({
    "KUMARU_BACKEND__MODEL": "Qwen/Qwen2.5-1.5B-Instruct",
    "KUMARU_SERVER__PORT": "9000",
})

assert overrides == {
    "backend": {"model": "Qwen/Qwen2.5-1.5B-Instruct"},
    "server": {"port": 9000},
}
```

Print the resolved config for a bug report:

```python
from kumaru.core.config import load_config

cfg = load_config("local-gpu")
print(cfg.dump_yaml())
```

## How to extend it

Add new settings to the specific section model that owns them. Pydantic forbids unknown fields, so YAML, environment, and CLI overrides will fail until the model includes the field.

For example, to add a server request body limit:

```python
class ServerConfig(_Base):
    host: str = "127.0.0.1"
    port: int = 8080
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_concurrent_requests: int = 8
    request_timeout_s: float = 600.0
    serve_ui: bool = True
    api_key: str | None = None
    max_request_bytes: int = 2_000_000
```

Operators can then set it in YAML:

```yaml
server:
  max_request_bytes: 2000000
```

or through the environment:

```shell
set KUMARU_SERVER__MAX_REQUEST_BYTES=2000000
```

If it needs a CLI flag, add a mapping in `src\kumaru\cli.py` so the flat flag becomes `{"server": {"max_request_bytes": value}}`.

## Failure modes
| Symptom | Likely cause | Fix |
|---|---|---|
| `Config file not found: name` | The path does not exist and no `configs\name.yaml` profile exists. | Use a real path, create the profile, or run `kumaru info` / `list_profiles()`. |
| `must contain a YAML mapping at the top level` | The YAML file contains a list, scalar, or empty invalid shape. | Make the top level a mapping such as `backend: ...`. |
| `Invalid configuration: extra fields not permitted` | A YAML key or environment variable field is misspelled or not modelled. | Correct the key or add the field to the right config model. |
| Environment variable has no effect | It is not shaped as `KUMARU_SECTION__FIELD` or the section is unknown. | Use known section names such as `BACKEND`, `SERVER`, or `GENERATION`. |
| A list contains duplicated entries after merging was expected | Lists are replaced, not appended. | Put the complete desired list in the higher-precedence layer. |
| Tests vary by developer machine | Ambient `KUMARU_` variables are being read. | Call `load_config(..., use_env=False)` in tests. |

## Related files
- `src\kumaru\cli.py` — converts command-line flags into `overrides` for `load_config()`.
- `src\kumaru\core\types.py` — defines `GenerationParams`, embedded in `KumaruConfig`.
- `src\kumaru\core\errors.py` — provides `ConfigError`.
- `src\kumaru\backends\base.py` — backend constructors receive `BackendConfig`.
- `src\kumaru\agent\chat.py` — reads agent and generation settings from `KumaruConfig`.
