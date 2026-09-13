# `src/kumaru/core/config.py`

> Typed configuration loaded from defaults, YAML, and `KUMARU_*` environment variables.

**Read this when:** editing config files, adding settings, or debugging effective runtime config.

---

## What it does
It defines dataclasses for backend, agent, and server settings, validates unsafe values, and applies environment overrides. Secrets are read through environment variable names rather than committed YAML values.

## Why it exists
Raw dicts make typos silent and push failures into runtime. Typed sections fail fast with actionable `ConfigError` messages before Kumaru accepts traffic.

## Mental model
Precedence is defaults, then YAML, then environment overrides, then CLI overrides in `cli.py`. `load_config()` always returns a validated `KumaruConfig`.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `DEFAULT_SYSTEM_PROMPT` | `str` | Built-in assistant system prompt. |
| `BackendConfig` | `BackendConfig(name: str = "echo", model: str = "", base_url: str = "http://localhost:11434/v1", api_key_env: str = "KUMARU_API_KEY", timeout_s: float = 120.0, device: str = "auto", dtype: str = "auto", context_length: int = 4096, n_threads: int = 0, gpu_layers: int = 0, extra: dict[str, Any] = ...)` | Backend settings. |
| `BackendConfig.api_key` | `api_key(self) -> str` | Read secret from `api_key_env`. |
| `AgentConfig` | `AgentConfig(system_prompt: str = DEFAULT_SYSTEM_PROMPT, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95, max_history_turns: int = 12, max_history_chars: int = 12000, stop: list[str] = ...)` | Conversation settings. |
| `ServerConfig` | `ServerConfig(host: str = "127.0.0.1", port: int = 8000, cors_origins: list[str] = ..., serve_ui: bool = True)` | HTTP settings. |
| `KumaruConfig` | `KumaruConfig(backend: BackendConfig = ..., agent: AgentConfig = ..., server: ServerConfig = ..., log_level: str = "INFO", log_json: bool = False)` | Whole config tree. |
| `KumaruConfig.to_dict` | `to_dict(self) -> dict[str, Any]` | Plain safe dict. |
| `KumaruConfig.validate` | `validate(self) -> None` | Fail fast on invalid values. |
| `ENV_OVERRIDES` | `dict[str, str]` | Environment variable to dotted config path map. |
| `from_dict` | `from_dict(data: dict[str, Any]) -> KumaruConfig` | Build config from parsed YAML dict. |
| `apply_env_overrides` | `apply_env_overrides(config: KumaruConfig, env: dict[str, str] | None = None) -> KumaruConfig` | Mutate config with `KUMARU_*` values. |
| `load_config` | `load_config(path: str | Path | None = None, *, env: dict[str, str] | None = None) -> KumaruConfig` | Load YAML, apply env, validate. |

## How to use it
```python
from kumaru.core.config import load_config
config = load_config("configs/openai-compat.yaml")
print(config.backend.base_url)
```

```python
from kumaru.core.config import load_config
config = load_config(env={"KUMARU_BACKEND": "echo", "KUMARU_PORT": "9000"})
print(config.server.port)
```

```python
from kumaru.core.config import from_dict
config = from_dict({"agent": {"max_tokens": 128}})
config.validate()
```

```bash
KUMARU_MODEL=llama3.1:8b kumaru config -c configs/openai-compat.yaml
```

## How to extend it
Add a dataclass field, optionally add an `ENV_OVERRIDES` entry, then include validation if bad values would fail later. Example: add `seed: int = 0` to `AgentConfig`; `from_dict()` will accept YAML automatically.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| `unknown key(s)` | YAML key does not match a dataclass field. | Fix spelling or add the setting to the correct dataclass. |
| Secret missing despite YAML edit | `api_key` is intentionally not a YAML field. | Set the env var named by `backend.api_key_env`. |
| Env var type error | `_coerce` could not convert the string. | Use integer/float/list syntax matching the existing value type. |

## Related files
- [Default config](../../../configs/default.yaml.md)
- [CLI](../cli.md)
- [OpenAI backend](../backends/openai_compat.md)
