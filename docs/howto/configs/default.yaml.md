# `configs/default.yaml`

> Documented default configuration matching Kumaru's built-in defaults.

**Read this when:** starting a new config or checking default behavior.

---

## What it does
It selects the `echo` backend, sets the default prompt and sampling, binds the server to loopback port 8000, serves the UI, and uses plain INFO logs.

## Why it exists
Operators need a safe copy-edit starting point. Keeping defaults visible in YAML makes changes reviewable even though Kumaru can run with no config file.

## Mental model
This file should mirror dataclass defaults. Precedence remains defaults < YAML < `KUMARU_*` env vars < CLI flags.

## Public API
| Setting | Signature | What it's for |
|---|---|---|
| `backend.name` | `echo | openai_compat | transformers | llamacpp` | Select generation backend. |
| `backend.model` | `str` | Model id or GGUF/local path, depending on backend. |
| `backend.base_url` | `str` | OpenAI-compatible server base URL. |
| `backend.api_key_env` | `str` | Environment variable containing bearer token. |
| `backend.timeout_s` | `float` | HTTP timeout seconds. |
| `backend.device` | `str` | Transformers device, commonly `auto`, `cuda`, or `cpu`. |
| `backend.dtype` | `str` | Transformers dtype, commonly `auto`. |
| `backend.context_length` | `int` | llama.cpp context window. |
| `backend.n_threads` | `int` | llama.cpp CPU threads; `0` lets runtime decide. |
| `backend.gpu_layers` | `int` | llama.cpp layers offloaded to GPU. |
| `agent.system_prompt` | `str` | Prompt template; supports `{date}` and `{time}`. |
| `agent.max_tokens` | `int > 0` | Maximum generated tokens. |
| `agent.temperature` | `0.0..2.0` | Sampling randomness. |
| `agent.top_p` | `(0.0, 1.0]` | Nucleus sampling. |
| `agent.max_history_turns` | `int >= 1` | Max stored messages per session. |
| `agent.max_history_chars` | `int >= 1` | Max stored history characters. |
| `server.host` | `str` | Bind host. |
| `server.port` | `1..65535` | Bind port. |
| `server.serve_ui` | `bool` | Serve static UI when true. |
| `log_level` | `str` | Root logging level. |
| `log_json` | `bool` | Emit JSON logs when true. |

## How to use it
```bash
kumaru serve -c configs/default.yaml
```

```bash
kumaru config -c configs/default.yaml
```

```yaml
backend:
  name: echo
agent:
  max_tokens: 512
server:
  host: 127.0.0.1
  port: 8000
```

## How to extend it
Copy this file and change only the layer you need. Example: change `backend.name` to `openai_compat`, then add `model` and `base_url` from the OpenAI-compatible example.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Assistant only echoes | Default backend is `echo`. | Use a real backend config. |
| Cannot connect from another machine | Host is loopback. | Bind to another host only behind appropriate access controls. |
| Env overrides surprise you | `KUMARU_*` wins over YAML. | Inspect with `kumaru config -c configs/default.yaml`. |

## Related files
- [Config loader](../src/kumaru/core/config.md)
- [Echo backend](../src/kumaru/backends/echo.md)
- [OpenAI config](./openai-compat.yaml.md)
