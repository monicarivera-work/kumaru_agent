# `configs/test.yaml`

> Deterministic no-network configuration for tests and CI.

**Read this when:** running Kumaru paths without model downloads, network calls, or slow streaming.

---

## What it does
It selects `echo`, uses a short test system prompt, low token/history limits, port 8765, and WARNING logs. The comments state it is for tests and CI.

## Why it exists
Tests should be deterministic and not require GPUs, weights, or external services. This config keeps end-to-end paths cheap and predictable.

## Mental model
Use this for local validation fixtures or CI processes. For direct Python tests, inject `EchoBackend(delay_s=0)` when you need no stream delay.

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
kumaru config -c configs/test.yaml
```

```bash
kumaru ask -c configs/test.yaml "ping"
```

```yaml
backend:
  name: echo
agent:
  system_prompt: "You are Kumaru (test mode)."
  max_tokens: 64
```

## How to extend it
Create narrower test profiles by reducing history limits or changing `server.port`. Keep `backend.name: echo` to avoid external dependencies.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Test expects real answer | Echo backend is deterministic, not intelligent. | Use a real backend only in explicit integration tests. |
| Port conflict | Test server uses 8765. | Override `KUMARU_PORT` or edit a test-specific copy. |
| Stream tests wait | EchoBackend default delay applies when constructed normally. | Inject `EchoBackend(..., delay_s=0)` in Python tests. |

## Related files
- [Echo backend](../src/kumaru/backends/echo.md)
- [Config loader](../src/kumaru/core/config.md)
- [Default config](./default.yaml.md)
