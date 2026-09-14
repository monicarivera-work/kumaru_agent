# `configs/openai-compat.yaml`

> Configuration for any OpenAI-compatible chat-completions server.

**Read this when:** pointing Kumaru at Ollama, llama.cpp server, LM Studio, vLLM, or OpenAI-compatible APIs.

---

## What it does
It selects `openai_compat`, sets a model name, base URL, API-key environment variable name, timeout, larger max tokens, and loopback server binding.

## Why it exists
HTTP-compatible model servers are the easiest scaling boundary. Move model inference to another process or machine and update `base_url` without changing Kumaru UI/agent code.

## Mental model
Secrets are never stored in YAML; `api_key_env` names the variable read at request time. Local servers may ignore the bearer token.

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
kumaru serve -c configs/openai-compat.yaml
```

```bash
KUMARU_API_KEY=sk-... kumaru serve -c configs/openai-compat.yaml
```

```yaml
backend:
  name: openai_compat
  model: llama3.1:8b
  base_url: http://localhost:11434/v1
```

## How to extend it
Make provider profiles by changing `base_url`, `model`, and optional `backend.extra`. Example: for vLLM set `base_url: http://localhost:8000/v1` and the served model name.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| HTTP 404 model not found | `model` does not match server inventory. | List models on the server and update `backend.model`. |
| Connection refused | Server is not running at `base_url`. | Start model server or fix host/port. |
| Authentication failed | Required API key env var is unset/wrong. | Set the variable named by `api_key_env`. |

## Related files
- [OpenAI backend](../src/kumaru/backends/openai_compat.md)
- [Default config](./default.yaml.md)
- [Config loader](../src/kumaru/core/config.md)
