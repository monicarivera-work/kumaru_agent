# `configs/cpu-only.yaml`

> llama.cpp GGUF configuration for CPU-only local inference.

**Read this when:** running a real local model on a laptop without GPU acceleration.

---

## What it does
It selects `llamacpp`, points at a quantized GGUF file under `models/`, uses a 4096-token context, lets llama.cpp pick CPU threads, and disables GPU layer offload.

## Why it exists
CPU-only GGUF is the most reliable real-model path on machines without CUDA. It trades speed for simple local operation.

## Mental model
Download the referenced model path yourself or change `backend.model`. `n_threads: 0` means use every core according to runtime defaults.

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
pip install -e '.[llamacpp]'
kumaru serve -c configs/cpu-only.yaml
```

```yaml
backend:
  name: llamacpp
  model: models/qwen2.5-7b-instruct-q4_k_m.gguf
  gpu_layers: 0
```

```bash
KUMARU_MODEL=models/your-model.gguf kumaru serve -c configs/cpu-only.yaml
```

## How to extend it
Create variants for different GGUFs by changing `backend.model`, `context_length`, and optionally `n_threads`. Keep `gpu_layers: 0` for pure CPU.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| GGUF file not found | Model file has not been downloaded or path differs. | Put the file at `backend.model` or override `KUMARU_MODEL`. |
| Very slow output | CPU inference speed is hardware-bound. | Use smaller quantization/model or an OpenAI-compatible server on stronger hardware. |
| Out of memory | Model or context too large. | Use a smaller GGUF or lower `context_length`. |

## Related files
- [llama.cpp backend](../src/kumaru/backends/llamacpp_backend.md)
- [Low VRAM config](./low-vram.yaml.md)
- [Config loader](../src/kumaru/core/config.md)
