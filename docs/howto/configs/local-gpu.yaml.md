# `configs/local-gpu.yaml`

> Transformers configuration for in-process GPU inference with unquantized weights.

**Read this when:** using CUDA-class hardware or local fine-tuned Hugging Face weights.

---

## What it does
It selects `transformers`, uses `Qwen/Qwen2.5-7B-Instruct`, and lets device/dtype auto-detection choose CUDA and bf16/fp16 when available.

## Why it exists
This profile supports direct Hugging Face model loading without a separate server. It is appropriate when VRAM is sufficient and local model customization matters.

## Mental model
First run may download weights to the Hugging Face cache. Swap `backend.model` for a local directory to use a fine-tune.

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
pip install -e '.[transformers]'
kumaru serve -c configs/local-gpu.yaml
```

```yaml
backend:
  name: transformers
  model: Qwen/Qwen2.5-7B-Instruct
  device: auto
  dtype: auto
```

```bash
KUMARU_MODEL=/models/my-finetune kumaru serve -c configs/local-gpu.yaml
```

## How to extend it
Add `backend.extra` for trusted Hugging Face loader options. Example: `extra: {trust_remote_code: true}` for a model that requires custom code and has been reviewed.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| CUDA OOM | Model is too large for VRAM. | Use a smaller model, quantization, or llama.cpp profile. |
| Runs on CPU unexpectedly | CUDA is unavailable or torch build lacks CUDA. | Install correct torch build and check drivers. |
| Slow first startup | Weights are downloading. | Wait for cache population or pre-download model files. |

## Related files
- [Transformers backend](../src/kumaru/backends/transformers_backend.md)
- [Low VRAM config](./low-vram.yaml.md)
- [Config loader](../src/kumaru/core/config.md)
