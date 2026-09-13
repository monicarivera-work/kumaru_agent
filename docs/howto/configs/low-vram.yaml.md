# `configs/low-vram.yaml`

> llama.cpp GGUF configuration for partial GPU offload on 6-8 GB VRAM machines.

**Read this when:** tuning a quantized model on limited GPU memory.

---

## What it does
It selects `llamacpp`, uses the same GGUF path as CPU config, keeps a 4096-token context, and offloads 20 layers to GPU.

## Why it exists
Partial offload improves speed without requiring enough VRAM for the full model. The comments describe the safe tuning loop: raise layers until failure, then back off.

## Mental model
KV cache grows with `context_length`; lower context before assuming the model itself is too large. `gpu_layers` is the primary VRAM tuning knob.

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
kumaru serve -c configs/low-vram.yaml
```

```yaml
backend:
  name: llamacpp
  context_length: 4096
  gpu_layers: 20
```

```bash
KUMARU_GPU_LAYERS=16 kumaru serve -c configs/low-vram.yaml
```

## How to extend it
Create hardware-specific profiles by changing `gpu_layers` in increments of 4 and lowering `context_length` if loading still fails.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Load fails or system swaps | Too many layers or too much KV cache for VRAM/RAM. | Lower `gpu_layers`, then lower `context_length`. |
| No speedup over CPU | GPU offload unavailable or too low. | Check llama.cpp build and raise `gpu_layers` carefully. |
| Model path error | Referenced GGUF is absent. | Download it or set `KUMARU_MODEL`. |

## Related files
- [llama.cpp backend](../src/kumaru/backends/llamacpp_backend.md)
- [CPU-only config](./cpu-only.yaml.md)
- [Local GPU config](./local-gpu.yaml.md)
