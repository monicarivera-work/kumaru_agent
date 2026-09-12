# `configs/cpu-only.yaml`

> llama.cpp/GGUF serving profile for machines with no usable GPU.

**Read this when:** you need Kumaru to run on CPU, verify behavior without CUDA, or operate on a host where GPU inference is unavailable.

---

## What it does

This profile uses the `llamacpp` backend with a small Qwen2.5 1.5B GGUF model. It sets `n_gpu_layers: 0` for pure CPU execution, uses a 4096-token context, limits replies to 512 tokens, and admits one request at a time.

## When to use this profile

Use it on CPU-only desktops, laptops, CI-like smoke environments where `echo` is too shallow, or shared machines where GPU use is not allowed. Expect about 3-7 tok/s for 7B on a modern 8-core CPU, and better usability from 1.5B as this profile chooses.

## Every field explained

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `backend.name` | string | `llamacpp` | Uses llama.cpp bindings and GGUF kernels. | Best local CPU choice. |
| `backend.model` | string | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | GGUF model repo/id. | Keep for usable CPU speed; larger models are much slower. |
| `backend.n_gpu_layers` | int | `0` | Number of layers offloaded to GPU. | `0` means pure CPU; use `-1` only if a GPU is available. |
| `backend.n_threads` | int/null | `null` | llama.cpp CPU thread count. | Leave null for physical-core default; reduce when sharing the CPU. |
| `backend.max_context_length` | int | `4096` | Context cap. | Lower for memory; raise only after measuring latency. |
| `backend.extra.gguf_filename` | string | `*q4_k_m.gguf` | Pattern used to pick a GGUF file from the repo. | Keep `q4_k_m` for size/quality balance; choose another quant if you know why. |
| `generation.max_tokens` | int | `512` | Reply length cap. | Shorter replies keep CPU latency tolerable. |
| `generation.temperature` | float | `0.7` | Sampling randomness. | General chat default. |
| `agent.max_history_tokens` | int | `2560` | Retained conversation budget. | Keeps prompts small enough for CPU latency. |
| `server.port` | int | `8080` | HTTP port. | Change if occupied. |
| `server.max_concurrent_requests` | int | `1` | Admission limit. | CPU inference already saturates cores; keep one. |

## How to use it

Prerequisites:

```powershell
pip install "kumaru[gguf]"
```

Run:

```powershell
kumaru serve -c cpu-only
```

Terminal prompt:

```powershell
kumaru chat -c cpu-only -q "Explain gradient accumulation briefly."
```

Inspect resolved config:

```powershell
kumaru config -c cpu-only
```

## Tuning it

1. If latency is too high, switch to an even smaller GGUF model.
2. If the CPU is shared, set `backend.n_threads` below the physical core count.
3. If memory pressure is high, reduce `backend.max_context_length`.
4. If replies take too long, reduce `generation.max_tokens`.
5. If you later add a GPU, change `n_gpu_layers` or move to `low-vram`/`local-gpu`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `llama_cpp` import error | GGUF extra is missing. | Install `pip install "kumaru[gguf]"`. |
| Model download cannot find a file | `gguf_filename` pattern does not match repo files. | Choose a model repo with matching GGUF files or change the pattern. |
| Host becomes unresponsive | llama.cpp uses all cores. | Set `backend.n_threads` lower. |
| Very slow replies | CPU inference is inherently slow or model is too large. | Use a smaller model or GPU/remote backend. |
| Concurrent requests time out | `max_concurrent_requests` is one and CPU is busy. | Keep concurrency one; scale by adding hosts or using vLLM. |

## Related files

- `configs/test.yaml` — faster no-model smoke profile.
- `configs/low-vram.yaml` — quantized CUDA profile.
- `src/kumaru/core/config.py` — backend schema, including `extra`.

