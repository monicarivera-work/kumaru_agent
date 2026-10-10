# `configs/local-gpu.yaml`

> Transformers serving profile for a single 24 GB CUDA GPU running Qwen2.5 7B in bfloat16 with a 16k context window.

**Read this when:** you are operating Kumaru on an RTX 3090, RTX 4090, A5000, or similar 24 GB Ampere-or-newer GPU.

---

## What it does

This profile runs `Qwen/Qwen2.5-7B-Instruct` locally with the `transformers` backend on CUDA, using `bfloat16`, a 16,384-token context, 2,048-token response cap, and moderate request concurrency.

The comments document the VRAM budget: about 15.2 GB for bf16 weights, about 0.9 GB for a 16k grouped-query KV cache on Qwen2.5, plus activations and fragmentation.

## When to use this profile

Use it when:

- one local CUDA GPU has about 24 GB VRAM;
- the GPU supports bf16 well, such as RTX 30-series, RTX 40-series, A5000, or A100;
- you want simpler local development rather than vLLM throughput;
- 16k context is useful and fits your workload.

Do not use it for 6-12 GB VRAM; use `configs/low-vram.yaml`. Do not use it for CPU-only machines; use `configs/cpu-only.yaml`.

## Every field explained

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `backend.name` | string | `transformers` | Loads the model through Hugging Face Transformers. | Correct for local GPU development. |
| `backend.model` | string | `Qwen/Qwen2.5-7B-Instruct` | Model checkpoint. | Keep for this profile, or choose another 7B-class instruct model that fits. |
| `backend.device` | string | `cuda` | Forces CUDA instead of auto placement. | Use `cuda:0` if you need a specific GPU. |
| `backend.dtype` | string | `bfloat16` | Weight/compute dtype. | Use bf16 on Ampere+; use `float16` only on older GPUs without bf16. |
| `backend.max_context_length` | int | `16384` | Prompt plus generation context cap. | Lower if KV cache or long prompts cause OOM. |
| `generation.max_tokens` | int | `2048` | Maximum reply length. | Lower for latency; keep if long responses are expected. |
| `generation.temperature` | float | `0.7` | Sampling randomness. | `0.7` is general chat; `0` for deterministic runs. |
| `generation.top_p` | float | `0.95` | Nucleus sampling cutoff. | Keep default unless output is too loose or too narrow. |
| `agent.max_history_tokens` | int | `12288` | Conversation history budget. | Leaves room for `generation.max_tokens` under 16k context. |
| `agent.memory_policy` | string | `sliding_window` | How history is trimmed. | Keeps system prompt and recent turns; safest default. |
| `server.host` | string | `127.0.0.1` | Server bind address. | Keep localhost unless you add `server.api_key`. |
| `server.port` | int | `8080` | Server port. | Change if another service uses 8080. |
| `server.max_concurrent_requests` | int | `4` | Number of admitted concurrent requests. | Keep low for one GPU; higher mostly adds queueing latency with Transformers. |
| `telemetry.log_level` | level name | `INFO` | Logging verbosity of `kumaru serve`. | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`, in any case (`WARN` and `FATAL` are aliases); any other name fails at load. Use `DEBUG` only while troubleshooting. |

## How to use it

Prerequisites, in the Kumaru folder:

```powershell
.\scripts\setup.ps1 -Gpu    # a CUDA build of PyTorch, then pip install -e ".[train]"
```

Outside Windows, `pip install -e ".[torch]"` in the Kumaru folder is enough; Linux PyTorch wheels from PyPI include CUDA.

Optional long-context attention optimization, if your CUDA toolchain supports it:

```powershell
pip install flash-attn --no-build-isolation
```

Then uncomment:

```yaml
backend:
  attn_implementation: flash_attention_2
```

Run:

```powershell
kumaru serve -c local-gpu
```

Terminal prompt:

```powershell
kumaru chat -c local-gpu -q "Summarize grouped-query attention."
```

Inspect resolved settings:

```powershell
kumaru config -c local-gpu
```

## Tuning it

1. If first request OOMs, reduce `backend.max_context_length` from `16384` to `8192`.
2. If long outputs are too slow, lower `generation.max_tokens` to `1024`.
3. If the GPU is shared, reduce `server.max_concurrent_requests` to `1` or `2`.
4. If attention memory dominates and you can install it, enable `flash_attention_2`.
5. If the model still does not fit, switch to `configs/low-vram.yaml`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is false | CUDA/PyTorch install mismatch. | Install a CUDA-enabled PyTorch build before running Kumaru. |
| OOM with long prompts | 16k KV cache plus output budget exceeds free VRAM. | Lower `max_context_length`, `max_history_tokens`, or `max_tokens`. |
| NaN/inf output on older GPU | GPU does not handle bf16 well. | Change `backend.dtype` to `float16` or use supported hardware. |
| Multiple users see high latency | Transformers backend does not continuously batch. | Keep concurrency low or switch to `configs/vllm.yaml`. |
| FlashAttention import/build failure | `flash-attn` or CUDA build toolchain is missing/incompatible. | Leave `attn_implementation` commented out. |

## Related files

- `configs/default.yaml` — full reference for omitted defaults.
- `configs/low-vram.yaml` — 4-bit profile for smaller GPUs.
- `configs/vllm.yaml` — throughput profile using an external vLLM server.
- `src/kumaru/core/config.py` — schema for all fields.

