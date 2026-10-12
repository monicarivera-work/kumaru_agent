# `configs/low-vram.yaml`

> 4-bit Transformers profile for serving Qwen2.5 7B on 6-12 GB CUDA GPUs.

**Read this when:** you have a small or laptop NVIDIA GPU and want the 7B model to fit locally.

---

## What it does

This profile uses the `transformers` backend with `load_in_4bit: true`. The base model is stored in 4-bit NF4 while compute uses `bfloat16`, reducing model memory enough for an 8k context on 6-12 GB VRAM cards.

## When to use this profile

Use it for GPUs such as RTX 3060 12 GB, RTX 4060, RTX 2080, and many laptop GPUs. It targets single-user or light local use, not production throughput.

Do not use it without the quantization extras. Do not expect identical quality to bf16; comments in the source call out small but real degradation in long-chain reasoning and exact recall.

## Every field explained

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `backend.name` | string | `transformers` | Local HF backend. | Required for this profile. |
| `backend.model` | string | `Qwen/Qwen2.5-7B-Instruct` | 7B instruct checkpoint. | Keep unless you need a smaller/larger compatible model. |
| `backend.device` | string | `cuda` | Forces GPU execution. | Use `cuda:0` for a specific card. |
| `backend.dtype` | string | `bfloat16` | Compute dtype; storage is still 4-bit. | Use bf16 on Ampere+; change to `float16` if hardware requires it. |
| `backend.load_in_4bit` | bool | `true` | Enables NF4 quantized weight loading. | Keep true; this is the point of the profile. |
| `backend.max_context_length` | int | `8192` | Context cap. | Lower to 4096 if KV cache causes OOM. |
| `generation.max_tokens` | int | `1024` | Reply cap. | Lower for latency or tighter memory headroom. |
| `agent.max_history_tokens` | int | `5120` | Conversation history budget. | Leaves room for a 1024-token answer inside 8192 context. |
| `server.port` | int | `8080` | HTTP port. | Change if occupied. |
| `server.max_concurrent_requests` | int | `2` | Admission limit. | Keep small; small GPUs OOM under concurrent long generations. |

## How to use it

Prerequisites, in the Kumaru folder. The profile runs on the `transformers` backend, so it needs `-Gpu` as well as `-Quant`:

```powershell
.\scripts\setup.ps1 -Gpu -Quant    # elsewhere: pip install -e ".[torch,quant]"
```

Run:

```powershell
kumaru serve -c low-vram
```

Run a terminal prompt:

```powershell
kumaru chat -c low-vram -q "What does 4-bit quantization cost?"
```

Inspect resolved settings:

```powershell
kumaru config -c low-vram
```

## Tuning it

1. Lower `backend.max_context_length` to `4096` if loading succeeds but long prompts OOM.
2. Lower `generation.max_tokens` to `512` if completions OOM or latency is too high.
3. Lower `server.max_concurrent_requests` to `1` on 6-8 GB cards.
4. If quality is insufficient and you have 24 GB VRAM, switch to `local-gpu`.
5. If it still does not fit, choose a smaller model with `--model`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Optional dependency 'bitsandbytes' is not installed.` when the model loads | The `quant` extra is missing; Kumaru checks for it before the weights load. | `.\scripts\setup.ps1 -Quant`, or `pip install -e ".[quant]"` in the Kumaru folder (the hint prints this). |
| OOM despite 4-bit loading | KV cache, activations, or concurrency exceed VRAM. | Lower context and concurrency. |
| Outputs are slightly worse than bf16 | NF4 quantization quality trade-off. | Use `local-gpu` if hardware permits. |
| Slow first run | Hugging Face weights are downloading. | Wait for cache population; subsequent starts are faster. |
| `bfloat16` unsupported | Older GPU architecture. | Try `backend.dtype: float16`. |

## Related files

- `configs/local-gpu.yaml` — bf16 profile for 24 GB GPUs.
- `configs/cpu-only.yaml` — GGUF CPU profile.
- `src/kumaru/core/config.py` — authoritative field schema.

