# `src\kumaru\backends\transformers_backend.py`

> Runs Hugging Face causal language models locally with optional quantization and LoRA adapters.

**Read this when:** you operate the default local backend, tune GPU memory and dtype, deploy a LoRA adapter, or debug prompt formatting and local generation.

---

## What it does

`TransformersBackend` loads an `AutoTokenizer` and `AutoModelForCausalLM`, resolves device and dtype, optionally applies bitsandbytes 4-bit/8-bit quantization, optionally attaches a PEFT LoRA adapter, and streams text through `TextIteratorStreamer`. It renders prompts with the tokenizer's chat template when present and falls back to a simple role-labeled format for base models. It is registered as `transformers` with alias `hf`.

## Why it exists

This is Kumaru's local-first development backend and the path that serves LoRA adapters produced by `kumaru train`. Without it, operators would need a separate inference server for local experiments. The file centralizes decisions that are easy to get wrong: bf16 vs fp16, device maps, pad-token repair, quantization config, adapter loading, context limits, and chat templates.

## Mental model

1. Single-stream decoding is memory-bandwidth-bound: a 7B bf16 model is about 15 GB of weights, so an RTX 3090 at about 936 GB/s has a rough ceiling near 60 tok/s and should see about 30-45 tok/s in practice.
2. Quantization reduces bytes read per token, so it saves VRAM and can improve speed, with a quality cost.
3. Prompt format is part of model behavior; use the tokenizer chat template when available.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `TransformersBackend` | `class TransformersBackend(config: BackendConfig)` | Local Hugging Face backend registered as `transformers` and `hf`. |
| `TransformersBackend.name` | `str = "transformers"` | Registry and introspection name. |
| `TransformersBackend.aclose` | `async def aclose(self) -> None` | Drops model/tokenizer references, resets state, runs GC, and empties CUDA cache when available. |
| `TransformersBackend.info` | `def info(self) -> BackendInfo` | Reports model id, device, dtype, context, quantization, parameter count, and adapter path. |
| `TransformersBackend.count_tokens` | `def count_tokens(self, messages: Sequence[Message]) -> int` | Counts tokens with the model tokenizer and rendered chat prompt, falling back to the base heuristic. |
| `TransformersBackend.stream` | `async def stream(self, messages: Sequence[Message], params: GenerationParams) -> AsyncIterator[StreamChunk]` | Loads on demand, renders prompt, checks budget, runs `model.generate()` on a thread, and yields chunks. |

## How to use it

### Serve the default local backend

```powershell
pip install -e ".[torch]"
kumaru serve -c default
kumaru chat -c default -q "Write one sentence about Kumaru."
```

```yaml
backend:
  name: transformers
  model: Qwen/Qwen2.5-7B-Instruct
  device: auto
  dtype: auto
  load_in_4bit: false
  load_in_8bit: false
  max_context_length: 8192
```

### Use the 24 GB GPU profile

```powershell
kumaru serve -c local-gpu
kumaru chat -c local-gpu --max-tokens 512 -q "Summarize the backend options."
```

```yaml
backend:
  name: transformers
  model: Qwen/Qwen2.5-7B-Instruct
  device: cuda
  dtype: bfloat16
  max_context_length: 16384

generation:
  max_tokens: 2048
```

### Fit a 7B model on 6-12 GB VRAM

```powershell
pip install -e ".[torch,quant]"
kumaru serve -c low-vram
```

```yaml
backend:
  name: transformers
  model: Qwen/Qwen2.5-7B-Instruct
  device: cuda
  dtype: bfloat16
  load_in_4bit: true
  max_context_length: 8192
```

### Attach a trained LoRA adapter

```powershell
kumaru chat -c local-gpu --adapter outputs\lora-qwen\final -q "Use the fine-tuned behavior."
```

```yaml
backend:
  name: transformers
  model: Qwen/Qwen2.5-7B-Instruct
  adapter_path: outputs\lora-qwen\final
  extra:
    merge_adapter: false
```

### Instantiate from Python

```python
from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS
from kumaru.core.types import Message

cfg = load_config("low-vram")
backend = BACKENDS.create(cfg.backend.name, cfg.backend)
await backend.load()
result = await backend.generate([Message.user("hello")], cfg.generation)
print(result.message.content)
await backend.aclose()
```

## Tuning / operational notes

| Knob | Effect | Trade-off / expected number |
|---|---|---|
| `backend.device` | `auto` chooses CUDA, then MPS, then CPU. | For CPU-only operation, `llamacpp` is usually faster than PyTorch CPU. |
| `backend.dtype` | `auto` chooses bf16 on supported CUDA, otherwise fp16; CPU uses fp32. | bf16 has fp32-like exponent range and avoids common fp16 inf/NaN failures. |
| `load_in_4bit` | Bitsandbytes NF4 with double quantization. | 7B drops from about 15 GB bf16 to about 4.5 GB; roughly doubles the bandwidth ceiling with small quality loss. |
| `load_in_8bit` | Bitsandbytes 8-bit loading. | Less memory reduction and less quality loss than 4-bit. |
| `attn_implementation` | Passed to `from_pretrained`, e.g. `flash_attention_2`. | Local profile notes FlashAttention cuts attention memory from O(n^2) to O(n) for long prompts. |
| `max_context_length` | Caps prompt + generation and is clamped to detected model context. | Local GPU profile budgets 16k context at about 0.9 GB KV cache for Qwen2.5-7B due to grouped-query attention. |
| `extra.device_map` | Defaults to `auto` on CUDA. | Allows accelerate sharding/spillover; CPU spill avoids OOM but slows generation. |
| `adapter_path` / `extra.merge_adapter` | Loads or merges a PEFT adapter. | Merging removes adapter overhead but prevents runtime adapter swapping. |
| `server.max_concurrent_requests` | Admission limit around local generation. | `local-gpu` uses 4; for many users use `openai_compat` with vLLM. |

## How to extend it

Prefer backend-specific options under `BackendConfig.extra` unless the setting belongs to all backends. The backend already supports custom device maps:

```yaml
backend:
  name: transformers
  model: Qwen/Qwen2.5-7B-Instruct
  device: cuda
  dtype: bfloat16
  extra:
    device_map: cuda:0
```

When adding optional features, follow the current pattern: import optional packages inside the method that needs them, raise `DependencyMissingError(package, extra=...)` when absent, and keep `__init__()` side-effect-free. Keep prompt rendering through `tokenizer.apply_chat_template()` unless the tokenizer has no template.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `DependencyMissingError: transformers` | ML extras are not installed. | Run `pip install -e ".[torch]"`. |
| `DependencyMissingError: bitsandbytes` | 4/8-bit loading requested without quant extra. | Run `pip install -e ".[torch,quant]"`. |
| CUDA out of memory during load | Model, KV cache, adapter, or fragmentation exceeds VRAM. | Use `low-vram`, lower context, enable 4-bit, or switch to `llamacpp`. |
| NaN/inf output | fp16 overflow on CUDA. | Use `dtype: bfloat16` on Ampere+ or `dtype: auto`. |
| Prompt is longer than context | Rendered prompt exceeds `max_context_length` after reserving output budget. | Lower `agent.max_history_tokens` or `generation.max_tokens`. |
| Answers are subtly bad | Wrong model type or missing chat template. | Use an instruct/chat checkpoint with a valid tokenizer chat template. |
| `Adapter not found` | Bad `adapter_path`. | Correct the path or train an adapter with `kumaru train ...`. |

## Related files

- `src\kumaru\backends\base.py` — backend contract, thread bridge, and stop tracking.
- `src\kumaru\core\config.py` — `BackendConfig` fields used here.
- `src\kumaru\core\types.py` — messages, params, stream chunks, usage, and info.
- `src\kumaru\core\registry.py` — lazy registration as `transformers` and `hf`.
- `configs\default.yaml` — default transformers profile.
- `configs\local-gpu.yaml` — 24 GB bf16 GPU profile.
- `configs\low-vram.yaml` — 4-bit low-VRAM profile.
- `src\kumaru\train\finetune_lora.py` — produces adapters loaded by `adapter_path`.
- `src\kumaru\backends\openai_compat.py` — vLLM/remote throughput alternative.
