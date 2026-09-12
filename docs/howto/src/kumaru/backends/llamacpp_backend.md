# `src\kumaru\backends\llamacpp_backend.py`

> Runs quantized GGUF models through `llama.cpp` for low-VRAM and CPU-only inference.

**Read this when:** you need inference on a small GPU, no GPU, a local `.gguf` file, or a Hugging Face GGUF repository.

---

## What it does

`LlamaCppBackend` loads `llama_cpp.Llama` from either `backend.model_path` or a Hugging Face repo id in `backend.model`. It passes `n_ctx`, `n_gpu_layers`, and optional `n_threads` to llama.cpp, streams chat completions, enforces stop sequences, and returns usage and backend info. It is registered as `llamacpp` with alias `gguf`.

## Why it exists

Transformers usually wants a full model in VRAM in 16-bit. GGUF supports aggressive k-quant formats such as Q4_K_M, Q5_K_M, and Q8_0, plus transparent CPU/GPU layer splitting. Without this backend, 8 GB VRAM and CPU-only machines would need remote inference or much smaller models.

## Mental model

1. GGUF is the model artifact; choose quantization for your memory/quality target.
2. `n_gpu_layers` is the main speed knob: `-1` offloads as many layers as fit, `0` is pure CPU.
3. Decoding is memory-bandwidth-bound, so moving layers from CPU memory to GPU memory usually improves tokens/sec until VRAM runs out.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `LlamaCppBackend` | `class LlamaCppBackend(config: BackendConfig)` | GGUF backend registered as `llamacpp` and `gguf`. |
| `LlamaCppBackend.name` | `str = "llamacpp"` | Registry and introspection name. |
| `LlamaCppBackend.aclose` | `async def aclose(self) -> None` | Drops the llama.cpp object and marks unloaded. |
| `LlamaCppBackend.info` | `def info(self) -> BackendInfo` | Reports model path/id, device (`cpu` or `cuda+cpu`), `dtype="gguf"`, context, quantization, and `n_gpu_layers`. |
| `LlamaCppBackend.count_tokens` | `def count_tokens(self, messages: Sequence[Message]) -> int` | Uses llama.cpp tokenization when loaded, otherwise the base heuristic. |
| `LlamaCppBackend.stream` | `async def stream(self, messages: Sequence[Message], params: GenerationParams) -> AsyncIterator[StreamChunk]` | Calls `create_chat_completion(..., stream=True)`, maps events to chunks, and tracks stops. |

## How to use it

### Run CPU-only

```powershell
pip install -e ".[gguf]"
kumaru serve -c cpu-only
kumaru chat -c cpu-only -q "Give one CPU inference tip."
```

```yaml
backend:
  name: llamacpp
  model: Qwen/Qwen2.5-1.5B-Instruct-GGUF
  n_gpu_layers: 0
  n_threads: null
  max_context_length: 4096
  extra:
    gguf_filename: "*q4_k_m.gguf"

generation:
  max_tokens: 512
```

### Load a local GGUF

```yaml
backend:
  name: llamacpp
  model: local-gguf
  model_path: C:\models\qwen2.5-7b-instruct-q4_k_m.gguf
  n_gpu_layers: -1
  max_context_length: 8192
```

```powershell
kumaru serve -c configs\my-gguf.yaml
```

### Pull a GGUF from a Hugging Face repo

```yaml
backend:
  name: llamacpp
  model: Qwen/Qwen2.5-7B-Instruct-GGUF
  n_gpu_layers: -1
  max_context_length: 8192
  extra:
    gguf_filename: "*Q4_K_M.gguf"
```

The backend treats `backend.model` as a repo id when the local path does not exist and the value contains `/`.

### Instantiate from Python

```python
from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS
from kumaru.core.types import Message

cfg = load_config("cpu-only")
backend = BACKENDS.create(cfg.backend.name, cfg.backend)
await backend.load()
async for chunk in backend.stream([Message.user("hello")], cfg.generation):
    print(chunk.delta, end="")
await backend.aclose()
```

## Tuning / operational notes

| Knob | Effect | Trade-off / expected number |
|---|---|---|
| `model_path` | Local `.gguf` file. | Avoids hub lookup and filename ambiguity; must exist. |
| `model` | Local fallback or HF repo id. | If it contains `/`, backend calls `Llama.from_pretrained(...)`. |
| `extra.gguf_filename` | Filename/pattern used with HF repos. | Defaults to `*Q4_K_M.gguf`; CPU profile uses `*q4_k_m.gguf`. |
| `n_gpu_layers` | Number of layers offloaded to GPU. | Tune upward until just below OOM; `-1` tries as many as fit, `0` is CPU-only. |
| `n_threads` | CPU thread count. | `null` lets llama.cpp choose; CPU inference typically saturates cores. |
| `max_context_length` | Passed as `n_ctx`. | Larger context consumes more KV cache; CPU-only profile uses 4096. |
| `generation.max_tokens` | Output cap. | CPU-only profile uses 512 to keep latency tolerable. |
| `server.max_concurrent_requests` | Admission limit. | CPU-only profile uses 1 because CPU generation uses all cores. |

Quantitative expectations from source/config: a 7B CPU model on a modern 8-core CPU is about 3-7 tok/s; a 1.5B model is about 15-30 tok/s. A 7B Q4_K_M GGUF is about 4.4 GB and fits in 8 GB VRAM.

## How to extend it

Add llama.cpp-specific constructor options through `BackendConfig.extra` and copy them into `_load()` kwargs:

```python
kwargs = {
    "n_ctx": cfg.max_context_length,
    "n_gpu_layers": cfg.n_gpu_layers,
    "verbose": False,
}
if "n_batch" in cfg.extra:
    kwargs["n_batch"] = int(cfg.extra["n_batch"])
```

```yaml
backend:
  name: llamacpp
  model_path: C:\models\model.gguf
  n_gpu_layers: -1
  extra:
    n_batch: 512
```

Keep chat formatting delegated to llama.cpp's `create_chat_completion`; the code relies on the chat template embedded in the GGUF.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `DependencyMissingError: llama-cpp-python` | GGUF optional dependency is missing. | Run `pip install -e ".[gguf]"`. |
| `GGUF model not found` | `model_path` does not exist and `model` is not a repo id. | Set a valid `.gguf` path or a repo id such as `Qwen/Qwen2.5-7B-Instruct-GGUF`. |
| Wrong GGUF downloaded | `extra.gguf_filename` pattern is wrong or too broad. | Use an exact filename or tighter pattern. |
| OOM at load | Too many GPU layers or too much context. | Lower `n_gpu_layers`, lower `max_context_length`, or choose a smaller/lower-bit GGUF. |
| CPU generation is too slow | Model/context/output too large. | Use 1.5B Q4, lower `max_tokens`, keep concurrency at 1, or offload layers to GPU. |
| Bad chat behavior | GGUF lacks the expected chat template or is a base model. | Use an instruct GGUF with embedded template. |

## Related files

- `src\kumaru\backends\base.py` — async contract, thread bridge, and stop tracker.
- `src\kumaru\core\config.py` — `model_path`, `n_gpu_layers`, `n_threads`, context, and `extra` settings.
- `src\kumaru\core\registry.py` — lazy registration as `llamacpp` and `gguf`.
- `configs\cpu-only.yaml` — CPU-only GGUF profile.
- `configs\default.yaml` — documents llama.cpp knobs.
- `src\kumaru\backends\transformers_backend.py` — local bf16 GPU path when the model fits.
- `src\kumaru\backends\openai_compat.py` — better path for many concurrent users.
