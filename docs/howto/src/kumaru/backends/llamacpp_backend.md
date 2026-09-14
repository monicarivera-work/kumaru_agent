# `src/kumaru/backends/llamacpp_backend.py`

> In-process llama.cpp backend for quantized GGUF models.

**Read this when:** running local CPU/GPU-layer-offloaded GGUF weights.

---

## What it does
It validates the `.gguf` path, lazily imports `llama_cpp`, creates a `Llama` context, serializes access with a lock, and exposes full/streamed chat completions.

## Why it exists
GGUF models make real local inference practical on laptops without CUDA. `gpu_layers` lets operators offload part of the model when limited VRAM is available.

## Mental model
`backend.model` must be a local GGUF file path. `context_length`, `n_threads`, `gpu_layers`, and `extra` are passed into the llama.cpp runtime.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `LlamaCppBackend` | `LlamaCppBackend(config: BackendConfig)` | Load a GGUF model. |
| `LlamaCppBackend.name` | `str` | Always `llamacpp`. |
| `LlamaCppBackend.generate` | `generate(self, request: ChatRequest) -> GenerationResult` | Run non-streamed chat completion. |
| `LlamaCppBackend.stream` | `stream(self, request: ChatRequest) -> Iterator[StreamChunk]` | Yield llama.cpp streamed deltas. |
| `LlamaCppBackend.health` | `health(self) -> dict[str, object]` | Report model file, context length, GPU layers, ready. |
| `LlamaCppBackend.close` | `close(self) -> None` | Close llama context if supported. |

## How to use it
```yaml
backend:
  name: llamacpp
  model: models/qwen2.5-7b-instruct-q4_k_m.gguf
  context_length: 4096
  n_threads: 0
  gpu_layers: 0
```

```python
from kumaru.core.config import BackendConfig
from kumaru.backends.llamacpp_backend import LlamaCppBackend
backend = LlamaCppBackend(BackendConfig(name="llamacpp", model="models/model.gguf"))
```

```python
from kumaru.core.types import ChatRequest, Message
for chunk in backend.stream(ChatRequest(messages=[Message.user("hello")])):
    print(chunk.delta, end="")
```

```bash
kumaru serve -c configs/cpu-only.yaml
```

## How to extend it
Pass advanced llama.cpp constructor options via `backend.extra`. Example: `extra: {chat_format: chatml}` if a GGUF requires a specific chat format.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| GGUF file not found | `backend.model` path is wrong or relative to another cwd. | Use a valid path, preferably under `models/` from the repo root. |
| Import error for llama_cpp | Optional dependency is missing. | Install the `llamacpp` extra. |
| OOM or swapping | Context or GPU layers too high. | Lower `context_length` or `gpu_layers`; use a smaller quantization. |
| Requests serialize | llama.cpp context is protected by a lock. | Run multiple processes for parallelism. |

## Related files
- [CPU config](../../../configs/cpu-only.yaml.md)
- [Low VRAM config](../../../configs/low-vram.yaml.md)
- [Backend base](./base.md)
