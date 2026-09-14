# `src/kumaru/backends/transformers_backend.py`

> In-process Hugging Face Transformers backend for local or fine-tuned models.

**Read this when:** running weights directly inside Kumaru without a separate model server.

---

## What it does
It lazily imports torch and transformers, loads tokenizer/model, formats chat prompts, serializes generation with a lock, and supports full and streamed generation.

## Why it exists
Operators with enough VRAM may want one process and direct access to local fine-tunes or LoRA-compatible directories. Lazy imports keep bare installs usable without torch.

## Mental model
`backend.model` is a Hugging Face id or local path. `device=auto` prefers CUDA; `dtype=auto` chooses bf16/fp16 on CUDA and float32 on CPU.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `TransformersBackend` | `TransformersBackend(config: BackendConfig)` | Load tokenizer and model. |
| `TransformersBackend.name` | `str` | Always `transformers`. |
| `TransformersBackend.generate` | `generate(self, request: ChatRequest) -> GenerationResult` | Run blocking model generation. |
| `TransformersBackend.stream` | `stream(self, request: ChatRequest) -> Iterator[StreamChunk]` | Stream via `TextIteratorStreamer` when available. |
| `TransformersBackend.health` | `health(self) -> dict[str, object]` | Report backend, model, device, ready. |
| `TransformersBackend.close` | `close(self) -> None` | Delete model and clear CUDA cache when available. |

## How to use it
```yaml
backend:
  name: transformers
  model: Qwen/Qwen2.5-7B-Instruct
  device: auto
  dtype: auto
```

```python
from kumaru.core.config import BackendConfig
from kumaru.backends.transformers_backend import TransformersBackend
backend = TransformersBackend(BackendConfig(name="transformers", model="Qwen/Qwen2.5-1.5B-Instruct"))
```

```python
from kumaru.core.types import ChatRequest, Message
result = backend.generate(ChatRequest(messages=[Message.user("Say hi")], max_tokens=16))
print(result.text)
```

```bash
kumaru serve -c configs/local-gpu.yaml
```

## How to extend it
Pass Hugging Face loader options through `backend.extra`. Example: set `extra: {trust_remote_code: true}` only for models you intentionally trust.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Requires torch + transformers | Optional dependencies are missing. | Install the `transformers` extra and an appropriate torch build. |
| OOM while loading | Model/context/dtype too large for hardware. | Use a smaller model, quantized backend, lower context, or CPU. |
| Unknown dtype | `backend.dtype` is not a torch dtype attribute. | Use values such as `float16`, `bfloat16`, `float32`, or `auto`. |
| Concurrent requests queue | A lock serializes one model. | Run more processes or a dedicated model server for concurrency. |

## Related files
- [Local GPU config](../../../configs/local-gpu.yaml.md)
- [Backend base](./base.md)
- [Llama.cpp backend](./llamacpp_backend.md)
