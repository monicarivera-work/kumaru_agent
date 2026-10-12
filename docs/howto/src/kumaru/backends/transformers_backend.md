# `src\kumaru\backends\transformers_backend.py`

> Runs Hugging Face causal LMs in-process with PyTorch: bf16/fp16/fp32, optional bitsandbytes 4-bit or 8-bit, and the only backend that attaches LoRA adapters.

**Read this when:**
- you run `default`, `local-gpu` or `low-vram`, or any config with `backend.name: transformers` (alias `hf`);
- you deploy a LoRA adapter produced by `kumaru train`;
- you tune dtype, quantization or device placement, or debug prompt format, thinking or out-of-memory errors.

---

## What it does

`TransformersBackend` is registered as `transformers`, alias `hf`; it is the `BackendConfig` default.

**`load()`:**

1. **Device.** `backend.device: auto` resolves to `cuda`, then `mps`, then `cpu`.
2. **Dtype.** `backend.dtype: auto` resolves to bf16 when the CUDA device supports it, else fp16. Off CUDA it is fp32. Explicit names: `bfloat16`, `bf16`, `float16`, `fp16`, `float32`, `fp32`.
3. **Tokenizer.** `AutoTokenizer.from_pretrained(model, trust_remote_code=...)`. A missing pad token is set to the EOS token.
4. **Model.** `AutoModelForCausalLM.from_pretrained(model, ...)` with:
   - `dtype`;
   - `low_cpu_mem_usage=True`;
   - `attn_implementation`, if set;
   - a bitsandbytes `quantization_config`, if `load_in_4bit` or `load_in_8bit`. Kumaru first checks that bitsandbytes is installed and raises `DependencyMissingError` (extra `quant`) if not, because transformers ships `BitsAndBytesConfig` without it;
   - on CUDA, `device_map = extra.device_map`, default `"auto"`.

   transformers older than 4.56 get `torch_dtype` instead of `dtype`. Off CUDA, the model is moved with `.to(device)`.
5. **Adapter.** If `backend.adapter_path` is set, `PeftModel.from_pretrained` layers the adapter on top. `extra.merge_adapter: true` then folds it into the weights with `merge_and_unload()`.
6. **Context clamp.** `backend.max_context_length` is lowered to the checkpoint's `max_position_embeddings` (or `n_positions`, or `seq_length`) when that is smaller. This mutates the config object.

**`stream()`:**

1. **Render.** The prompt comes from `tokenizer.apply_chat_template(..., add_generation_prompt=True)`. `enable_thinking=<params.thinking>` is added only when `thinking` is not `None`. A tokenizer with no template falls back to `ROLE: content` lines plus `ASSISTANT:`.
2. **Budget.** It raises `BackendError` when the prompt fills the context window. Otherwise `max_new_tokens = min(params.max_tokens, max_context_length - prompt_tokens)`.
3. **Generate.** `model.generate` runs on its own daemon thread and feeds a `TextIteratorStreamer` (`skip_prompt=True`, `skip_special_tokens=True`, `timeout=600`). A second thread, from `iterate_in_thread`, drains it. A `StoppingCriteria` (`HaltWhenAsked`) reads a per-request `threading.Event` (`halt`) after every token, so the drain can stop `generate` early.
4. **Split.** Each piece goes through `ReasoningSplitter`, then `StopSequenceTracker`. The splitter starts in reasoning mode when the rendered prompt ends in `<think>`. The last chunk carries `finish_reason` and `Usage`.

### What changed in v0.2

| v0.2 | v0.1 |
|---|---|
| `GenerationParams.thinking` becomes the template variable `enable_thinking`, when not `None`. | Not passed. |
| `<think>…</think>` streams on the reasoning channel; a template that pre-opens `<think>` is detected. | Thinking streamed as answer text. |
| `min_p` is forwarded when sampling and non-zero. | Not sent. |
| `presence_penalty` and `thinking_budget` are accepted and ignored. | — |
| A stop sequence, a client disconnect or the UI's Stop halts `generate` a few tokens later. | `generate` ran on to EOS or `max_new_tokens` after the stream had ended. |
| `completion_tokens` counts the tokens `generate` produced. `finish_reason` is `"length"` only when the budget ran out on a token other than EOS. | Both came from a count of the streamer's text pieces, which are roughly words: `completion_tokens` undercounted, and a truncated reply usually reported `"stop"`. |

## Why it exists

- **Any Hugging Face checkpoint, unconverted.** It needs safetensors and a tokenizer, and nothing else. It is the reference path for prompt format: `count_tokens()` renders the real chat template.
- **The training loop closes here.** It is the only backend that attaches `adapter_path` (`supports_adapters = True`). `kumaru train` writes adapters; this backend serves them.
- **Where it is not the best choice:**
  - GGUF quantization, the newest architectures, thinking budgets or continuous batching → [`llamacpp_server`](llamacpp_server.md);
  - many concurrent users → [`openai_compat`](openai_compat.md) with vLLM.

## Mental model

1. **Decoding is memory-bandwidth bound.** tok/s ≈ bandwidth ÷ bytes of weights read per token. The module docstring's estimate: a 7B model in bf16 is about 15 GB, and an RTX 3090 has about 936 GB/s. That gives a ceiling near 60 tok/s and about 30–45 tok/s in practice. 4-bit reads fewer bytes, so it raises the ceiling as well as saving VRAM.
2. **The prompt format is the checkpoint's.** `apply_chat_template` reads it from the tokenizer. The plain-text fallback is right for base models and wrong for chat models whose tokenizer lacks a template.
3. **Knobs Kumaru does not pass come from the checkpoint's `generation_config.json`.** For example, `Qwen/Qwen2.5-0.5B-Instruct` ships `repetition_penalty: 1.1`, `top_k: 20` and `top_p: 0.8`. With `temperature: 0`, Kumaru passes only `do_sample=False`, so the checkpoint's `repetition_penalty` still applies to greedy decoding.
4. **Thinking is a template variable.**
   - `thinking: true` or `false` renders the template with `enable_thinking`. Qwen3's template answers `false` by writing an empty `<think></think>` block into the prompt.
   - `None` keeps the template default.
   - Templates without the variable, such as Qwen2.5's, ignore it.
   - There is no thinking budget.
5. **One `generate` call per request, with no batching.** Concurrent requests run concurrent `generate` threads on the same weights. Each has its own KV cache, and they share GPU time. Hence `local-gpu` admits 4 requests and `low-vram` admits 2.
6. **Ending the stream halts the GPU work a few tokens later.**
   - The producer's `finally` sets `halt` and waits up to 1 s (`worker.join(timeout=1.0)`) for `generate` to return. It runs when `generate` finishes, when a stop sequence matches, and when the reader goes away; for the last case, `iterate_in_thread` closes the producer on its worker thread.
   - `HaltWhenAsked` then ends `generate` about one decoding step later.
   - `stream()` wraps `iterate_in_thread(produce)` in `contextlib.aclosing`, so closing `stream()` closes the producer at once instead of at garbage collection.
   - A departed reader is noticed only when the next chunk is handed over, and the streamer releases text at word boundaries. A long prefill therefore runs to the first token before anything can stop it.
   - Backpressure stops at the streamer. `generate` pushes text into `TextIteratorStreamer`'s unbounded queue, so a slow reader does not slow the GPU; only `halt` stops it.
7. **Text arrives in words; usage counts tokens.** `TextIteratorStreamer` releases text at spaces, newlines and CJK characters, so one chunk can carry several tokens. With the Qwen2.5 tokenizer, 32 tokens of English prose arrive as 22 pieces. The counts do not depend on the pieces:
   - `CountingStreamer`, a `TextIteratorStreamer` subclass, counts the token ids `generate` passes to `put()`. It skips the first call, which carries the prompt. `completion_tokens` is that count: the end-of-sequence token is included, and so are the tokens produced after a stop sequence matched and before `halt` took effect.
   - `finish_reason` is `"length"` when the count reached `max_new_tokens`, the last token is not the tokenizer's EOS, and no stop sequence matched. Otherwise it is `"stop"`, which covers a model that ends exactly on its last allowed token.
   - `prompt_tokens` is exact.
8. **Loading blocks the event loop.** Both `from_pretrained` calls run synchronously, first-run download included. `kumaru serve` answers nothing, `/healthz` included, until they finish.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `TransformersBackend` | `class TransformersBackend(config: BackendConfig)` | Registered as `transformers` and `hf`, by decorator and lazily in `src\kumaru\core\registry.py`. Construction is cheap; nothing is imported until `load()`. |
| `TransformersBackend.name` | `name = "transformers"` | Registry and introspection name. |
| `TransformersBackend.supports_adapters` | `supports_adapters = True` | Declares that `_load()` attaches `adapter_path`. It is the only backend that sets it; `Backend.load()` refuses an adapter on every other backend. |
| `TransformersBackend.model`, `.tokenizer` | `model: Any = None`, `tokenizer: Any = None` (set in `__init__`) | The loaded `PreTrainedModel` (or `PeftModel`) and tokenizer; `None` before load and after `aclose()`. |
| `TransformersBackend._load` | `async def _load(self) -> None` | The load sequence above. Raises `DependencyMissingError` (`transformers` → extra `torch`; `bitsandbytes` → extra `quant`; `peft` → extra `train`) or `BackendError` (`Unknown dtype`, `Adapter not found`). |
| `TransformersBackend.aclose` | `async def aclose(self) -> None` | Drops the model and tokenizer, runs `gc.collect()`, and empties the CUDA cache. |
| `TransformersBackend.info` | `def info(self) -> BackendInfo` | `model_id` = `backend.model`; `device` and `dtype` as resolved (`"cpu"` and `"unknown"` before load); `context_length` after the clamp; `quantization` = `"4bit-nf4"`, `"8bit"` or `None`; `extra` = `{"parameters": ..., "adapter": adapter_path}`. |
| `TransformersBackend.count_tokens` | `def count_tokens(self, messages: Sequence[Message]) -> int` | Exact token count of the rendered prompt (rendered without the thinking flag). The base heuristic before load. |
| `TransformersBackend._render_prompt` | `def _render_prompt(self, messages: Sequence[Message], thinking: bool \| None = None) -> str` | The exact prompt string. Use it to debug prompt format. |
| `TransformersBackend.stream` | `async def stream(self, messages: Sequence[Message], params: GenerationParams) -> AsyncIterator[StreamChunk]` | Loads if needed, then streams reasoning and answer chunks, then one final chunk with `finish_reason` and `usage`. |

`load()`, `generate()` and `is_loaded` are inherited from `Backend`; see [`base.md`](base.md).

### Parameter mapping

| `GenerationParams` field | Passed to `generate` |
|---|---|
| `max_tokens` | `max_new_tokens = min(max_tokens, context - prompt_tokens)`. |
| `temperature` | `> 0`: `do_sample=True` plus `temperature`, `top_p` and `repetition_penalty`. `0`: only `do_sample=False` (greedy). |
| `top_k` | When sampling and `> 0`. Otherwise the checkpoint's value applies. |
| `min_p` | When sampling and non-zero. |
| `stop` | Not passed. `StopSequenceTracker` cuts the stream, and the producer's `finally` then halts `generate` (see the mental model). |
| `seed` | `torch.manual_seed(seed)`, which is process-global. |
| `thinking` | `enable_thinking` template variable, when not `None`. |
| `presence_penalty`, `thinking_budget` | Ignored: transformers has no equivalent. `llamacpp_server` honours both. |

## How to use it

### Install

```powershell
.\scripts\setup.ps1 -Gpu           # PyTorch CUDA wheel (-CudaVersion, default cu126) + ".[train]": transformers, accelerate, peft
.\scripts\setup.ps1 -Gpu -Quant    # + bitsandbytes, for load_in_4bit / load_in_8bit
```

### Run a profile

```powershell
kumaru serve -c local-gpu    # Qwen2.5-7B-Instruct, bf16, 16k context: 24 GB GPUs
kumaru serve -c low-vram     # the same model in 4-bit NF4, 8k context: 6-12 GB GPUs
kumaru chat -c local-gpu --max-tokens 512 -q "Summarize the backend options."
```

The first run downloads the checkpoint into the Hugging Face cache; `setup.ps1`'s closing hint puts `local-gpu`'s first run at about 15 GB. The profiles' VRAM budgets are on [`local-gpu.yaml.md`](../../../configs/local-gpu.yaml.md) and [`low-vram.yaml.md`](../../../configs/low-vram.yaml.md).

### Use a thinking model

```powershell
kumaru chat -c local-gpu -m Qwen/Qwen3-8B --think -q "Is 2^31-1 prime?"
kumaru chat -c local-gpu -m Qwen/Qwen3-8B --no-think -q "Is 2^31-1 prime?"
```

- **Thinking on.** `--think` sets `thinking: true`, which renders `enable_thinking=True` and selects the agent's `thinking_mode` sampling preset.
- **Per request.** Over HTTP, send `"thinking": false`, `"chat_template_kwargs": {"enable_thinking": false}` or `"reasoning_effort": "none"` in the `/v1/chat/completions` body.
- **No budget.** `--thinking-budget` has no effect on this backend, so give thinking models a generous `--max-tokens`.

### Serve a LoRA adapter

```powershell
kumaru train configs\train\lora-qwen.yaml                # writes runs\lora-qwen\adapter
kumaru chat -c local-gpu --adapter runs\lora-qwen\adapter -q "Use the fine-tuned behaviour."
```

```yaml
backend:
  name: transformers
  model: Qwen/Qwen2.5-7B-Instruct     # the base the adapter was trained on
  adapter_path: runs\lora-qwen\adapter
  extra:
    merge_adapter: true               # fold into the weights: no per-layer LoRA overhead
```

- **Adapters are transformers-only.** Every other backend's `load()` refuses `adapter_path` with `ConfigError`: `The llamacpp_server backend cannot attach a LoRA adapter (backend.adapter_path is '…').` Ignoring it would quietly serve, or benchmark, the base model under the adapter's name.
- **Path rules.** A relative path resolves against the working directory, and `~` is not expanded.
- **Merging is safe here.** A running server switches adapters by reloading the whole backend: `POST /v1/backend` with `{"adapter_path": "runs\\lora-qwen\\adapter"}`, plus `Authorization: Bearer <api_key>` if `server.api_key` is set. So merging gives up no runtime flexibility.
- **Merging into a 4-bit base is lossy.** peft re-rounds the merged weights to 4-bit and warns that generations may differ.

### Inspect the exact prompt without loading weights

```python
from transformers import AutoTokenizer

from kumaru.backends.transformers_backend import TransformersBackend
from kumaru.core.config import load_config
from kumaru.core.types import Message

cfg = load_config("local-gpu")
backend = TransformersBackend(cfg.backend)
backend.tokenizer = AutoTokenizer.from_pretrained(cfg.backend.model)  # tokenizer files only
messages = [Message.system("Be brief."), Message.user("hello")]
print(backend._render_prompt(messages, thinking=False))
print(backend.count_tokens(messages), "prompt tokens")
```

### Generate from Python

```python
import asyncio

from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS
from kumaru.core.types import Message


async def main() -> None:
    cfg = load_config("local-gpu")
    backend = BACKENDS.create(cfg.backend.name, cfg.backend)
    await backend.load()
    try:
        result = await backend.generate([Message.user("hello")], cfg.generation)
        print(result.message.content, result.finish_reason, result.usage)
        print(backend.info())
    finally:
        await backend.aclose()


asyncio.run(main())
```

## Tuning

| Knob | Effect | Guidance |
|---|---|---|
| `backend.dtype` | Weight and compute format. | bf16 on Ampere and newer: it has fp32's exponent range, so no fp16 overflow. CPU defaults to fp32. |
| `backend.load_in_4bit` | bitsandbytes NF4 with double quantization; compute in `dtype`. | `low-vram`'s comment: 7B drops from about 15 GB to about 4.5 GB, at a small quality cost. If both flags are set, 4-bit wins. |
| `backend.load_in_8bit` | bitsandbytes int8. | Halves the bf16 footprint, with less quality loss than 4-bit. |
| `backend.max_context_length` | Prompt plus output limit, and the KV cache ceiling. | Qwen2.5-7B: 56 KiB/token, so 16k ≈ 0.9 GB (`local-gpu`'s budget). It is clamped to the checkpoint's limit at load. |
| `backend.attn_implementation` | Forwarded to `from_pretrained`. | `local-gpu` notes `flash_attention_2` after `pip install flash-attn --no-build-isolation`. |
| `backend.extra.device_map` | accelerate placement on CUDA; default `"auto"`. | `"auto"` spreads over all visible GPUs and spills layers to CPU RAM when VRAM runs short (slow, but no OOM). `"cuda:0"` pins one GPU and fails fast. `device: cuda:1` alone does not pin a GPU; set `device_map` or `CUDA_VISIBLE_DEVICES`. |
| `backend.trust_remote_code` | Lets the checkpoint run its own Python code. | Only for repos you trust; some new architectures need it. |
| `server.max_concurrent_requests` | Concurrent `generate` threads. | Default 8. More threads add latency, not throughput; for real concurrency use vLLM or `llamacpp_server`. |

## How to extend it

- **Backend-specific options** go under `backend.extra`, read with `cfg.extra.get(...)`, as `device_map` and `merge_adapter` already are.
- **Optional imports.** Import optional packages inside the method that needs them, and raise `DependencyMissingError(package, extra=...)` when they are absent. Keep `__init__` free of side effects.
- **New reasons to stop early** need no new plumbing. `break` out of the streamer loop in `produce()`, and the `finally` sets `halt`. Alternatively, append another `StoppingCriteria` to the same `StoppingCriteriaList`. Keep `halt.set()` in the `finally`: it is the only thing that stops `generate` when the reader goes away.
- **A thinking budget** would be a `LogitsProcessor` that forces `</think>` after N reasoning tokens. `llamacpp_server` already gets this from llama.cpp.
- **Keep prompt rendering in `_render_prompt`**, so that `count_tokens()` and `stream()` see the same string.

## Failure modes

Under `kumaru serve`, load errors show in `/healthz` (`status: "error"`) and as HTTP 503 `Backend failed to load: <message>`.

| Symptom | Likely cause | Fix |
|---|---|---|
| `Optional dependency 'transformers' is not installed.` | The ML stack is missing. | `.\scripts\setup.ps1 -Gpu` |
| `Optional dependency 'bitsandbytes' is not installed.` | `load_in_4bit` or `load_in_8bit` without the `quant` extra. Kumaru checks before the weights load. | `.\scripts\setup.ps1 -Quant`, or `pip install -e ".[quant]"` in the Kumaru folder (the hint prints this). |
| `Optional dependency 'peft' is not installed.` | `adapter_path` is set without the `train` extra. | `.\scripts\setup.ps1 -Gpu` |
| `Adapter not found: <path>`, hint ``Train one with `kumaru train configs/train/lora-qwen.yaml` `` | Wrong path: it is relative to the working directory, and `~` is not expanded. Or the adapter was never trained. | Point at `<output_dir>\adapter`, e.g. `runs\lora-qwen\adapter`, or train one with the hint's command. |
| Size-mismatch error while loading the adapter | The adapter was trained on a different base. | Set `backend.model` to `base_model_name_or_path` from the adapter's `adapter_config.json`. |
| `Unknown dtype '<x>'` | Unsupported name. | `auto`, `bfloat16`, `bf16`, `float16`, `fp16`, `float32` or `fp32`. |
| CUDA out of memory at load or on long prompts | Weights, KV cache and activations exceed VRAM. | `low-vram` (4-bit), a smaller `max_context_length`, or `llamacpp_server` with a GGUF. |
| Loads fine, but tok/s is a fraction of expectations | `device_map: "auto"` spilled layers to CPU RAM, or Windows (WDDM) spilled VRAM into shared system memory. | Check `backend.model.hf_device_map` and Task Manager's *Shared GPU memory*. Shrink the model or context; `extra.device_map: "cuda:0"` turns silent spill into a load error. |
| NaN or inf errors, or garbage output, in fp16 | fp16 overflow. | `dtype: bfloat16`, or `auto` on Ampere and newer. |
| `Prompt is <n> tokens but the context window is <m>.` | The prompt alone fills the window, possibly after the clamp. The agent budgets history with this backend's exact count, so the cause is usually one oversized message (`sliding_window` always keeps the two newest messages) or a very long system prompt. | Shorten the message, or raise `backend.max_context_length` up to the checkpoint's limit. |
| Replies cut short on long conversations | `max_tokens` is more than half the window. The agent reserves at most half the window for the reply and trims `max_tokens` to the room the prompt leaves; this backend caps `max_new_tokens` the same way. | Raise `backend.max_context_length`, or lower `agent.max_history_tokens`. |
| The reply runs on past the end of the turn | `eos_token_id` is the tokenizer's single `eos_token`, which overrides a multi-token EOS list in `generation_config.json`. | Use a checkpoint whose tokenizer `eos_token` is its end-of-turn token, as Qwen2.5-Instruct's is, or serve the model with `llamacpp_server`. A stop sequence cannot catch the marker: `skip_special_tokens=True` strips special tokens before `StopSequenceTracker` sees the text. |
| Thinking shows up in the answer | The model registers `<think>` and `</think>` as special tokens, so `skip_special_tokens=True` strips them before the splitter sees them. Qwen3's are not special. | Use `llamacpp_server`, which parses reasoning server-side. |
| `--thinking-budget` or `presence_penalty` changes nothing | Not supported by this backend. | `llamacpp_server` |
| `--no-think` changes nothing | The template has no `enable_thinking` variable (e.g. Qwen2.5, which does not think anyway). | — |
| The GPU stays busy for a while after Stop or a disconnect | A long prompt is still being prefilled. Kumaru notices the departed reader only when the first chunk of output is handed over, and only then sets `halt`; after that, `generate` stops about one decoding step later. | Expected. Shorter history (`agent.max_history_tokens`) shortens the prefill. |
| `kumaru serve` does not respond at all, even `/healthz`, during startup | The first-run download and load run synchronously on the event loop. | Wait; pre-download with `kumaru chat -c <profile> -q hi` once. |
| `ValueError` asking for `trust_remote_code=True` | The checkpoint ships custom modelling code. | `backend.trust_remote_code: true`, only for repos you trust. |

## Known limitations

- **`device: cuda:N` does not pin a GPU.** On CUDA the model loads with `device_map="auto"`, which spreads it over every visible GPU. Workaround: `$env:CUDA_VISIBLE_DEVICES='1'` before starting Kumaru, or `backend.extra.device_map: "cuda:1"`.
- **Only the tokenizer's EOS ends generation.** `generate()` receives `eos_token_id=tokenizer.eos_token_id`, which replaces the EOS list in the checkpoint's `generation_config.json` (Qwen2.5-0.5B-Instruct's lists `<|im_end|>` and `<|endoftext|>`; its tokenizer names only `<|im_end|>`). A checkpoint whose tokenizer EOS is not its end-of-turn token runs on into the next turn. Workaround: serve such a model with `llamacpp_server`.

## Related files

- `src\kumaru\backends\base.py` — `Backend`, `iterate_in_thread`, `StopSequenceTracker`, `ReasoningSplitter`; see [`base.md`](base.md).
- `src\kumaru\core\config.py` — `BackendConfig`: `device`, `dtype`, `load_in_4bit`, `load_in_8bit`, `attn_implementation`, `trust_remote_code`, `adapter_path` and `extra`.
- `src\kumaru\core\types.py` — `Message`, `GenerationParams`, `StreamChunk`, `Usage`, `BackendInfo`.
- `src\kumaru\core\registry.py` — lazy registration as `transformers` and `hf`.
- `src\kumaru\train\finetune_lora.py` — writes `<output_dir>\adapter`, which `adapter_path` loads.
- `configs\local-gpu.yaml`, `configs\low-vram.yaml`, `configs\default.yaml` — profiles for this backend; see [`local-gpu.yaml.md`](../../../configs/local-gpu.yaml.md), [`low-vram.yaml.md`](../../../configs/low-vram.yaml.md) and [`default.yaml.md`](../../../configs/default.yaml.md).
- `configs\train\lora-qwen.yaml` — the QLoRA recipe whose adapter this backend serves.
- `src\kumaru\backends\llamacpp_server.py` — GGUF, thinking budgets and continuous batching; see [`llamacpp_server.md`](llamacpp_server.md).
- `src\kumaru\backends\openai_compat.py` — vLLM and remote servers; see [`openai_compat.md`](openai_compat.md).
- `scripts\setup.ps1` — `-Gpu` and `-Quant`; see [`setup.ps1.md`](../../../scripts/setup.ps1.md).
