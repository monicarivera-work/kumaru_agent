# `src\kumaru\backends\llamacpp_backend.py`

> Runs a GGUF model inside the Kumaru process through llama-cpp-python: the CPU-only path, with no child process, port or separate binary.

**Read this when:**
- you run `configs\cpu-only.yaml`, or any config with `backend.name: llamacpp` (alias `gguf`);
- you are choosing between this backend and [`llamacpp_server`](llamacpp_server.md);
- a GGUF will not download or load, runs on the wrong device, or leaks `<think>` into answers.

---

## What it does

`LlamaCppBackend` wraps one `llama_cpp.Llama` object. It is registered as `llamacpp`, alias `gguf`.

`load()` takes `path = os.path.expanduser(backend.model_path or backend.model)` and resolves it in this order:

1. **`path` exists on disk:** `Llama(model_path=path, ...)`.
2. **Otherwise, `path` contains `/`:** it is a Hugging Face repo id. `Llama.from_pretrained(repo_id=path, filename=backend.extra["gguf_filename"], ...)` downloads the one matching file. The default pattern is `"*Q4_K_M.gguf"`. The file goes into the Hugging Face cache, not `backend.models_dir`.
3. **Otherwise:** `BackendError("GGUF model not found: <path>")`.

Both constructors run on a worker thread (`asyncio.to_thread`) and receive `n_ctx=backend.max_context_length`, `n_gpu_layers=backend.n_gpu_layers` and `verbose=False`. `n_threads` is added only when `backend.n_threads` is set and non-zero.

`stream()` loads on first use if needed. It runs `Llama.create_chat_completion(..., stream=True)` on a worker thread through `iterate_in_thread`, and holds the backend's `_generate_lock` for the whole generation. llama-cpp-python renders the prompt with the chat template embedded in the GGUF, and Kumaru post-processes the raw text:
- `ReasoningSplitter` routes a leading `<think>…</think>` block to `StreamChunk.reasoning`, and the rest to `StreamChunk.delta`;
- `StopSequenceTracker` enforces `stop` on the answer text;
- the last chunk carries `Usage` and a `finish_reason` normalised by `normalize_finish_reason` from `openai_compat.py`.

### What changed in v0.2

| v0.2 | v0.1 |
|---|---|
| A leading `<think>…</think>` block streams on the reasoning channel. | Thinking streamed as answer text. |
| `min_p` and `presence_penalty` are forwarded when set. | Not sent. |
| `finish_reason` goes through `normalize_finish_reason`. | Raw engine value. |
| Generations take turns on a lock, so parallel requests are safe. | No lock: parallel requests used one `Llama` at the same time. |
| A client disconnect or the UI's Stop ends the generation at the next token. | The worker kept generating until 256 chunks were queued, then blocked forever. A stop-sequence hit already ended it. |
| The download and load run on a worker thread; `/healthz` answers `loading` meanwhile. `~` in the path is expanded. | Both ran on the event loop, so the server answered nothing until they finished. `~` was not expanded. |
| `thinking` and `thinking_budget` are still **not** sent. | Same. |

The last row is a limit of the binding. `create_chat_completion` has no parameter for chat-template variables, such as Qwen's `enable_thinking`, and no reasoning budget. The model thinks exactly when its template's default says so.

## Why it exists

- **GGUF with nothing but pip.** One Python package brings GGUF quantisations (Q4_K_M, Q5_K_M, Q8_0) and CPU/GPU layer splitting. There is no binary to fetch and no port to manage.
- **CPU-only machines.** `configs\cpu-only.yaml` uses this backend. Its comments note that llama.cpp's AVX2/AVX-512 kernels are several times faster than PyTorch on CPU.

Since v0.2, the GPU profiles that use GGUF (`qwen36`, `qwen25-7b-gguf`) run on [`llamacpp_server`](llamacpp_server.md) instead. The reasons are in [When to use `llamacpp_server` instead](#when-to-use-llamacpp_server-instead).

## Mental model

1. **One `Llama` object: the weights plus one KV cache of `n_ctx` tokens, allocated at load.**
   - It decodes one sequence at a time, and a llama.cpp context is not thread-safe. Generations therefore take turns on a `threading.Lock` (`_generate_lock`).
   - The worker thread holds the lock for the whole generation, including closing the engine's stream, so the next request starts clean. A reader that stalls without closing the stream (a client that stops reading) keeps the lock once the 256-chunk queue is full.
   - Requests admitted beyond the first wait on the lock, each with a `kumaru-generate` thread blocked in `acquire`. `server.max_concurrent_requests: 1`, as `cpu-only` sets, keeps them in the server's queue instead; the default is 8.
2. **Loading runs on a worker thread.** `_load()` hands the `Llama` constructor, or `Llama.from_pretrained`, to `asyncio.to_thread`. While a GGUF downloads or loads, `kumaru serve` keeps answering: `/healthz` reports `loading`, and chat requests wait for the load (up to 600 s, then `Backend is still loading.`).
3. **`n_gpu_layers` has llama-cpp-python's meaning.**
   - `-1`, the `BackendConfig` default, puts *all* layers on the GPU. It does not mean "as many as fit": when they do not fit, loading fails with an out-of-memory error, or, on Windows, may spill into shared system memory and run slowly.
   - `0` is CPU only.
   - `N` puts N layers on the GPU and the rest on the CPU.

   Decoding is memory-bandwidth bound, so each layer moved into VRAM speeds it up, until VRAM runs out. Only `llamacpp_server` fits layers to free VRAM automatically.
4. **The prompt format comes from the GGUF's `tokenizer.chat_template` metadata.** If the file has none, llama-cpp-python silently falls back to the Llama-2 format.
5. **Thinking follows the template's default.** Qwen3-style templates think unless the caller passes `enable_thinking=false`, and this backend cannot pass it. `GenerationParams.thinking` only selects the agent's `thinking_mode` sampling preset; it never reaches the engine.
6. **The reasoning split expects the output to open with `<think>`.** `ReasoningSplitter()` runs with `start_in_reasoning=False`. If a template pre-opens `<think>` in the prompt, the output starts mid-thought. The whole thought, followed by a stray `</think>`, then lands in the answer.
7. **`stop` is matched twice.** Kumaru's tracker sees only answer text. The same list also goes to llama-cpp-python, which matches it on the raw output, thinking included.
8. **Usage is approximate.** llama-cpp-python's streamed chat chunks carry no `usage`.
   - `completion_tokens` counts non-empty deltas: about one per token, thinking included.
   - `prompt_tokens` is `count_tokens(messages)`: the model's tokenizer run over `role: content` lines, plus 8. The chat template is not rendered.
9. **Closing the stream stops the generation at the next token.**
   - A stop-sequence hit, a client disconnect or the UI's Stop closes llama-cpp-python's stream in a `finally` on the worker thread.
   - The stream is a lazy generator: llama.cpp computes tokens only while it is asked for the next event. Closing it therefore ends the computation, and the lock is released.
   - Chunks a slow reader had not taken yet were computed and are discarded: at most a full queue of 256 plus the one being handed over.
   - A request cancelled while it waited for the lock still evaluates its prompt when its turn comes. It decodes until it has a first chunk to hand over (a token or a few), then stops.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `LlamaCppBackend` | `class LlamaCppBackend(config: BackendConfig)` | GGUF backend, registered as `llamacpp` and `gguf`, both by decorator and lazily in `src\kumaru\core\registry.py`. Construction is cheap. |
| `LlamaCppBackend.name` | `name = "llamacpp"` | Registry and introspection name. |
| `LlamaCppBackend.llm` | `llm: Any = None` (set in `__init__`) | The `llama_cpp.Llama` instance after load; `None` before load and after `aclose()`. |
| `LlamaCppBackend._generate_lock` | `threading.Lock` (set in `__init__`) | Serialises generations on the one `Llama`. Taken on the worker thread, never on the event loop. Held for all of `produce()`: from before `create_chat_completion` until the final chunk is queued, or until an abandoned stream has been closed. |
| `LlamaCppBackend._load` | `async def _load(self) -> None` | Applies the resolution order above, with the constructor on a worker thread. Raises `DependencyMissingError("llama-cpp-python", extra="gguf")` or `BackendError`. llama-cpp-python's own exceptions pass through unwrapped. |
| `LlamaCppBackend.aclose` | `async def aclose(self) -> None` | Sets `llm = None` and marks the backend unloaded. It does not call `Llama.close()`; memory is freed when the object is garbage-collected. A generation that is still running keeps its own reference until it ends. |
| `LlamaCppBackend.info` | `def info(self) -> BackendInfo` | `model_id` = `model_path or model`; `device` = `"cuda+cpu"` if `n_gpu_layers != 0`, else `"cpu"`; `dtype` = `quantization` = `"gguf"`; `context_length` = `max_context_length`; `extra` = `{"n_gpu_layers": ...}`. Derived from config, not from the loaded build. |
| `LlamaCppBackend.count_tokens` | `def count_tokens(self, messages: Sequence[Message]) -> int` | Before load, the base heuristic. After load, `len(llm.tokenize(...)) + 8` over `role: content` lines. Takes no lock, on purpose: the agent calls it synchronously on the event loop, where waiting for a running generation would stall the server, and tokenizing only reads the vocabulary. |
| `LlamaCppBackend.stream` | `async def stream(self, messages: Sequence[Message], params: GenerationParams, *, tools: Sequence[dict[str, Any]] \| None = None) -> AsyncIterator[StreamChunk]` | Loads if needed, waits for `_generate_lock`, streams reasoning and answer chunks, then one final chunk with `finish_reason` and `usage`. Non-empty `tools` raise `ConfigError` (`check_tools`): this backend cannot call tools. |

`load()`, `generate()` and `is_loaded` are inherited from `Backend`; see [`base.md`](base.md).

### Parameter mapping

`GenerationParams` field → `create_chat_completion` keyword:

| Field (default) | Sent as | Rule |
|---|---|---|
| `max_tokens` (1024) | `max_tokens` | Always. |
| `temperature` (0.7) | `temperature` | Always. `0` selects greedy decoding. |
| `top_p` (0.95) | `top_p` | Always. |
| `top_k` (50) | `top_k` | `top_k or 0`. `0` disables top-k. |
| `repetition_penalty` (1.05) | `repeat_penalty` | Always. |
| `min_p` (`None`) | `min_p` | Only when set. Otherwise llama-cpp-python's default `0.05` applies; set `0.0` to turn min-p off. |
| `presence_penalty` (`None`) | `presence_penalty` | Only when set. llama-cpp-python's default is `0.0`. |
| `stop` (`[]`) | `stop` | The list, or `None` when empty. `StopSequenceTracker` also enforces it on the answer. |
| `seed` (`None`) | `seed` | `None` → `-1`, a random seed. |
| `thinking`, `thinking_budget` | — | Never sent. |
| — | `stream=True` | Fixed. |

## How to use it

### Install, and check which build you got

```powershell
.\scripts\setup.ps1 -Gguf                                   # pip install -e ".[gguf]"
.\.venv\Scripts\python.exe -c "import llama_cpp; print(llama_cpp.__version__, llama_cpp.llama_supports_gpu_offload())"
```

- **What the extra installs.** The `gguf` extra is `llama-cpp-python>=0.2.90`. PyPI ships current releases as source only, so pip compiles llama.cpp during the install. That needs a C/C++ compiler; on Windows, Visual Studio Build Tools.
- **The default build is CPU-only.** For it, the check prints `False`.
- **Repo-id downloads need nothing extra.** `Llama.from_pretrained` needs `huggingface_hub`, which is a core dependency of Kumaru.

For a CUDA build you also need the CUDA Toolkit:

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
.\.venv\Scripts\python.exe -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

Prebuilt CUDA wheels exist at `https://abetlen.github.io/llama-cpp-python/whl/cu121` through `.../whl/cu124`. For Windows and Python 3.12, that index stops at 0.3.4 (December 2024; checked October 2026), which predates many current architectures.

### Run the CPU-only profile

```powershell
kumaru config -c cpu-only                               # resolved settings; loads nothing
kumaru chat -c cpu-only -q "Give one CPU inference tip."
kumaru serve -c cpu-only
```

The first run downloads `qwen2.5-1.5b-instruct-q4_k_m.gguf` (1.12 GB) from `Qwen/Qwen2.5-1.5B-Instruct-GGUF`. The profile selects it with `extra.gguf_filename: "*q4_k_m.gguf"`, lower-case like the file name. `fnmatch` is case-insensitive on Windows only, so the code default `*Q4_K_M.gguf` would not match this file on Linux or macOS.

### Load a local GGUF file

```yaml
# configs\my-gguf.yaml
backend:
  name: llamacpp
  model: qwen2.5-7b-instruct-q4_k_m     # label shown by /healthz; model_path wins
  model_path: C:\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf
  n_gpu_layers: -1                      # all layers: only if weights + KV cache fit
  max_context_length: 8192

server:
  max_concurrent_requests: 1
```

```powershell
kumaru serve -c configs\my-gguf.yaml
```

- **Use an absolute path.** A relative path resolves against the working directory, and a missing path that contains `/` is treated as a repo id. `~` is expanded.
- **Split GGUFs** (`<name>-00001-of-0000N.gguf`): point `model_path` at the first shard, with every shard in the same folder; llama.cpp loads the rest. The `qwen25-7b-gguf` profile (on `llamacpp_server`) leaves exactly such a pair in `<models_dir>\Qwen\Qwen2.5-7B-Instruct-GGUF\`.

### Download from a Hugging Face repo

```yaml
backend:
  name: llamacpp
  model: bartowski/Qwen2.5-7B-Instruct-GGUF
  max_context_length: 8192
  extra:
    gguf_filename: Qwen2.5-7B-Instruct-Q4_K_M.gguf   # exact name: one match on every OS
```

- **How the file is chosen.** llama-cpp-python lists the repo recursively and needs exactly one `fnmatch` match.
- **Split repos do not work.** `Qwen/Qwen2.5-7B-Instruct-GGUF` ships Q4_K_M as two shards, and it is the repo this backend's own error hint suggests. `from_pretrained` downloads a single file, so no pattern gets both shards. Use `model_path` for split models.
- **Where the file goes.** Downloads go to `%USERPROFILE%\.cache\huggingface\hub`, unless `HF_HOME` is set. Set `HF_TOKEN` for gated repos or to avoid anonymous rate limits.

### Use it from Python

```python
import asyncio

from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS
from kumaru.core.types import Message


async def main() -> None:
    cfg = load_config("cpu-only")
    backend = BACKENDS.create(cfg.backend.name, cfg.backend)
    await backend.load()  # first run downloads the GGUF, on a worker thread
    try:
        async for chunk in backend.stream([Message.user("Name one GGUF quant type.")], cfg.generation):
            print(chunk.delta, end="", flush=True)
            if chunk.finish_reason:
                print(f"\n[{chunk.finish_reason}] {chunk.usage}")
        print(backend.info())
    finally:
        await backend.aclose()


asyncio.run(main())
```

## Tuning

| Knob | Effect | Guidance |
|---|---|---|
| `backend.n_gpu_layers` | Layers on the GPU: `-1` all, `0` none. | On a GPU, keep `-1` only when the weights and KV cache fit. Otherwise use a positive count, and raise it until the load fails or Task Manager's *Shared GPU memory* starts to grow. |
| `backend.max_context_length` → `n_ctx` | KV cache size, allocated at load. | Qwen2.5-7B costs 56 KiB/token: 448 MiB at the default 8,192 tokens, 1.75 GiB at 32,768. |
| The GGUF file | Weight size and quality. | Qwen2.5-7B Q4_K_M is a 4.7 GB file (4.4 GiB). With an 8k context it fits an 8 GB GPU. |
| `backend.n_threads` | Generation threads. | `null` or `0` means llama-cpp-python's default: half the logical CPUs, which is one per physical core on SMT CPUs. Prompt processing always uses every logical CPU. |
| `server.max_concurrent_requests` | Admission limit. | `1`, as `cpu-only` sets. Higher values are safe, because requests take turns on the lock, but gain nothing. Each extra admitted request waits with a thread blocked on the lock. If its client gives up meanwhile, it still costs a prompt evaluation and a token or a few. |
| `generation.max_tokens` | Output cap. | `cpu-only` uses 512 to keep CPU latency tolerable. Thinking models need far more; this backend has no thinking budget. |

`configs\cpu-only.yaml`'s comments set the CPU expectations: roughly 3–7 tok/s for a 7B model on a modern 8-core CPU, and 15–30 tok/s for 1.5B. These are not measured here.

## When to use `llamacpp_server` instead

Both backends read the same GGUF files. Switching is a config change: set `backend.name: llamacpp_server`, plus `backend.model: org/repo:QUANT` or `backend.model_path`.

| Concern | `llamacpp` (this file) | [`llamacpp_server`](llamacpp_server.md) |
|---|---|---|
| New architectures (e.g. Qwen3.6's Gated DeltaNet) | Wait for a llama-cpp-python release that bundles them, then rebuild the wheel. | Install a newer binary: `.\scripts\get-llama-server.ps1`. |
| Thinking on/off, thinking budget | Template default only. | Per request: `chat_template_kwargs.enable_thinking`, `reasoning_budget_tokens`. |
| Reasoning split | Client-side `ReasoningSplitter`; needs `<think>` at the start of the output. | Server-side; arrives in `reasoning_content`. |
| Concurrency | One generation at a time; requests take turns on a lock. | `parallel` slots (default 4) with continuous batching. |
| GPU placement | `n_gpu_layers` only; `-1` = all layers. | `-1` lets llama.cpp fit layers to free VRAM; `n_cpu_moe`, `flash_attn` and KV cache types are also exposed. |
| Failure isolation | A crash inside llama.cpp takes down Kumaru. | Only the child process dies, and stopping it frees its VRAM. |
| Startup | Download and load on a worker thread. | Download on a worker thread, then an async health wait. |
| Client disconnect | Generation stops at the next token. | llama-server cancels the request when the connection closes. |
| Install | pip; current releases compile from source. | Prebuilt binary; no compiler. |

Keep `llamacpp` when you already have a working wheel, want a single process with no port or binary, or run the `cpu-only` profile as is.

## How to extend it

- **Expose more `Llama` constructor options** through `backend.extra`, the same way `gguf_filename` is read:

  ```python
  # in _load(), after kwargs is built
  for key in ("n_batch", "flash_attn", "chat_format"):
      if key in cfg.extra:
          kwargs[key] = cfg.extra[key]
  ```

  ```yaml
  backend:
    name: llamacpp
    model_path: C:\models\model.gguf
    extra:
      n_batch: 1024
      flash_attn: true
  ```
- **Leave prompt formatting to llama-cpp-python** and the GGUF's template. Rendering prompts in Kumaru is `transformers_backend.py`'s job.
- **Templates that pre-open `<think>`** need `ReasoningSplitter(start_in_reasoning=True)`; see [`base.md`](base.md). To detect one, render `llm.metadata["tokenizer.chat_template"]` once with `llama_cpp.llama_chat_format.Jinja2ChatFormatter`, and check whether the prompt ends in `<think>`. `transformers_backend.py` makes the same check on its own rendered prompt.
- **Honouring `thinking`** would need a custom `chat_handler` that renders the template with `enable_thinking`. `llamacpp_server` already does this per request.
- **Keep the lock discipline.** Everything that touches the model stays inside `with self._generate_lock:` in `produce()`, including closing the engine's stream in `finally`. The lock is taken on the worker thread. Taking it in `stream()` itself would block the event loop.
- **Parallel decoding** is out of reach of `create_chat_completion`, because one `Llama` decodes one sequence. Several `Llama` objects would each load the model and allocate their own KV cache. For parallel requests, use `llamacpp_server`, whose slots batch them into one forward pass.
- **Keep the stream contract.** End with `StreamChunk(finish_reason=..., usage=...)`, and run engine finish reasons through `normalize_finish_reason`.

## Failure modes

Under `kumaru serve`, load errors appear in `/healthz` (`status: "error"`, with the message in `detail`) and as HTTP 503 `Backend failed to load: <message>`. llama-cpp-python's own `ImportError` and `ValueError` are not wrapped.

| Symptom | Likely cause | Fix |
|---|---|---|
| `Optional dependency 'llama-cpp-python' is not installed.` (hint: `Install it with: .\scripts\setup.ps1 -Gguf, or pip install -e ".[gguf]" in the Kumaru folder`) | The `gguf` extra is missing. | `.\scripts\setup.ps1 -Gguf` |
| `pip` fails while building `llama-cpp-python` | Current releases are source only, and no C/C++ compiler is installed. | Install Visual Studio Build Tools (C++), or use `llamacpp_server`, which ships as a binary. |
| `ImportError: Llama.from_pretrained requires the huggingface-hub package.` | A broken environment: `huggingface_hub` is a core dependency. | `.\.venv\Scripts\python.exe -m pip install -e "."` in the Kumaru folder, or set `model_path`. |
| `ValueError: No file found in <repo> that match <pattern>` | The pattern matches nothing. Matching is case-sensitive outside Windows, and split shards never end in the quant name. | Put the exact file name in `extra.gguf_filename`. For split models, use `model_path`. |
| The same error for `Qwen/Qwen2.5-7B-Instruct` with `-b llamacpp` | `backend.model` was left at the `BackendConfig` default, a safetensors repo with no GGUF. | Pass `-m` with a GGUF repo, or set `model_path`. |
| `ValueError: Multiple files found in <repo> matching <pattern>` | The pattern is too broad, e.g. it matches every shard. | Use the exact file name. |
| `GGUF model not found: <path>` | The path does not exist and contains no `/`. The message shows the path after `~` expansion. | Check the path; use an absolute one. |
| A Hugging Face repo-id error for something that is a file path | A missing `model_path` that contains `/` is treated as a repo id. | Fix the path; prefer backslashes. |
| `ValueError: Failed to load model from file: <path>` | The bundled llama.cpp does not know the architecture, a shard is missing, or the file is corrupt. llama.cpp's own log is suppressed (`verbose=False`). | Upgrade `llama-cpp-python` or re-download. For new architectures, use `llamacpp_server`; its log is in `runs\logs\`. |
| Slow, and `info()` still says `cuda+cpu` | A CPU-only build. `device` comes from `n_gpu_layers`, not from the build. | Check `llama_supports_gpu_offload()`; rebuild with CUDA. |
| Out-of-memory at load, or tok/s collapses on Windows | `n_gpu_layers: -1` asks for all layers, and they do not fit. Windows may spill into shared system memory instead of failing. | Use a positive layer count, a smaller context or quant, or `llamacpp_server`. |
| Rambling replies or role confusion | The GGUF has no embedded chat template, so the Llama-2 format is used; or it is a base model. | Use an instruct GGUF with an embedded template. |
| The answer contains the thought and a stray `</think>` | The template pre-opens `<think>`. | Use `llamacpp_server`, or extend the splitter (see above). |
| `--no-think`, the UI toggle or the thinking budget change nothing | `thinking` and `thinking_budget` are not sent. | Use `llamacpp_server`. |
| Empty answer with `finish_reason: "length"` | Thinking used up `max_tokens`. Unterminated reasoning stays reasoning. | Raise `max_tokens`, or use `llamacpp_server` with a thinking budget. |
| Generation ends in the middle of the thought | llama-cpp-python matched a stop string inside the reasoning. | Use stop strings that cannot occur in thoughts. |
| Parallel requests are answered one after another; latency grows with load | Expected: generations take turns on `_generate_lock` (one `Llama`, one sequence). | For parallel decoding, use `llamacpp_server` (`parallel` slots). |
| `/healthz` reports `loading` for minutes at startup; a chat request fails after 600 s with `Backend is still loading.` | The first run downloads the GGUF, then loads it. | Wait, or pre-download the file and use `model_path`. |
| The CPU/GPU works briefly on a request whose client already left | The request was cancelled while it waited for the lock. When its turn comes it still evaluates its prompt and decodes until its first chunk, then stops. | `server.max_concurrent_requests: 1` keeps waiting requests in the server's queue, where giving up costs nothing. |
| `The llamacpp backend cannot attach a LoRA adapter (backend.adapter_path is '<path>').` | `--adapter` or `adapter_path` is set. LoRA adapters are supported by the transformers backend only, and `load()` refuses them elsewhere rather than serve the base model under the adapter's name. | Serve the adapter with `backend.name: transformers` and its base model. GGUF models need the adapter converted first (`docs\TRAINING.md`). |
| `The llamacpp backend cannot call tools, and tools are enabled (agent.enabled_tools, or kumaru chat --workspace).` | Tools are enabled. This backend does not pass tool declarations to llama-cpp-python or read calls out of the output, so it refuses them instead of dropping them silently. `ChatAgent` checks when it is built, before the GGUF loads. | Use `llamacpp_server` with the same GGUF: llama-server parses tool calls with the model's own template. |

## Known limitations

- **Repo-id downloads fetch exactly one file.** `extra.gguf_filename` (default `*Q4_K_M.gguf`) is matched case-sensitively outside Windows, and split shards never match a quant-name pattern, so a split GGUF cannot be loaded by repo id. Workaround: the exact file name; for split models, `model_path` at the first shard, with all shards in one folder.
- **`aclose()` does not call `Llama.close()`.** VRAM and RAM come back only when Python collects the `Llama` object, which a running generation keeps alive until it ends. If a backend swap then fails to load for lack of memory, restart `kumaru serve`.
- **Thinking settings are ignored.** `thinking`, `--think`/`--no-think`, the UI toggle and `thinking_budget` never reach the engine. Workaround: `llamacpp_server`.
- **No tool calling.** `supports_tools` is `False`, so `kumaru chat --workspace` and `agent.enabled_tools` fail with `ConfigError`. Workaround: `llamacpp_server`.
- **A cancelled request still waiting for the lock still runs.** When its turn comes it evaluates its whole prompt and decodes to its first chunk (a token or a few) before it notices. Workaround: `server.max_concurrent_requests: 1`, so waiting requests queue in the server instead.

## Related files

- `src\kumaru\backends\base.py` — `Backend`, `iterate_in_thread`, `StopSequenceTracker` and `ReasoningSplitter`; see [`base.md`](base.md).
- `src\kumaru\backends\openai_compat.py` — `normalize_finish_reason`; see [`openai_compat.md`](openai_compat.md).
- `src\kumaru\backends\llamacpp_server.py` — the GGUF engine of the v0.2 GPU profiles; see [`llamacpp_server.md`](llamacpp_server.md).
- `src\kumaru\backends\transformers_backend.py` — the bf16 path, and the only backend that attaches LoRA adapters; see [`transformers_backend.md`](transformers_backend.md).
- `src\kumaru\core\config.py` — `model_path`, `n_gpu_layers`, `n_threads`, `max_context_length` and `extra`.
- `src\kumaru\core\registry.py` — lazy registration as `llamacpp` and `gguf`.
- `configs\cpu-only.yaml` — the profile that uses this backend; see [`cpu-only.yaml.md`](../../../configs/cpu-only.yaml.md).
- `configs\qwen25-7b-gguf.yaml` — the same 7B model on `llamacpp_server`; see [`qwen25-7b-gguf.yaml.md`](../../../configs/qwen25-7b-gguf.yaml.md).
- `scripts\setup.ps1` — `-Gguf` installs the extra; see [`setup.ps1.md`](../../../scripts/setup.ps1.md).
- `pyproject.toml` — the `gguf` extra.
