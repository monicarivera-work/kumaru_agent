# `src\kumaru\backends\llamacpp_server.py`

> Kumaru's default GGUF engine. It resolves or downloads a GGUF, runs llama.cpp's `llama-server` as a managed child process, and talks to it over the OpenAI protocol.

**Read this when:**
- you run the `qwen36` or `qwen25-7b-gguf` profiles;
- you size context, VRAM or concurrency;
- you share the GPU with a game;
- you upgrade llama.cpp for a new architecture;
- you attach Kumaru to a llama-server you started yourself;
- `llama-server` will not start.

---

## What it does

`LlamaServerBackend` is registered as `llamacpp_server` (aliases `llama-server`, `llama_server`). It is an `OpenAICompatBackend` with a process attached. `load()` does five things:

1. It locates the `llama-server` executable.
2. It turns `backend.model` into a local GGUF. On first use it downloads the file from Hugging Face into `backend.models_dir`.
3. It builds the command line from `backend.*` and `backend.llama_server.*`.
4. It starts the server on a free localhost port, or on a fixed `llama_server.port` that nothing else may be using. stdout and stderr go to `runs\logs\llama-server-<port>.log`.
5. It polls `/health` until the weights are loaded, checks `/v1/models`, and reads `/props`.

After that, every request is a plain OpenAI chat-completions call through the inherited `stream()`. The call adds llama.cpp's extra fields: `top_k`, `min_p`, `repeat_penalty`, `chat_template_kwargs.enable_thinking` and a per-request thinking budget. `aclose()` stops the process, which gives its VRAM back to the OS.

If you set `backend.base_url`, the backend runs in **attach mode** instead. It uses a server you started and never starts or stops a process.

## Why it exists

- **Currency.** New architectures and kernels land in llama.cpp's C++ code weeks before the Python bindings get them; the `llamacpp` backend uses those bindings. Qwen3.6's Gated DeltaNet layers are the case that motivated this backend. Staying current means installing a newer binary with `.\scripts\get-llama-server.ps1 -Version bNNNNN`. There is no wheel to rebuild and no CUDA toolchain to install.
- **Throughput.** llama-server batches requests across `parallel` slots, so concurrent requests (from the benchmark runner, or several UI tabs) share forward passes. It also caches prompts across turns and returns thinking in a separate `reasoning_content` field.
- **Isolation.** A CUDA out-of-memory error or crash kills the child process, not the API server. The VRAM is released when the child exits, and requests then fail with the child's exit code and the end of its log.
- **One command.** `openai_compat` plus a server you start by hand has no model resolution, lifecycle management or orphan protection. With this backend, `kumaru serve -c qwen36` is the whole setup.

| Design decision | Why |
|---|---|
| Subclass of `OpenAICompatBackend` | The protocol, SSE parsing, reasoning split and stop handling already exist and are tested. This file adds only the lifecycle and llama.cpp's request fields. |
| Free port by default (`llama_server.port: null`) | It cannot collide with a second Kumaru or a leftover server, so the health check always reaches the child that was just started. |
| A fixed port is probed before the spawn (`port_in_use`) | A server already listening there, such as an earlier llama-server, would answer the health check in place of the new child, and Kumaru would serve the wrong model. |
| Kumaru dials `dial_address(host)`, not the bind address | A wildcard bind (`0.0.0.0`, `::`) listens everywhere but is not a destination; Windows refuses to connect to `0.0.0.0`. Wildcards are reached on `127.0.0.1` and `::1`. |
| A cached split model counts only when complete (`shards_complete`) | A download interrupted between parts leaves a set that llama-server cannot load. The next start fetches the missing parts instead of failing. |
| `--n-gpu-layers` is left out when `n_gpu_layers: -1` | llama.cpp's own fit logic (`--fit on`, its default) measures free VRAM at load time. It then decides the GPU offload for arguments you did not set, keeping a margin (`--fit-target`, 1024 MiB by default). Kumaru does not guess. |
| `--ctx-size` is always passed | The context window is your decision (VRAM against history length). Fit therefore moves layers and experts, never the window. |
| `--kv-unified` with `parallel: 4` | All slots share one KV pool. Without it each slot gets `ctx/parallel` tokens, which for `qwen36` is 32,768 of 131,072, and a single long chat could not use the full window. |
| Thinking budget sent per request, not as a server flag | The UI changes the budget per message without restarting the server. The budget message contains newlines, which do not survive command-line quoting. |
| Binary found before the model is resolved | A missing binary fails in milliseconds, not after a 17.73 GB download. |
| Windows Job Object or `PR_SET_PDEATHSIG`, plus `atexit` | A model server that outlives Kumaru silently holds its VRAM: ~19.7 GiB for `qwen36` at 131k. With `KILL_ON_JOB_CLOSE`, Windows kills the child even when Kumaru is ended from Task Manager. |

## Mental model

```text
load()                         Backend.load: idempotent; an asyncio.Lock ensures one spawn even if many requests arrive early
 └─ _load()
     ├─ managed (backend.base_url is null) → _spawn()
     │    1. find_llama_server(llama_server.binary)   explicit path → tools\llama.cpp\ → PATH
     │    2. resolve_model_file(config)               worker thread; complete cached set, else Hugging Face download
     │    3. fixed llama_server.port? port_in_use()   taken → BackendError; nothing is started
     │    4. port = llama_server.port or free_port(host)
     │    5. build_server_args(...)                   pure; tests assert the exact list
     │    6. Popen: stdout+stderr → log (append), stdin closed, no console window, Job Object / PDEATHSIG
     │    7. url_host(host)                           base URL; the wildcards 0.0.0.0 and :: are dialled as 127.0.0.1 and [::1]
     │    8. _wait_until_healthy(startup_timeout_s)   every 0.5 s: child exited? /health 200? deadline passed?
     ├─ OpenAICompatBackend._load()                   httpx client; GET /v1/models must succeed
     ├─ GET /props (best effort)                      n_ctx, slots, build → info()
     └─ on any exception or cancellation: stop the child, re-raise
stream()   inherited: POST /v1/chat/completions; reasoning_content → StreamChunk.reasoning, content → .delta
           on httpx.TransportError, _connection_lost(): exit code + log tail if the child has exited, else the parent's error
aclose()   close the HTTP client; managed: terminate → wait 10 s → kill → wait 5 s; close job and log
```

Invariants:

1. One backend instance owns at most one child process. Failing or cancelling `_load` stops the child, so a half-started server never keeps VRAM.
2. `startup_timeout_s` (900 s) starts **after** the download. Downloading 17.73 GB does not count against it.
3. Nothing supervises or restarts the child after startup. If it dies mid-session (out of VRAM, a driver reset, a manual kill), every request fails with `llama-server exited (code N) during the request.` plus the last lines of its log, until you restart `kumaru serve` or reload the backend with `POST /v1/backend` `{"name": "llamacpp_server"}`. A request whose connection drops while the process is still exiting gets the parent's `Lost the connection to …` instead ([openai_compat.md](openai_compat.md#failure-modes)).
4. Attach mode never manages processes. `extra.managed` is `false`, and no `pid` or `log_file` is reported.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `LlamaServerBackend` | `LlamaServerBackend(config: BackendConfig)` | The backend; `name = "llamacpp_server"`. The constructor has no side effects. `load()` (inherited) spawns or attaches. |
| `LlamaServerBackend.aclose` | `async aclose() -> None` | Closes the HTTP client. In managed mode it also stops the child, which frees its VRAM. |
| `LlamaServerBackend.info` | `info() -> BackendInfo` | Metadata, read from `/props` once loaded; see [Read `info()`](#read-info). |
| `parse_model_spec` | `parse_model_spec(spec: str) -> tuple[str, str \| None] \| None` | Turns `"org/repo:QUANT"` into `(repo, quant)`. Returns `None` for anything that looks like a path. |
| `match_gguf_files` | `match_gguf_files(files: Iterable[str], quant: str \| None) -> list[str]` | Picks the shard list for one quant from a file listing. `[]` means no match or an ambiguous match. |
| `shards_complete` | `shards_complete(files: Sequence[str]) -> bool` | `True` for one whole GGUF, or for a split set whose parts 1..N are all present with one total N. `[]`, two unsplit files or a gap give `False`. |
| `resolve_model_file` | `resolve_model_file(config: BackendConfig) -> Path` | Returns the local path of the GGUF (its first shard), downloading if needed. Blocking. |
| `find_llama_server` | `find_llama_server(binary: str \| None) -> Path` | Finds the executable: explicit path, then the bundled copy, then `PATH`. |
| `dial_address` | `dial_address(host: str) -> str` | The address to connect to for a server bound to `host`. Strips `[]`; `0.0.0.0` and `""` become `127.0.0.1`, `::` becomes `::1`; anything else is returned unchanged. |
| `url_host` | `url_host(host: str) -> str` | `dial_address(host)`, bracketed when it contains `:` (IPv6), for use in a URL: `"::"` → `"[::1]"`. |
| `free_port` | `free_port(host: str) -> int` | Asks the OS for an unused TCP port on `host`, over IPv6 when the address contains `:`. |
| `port_in_use` | `port_in_use(host: str, port: int, timeout_s: float = 0.5) -> bool` | `True` when a TCP connection to `dial_address(host)`:`port` succeeds within `timeout_s`. |
| `build_server_args` | `build_server_args(binary: Path, model_file: Path, config: BackendConfig, port: int) -> list[str]` | Builds the exact command line. Pure. |
| `EXE_NAME` | `str` | `llama-server.exe` on Windows, `llama-server` elsewhere. |
| `BUNDLED_BINARY` | `Path` | `<repo>\tools\llama.cpp\<EXE_NAME>`, where `get-llama-server.ps1` installs the binary. |

Inherited unchanged: `load`, `stream`, `generate`, `list_remote_models`, `strict_openai`, `count_tokens`, `context_length`, `is_loaded`. See [openai_compat.md](openai_compat.md) and [base.md](base.md).

## How to use it

### Run a profile

```powershell
.\scripts\get-llama-server.ps1        # once; installs tools\llama.cpp\
kumaru serve -c qwen36                # UI at http://127.0.0.1:8080
```

On the first run, the backend downloads the GGUF before llama-server starts (17.73 GB for `qwen36`). The HTTP port opens immediately and `/healthz` reports `"status": "loading"`. A chat request waits up to 600 s for the backend and then gets a 503, "Backend is still loading." To download without that limit, run `kumaru chat -c qwen36 -q "hi"` once; the CLI waits for as long as the load takes.

### Choose where the weights live

```powershell
setx KUMARU_BACKEND__MODELS_DIR D:\kumaru\models    # affects new shells only
```

The default location is `~\.cache\kumaru\models`. The cache layout is `<models_dir>\<org>\<repo>\<file>.gguf`, for example `D:\kumaru\models\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`. A cache hit makes no network calls, so a GGUF you copy into that layout is used offline. huggingface_hub keeps its metadata and partial `.incomplete` files under `<repo folder>\.cache\huggingface\`. The cache scan only matches `*.gguf`, so it never picks up a partial file, and an interrupted download resumes on the next start.

A split model is a cache hit only when `shards_complete` confirms every part. If one is missing, Kumaru logs `cached GGUF <path> is missing parts; fetching them` (WARNING) and runs the download path: parts already on disk are checked against the Hub and kept, and only the missing ones are downloaded. That needs the network.

### Model specs

| `backend.model` / `backend.model_path` | Resolution |
|---|---|
| `backend.model_path` set | Takes priority over `backend.model`. The file must exist, otherwise: `backend.model_path does not exist: …`. |
| An existing file, e.g. `C:\models\foo.gguf` | Used as is. A relative path resolves against the current directory. |
| `org/repo:QUANT`, e.g. `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS` | Looked up under `models_dir\org\repo\` (recursively), otherwise downloaded. |
| `org/repo` | Same as above, but only if the repo has a single GGUF variant. |

`parse_model_spec` rules (verified outputs):

```text
'unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS' -> ('unsloth/Qwen3.6-35B-A3B-GGUF', 'UD-IQ4_XS')
'Qwen/Qwen2.5-7B-Instruct-GGUF'          -> ('Qwen/Qwen2.5-7B-Instruct-GGUF', None)
'Qwen/Qwen2.5-7B-Instruct-GGUF:'         -> ('Qwen/Qwen2.5-7B-Instruct-GGUF', None)
'C:\models\a.gguf', 'models/a.gguf'      -> None    (ends in .gguf, contains '\', or absolute)
'org/sub/repo:Q4_0', 'repo-only:Q4_0', 'org/:Q4_0' -> None   (needs exactly one '/', both parts non-empty)
```

`match_gguf_files` rules:

- Only `.gguf` files count. Files whose basename starts with `mmproj` (vision projectors) are skipped.
- Matching uses the *stem*: the basename without `.gguf` and without a `-NNNNN-of-NNNNN` split suffix, compared case-insensitively.
- A file matches `QUANT` if its stem ends with it and the character before it is `-`, `_` or `.`, or if the stem equals the quant. So `Q4_K_M` matches `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`, `4_K_M` matches nothing, and `Q4_0` matches `model.Q4_0.gguf`.
- Shards are grouped by folder and stem, and the shortest stem wins. `IQ4_XS` selects `Model-IQ4_XS.gguf` over `Model-UD-IQ4_XS.gguf`; ask for `UD-IQ4_XS` to get the Unsloth Dynamic file. Stems of equal length are ordered alphabetically by folder, then by stem.
- Without a quant, more than one group is ambiguous. The function returns `[]`, and the error names the available quants.
- The result is every shard, sorted. llama-server receives the first shard and loads the rest from the same folder.

```text
match_gguf_files([...q4_k_m-00002-of-00002.gguf, ...q4_k_m-00001-of-00002.gguf, ...q5_k_m-0000{1,2}-of-00002.gguf], "Q4_K_M")
  -> ['qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf', 'qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf']
```

### From config to command line (`build_server_args`)

| Setting | Flag | Passed when |
|---|---|---|
| resolved GGUF (first shard) | `--model <file>` | always |
| `llama_server.host` (`127.0.0.1`) | `--host` | always, as written. Kumaru connects to `dial_address(host)`, so the wildcards `0.0.0.0` and `::` work and are reached on `127.0.0.1` and `[::1]`. A wildcard also exposes llama-server to your network, with no API key. |
| `llama_server.port`, else a free port | `--port` | always |
| `backend.max_context_length` (CLI `--context`) | `--ctx-size` | always |
| `llama_server.parallel` (4, range 1–64) | `--parallel` | always |
| `llama_server.flash_attn` (`"on"`) | `--flash-attn` | always |
| — | `--jinja` | always. Renders the GGUF's own Jinja chat template, which is what reads `enable_thinking`. |
| `backend.model` | `--alias` | always. This is the id that `/v1/models` reports. |
| — | `--no-webui` | always. Kumaru serves its own UI. |
| — | `--log-colors off` | always. Keeps ANSI codes out of the log file. |
| `llama_server.kv_unified` (true) | `--kv-unified` | if true |
| `backend.n_gpu_layers` (-1) | `--n-gpu-layers N` | if not -1. `0` means CPU only. |
| `backend.n_threads` | `--threads N` | if set and non-zero |
| `llama_server.cache_type_k` / `cache_type_v` | `--cache-type-k` / `--cache-type-v` | if set |
| `llama_server.n_cpu_moe` | `--n-cpu-moe N` | if not null (`0` is passed) |
| `llama_server.batch_size` / `ubatch_size` | `--batch-size` / `--ubatch-size` | if set |
| `llama_server.extra_args` | appended verbatim, last | always |

`--reasoning-format` is never passed; llama-server's default already puts thinking into `reasoning_content`. The transformers-only settings `device`, `dtype`, `load_in_4bit`/`load_in_8bit`, `attn_implementation` and `trust_remote_code` are ignored. LoRA adapters are supported by the transformers backend only: with `adapter_path` (`--adapter`) set, `load()` refuses to start with `ConfigError` `The llamacpp_server backend cannot attach a LoRA adapter …` ([base.md](base.md)).

Print the command line offline (env overrides included):

```powershell
@'
from pathlib import Path
from kumaru.core.config import load_config
from kumaru.backends.llamacpp_server import build_server_args
cfg = load_config("qwen36")
print(" ".join(build_server_args(Path("llama-server.exe"), Path("model.gguf"), cfg.backend, 8081)))
'@ | .\.venv\Scripts\python.exe -
```

For `qwen36` this prints the following:

```text
llama-server.exe --model model.gguf --host 127.0.0.1 --port 8081 --ctx-size 131072 --parallel 4 --flash-attn on --jinja --alias unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS --no-webui --log-colors off --kv-unified
```

At runtime, `kumaru serve -c qwen36 --log-level DEBUG` logs the real command line (`command: …`). `kumaru chat` and `kumaru bench run` ignore `telemetry.log_level`; set `$env:KUMARU_LOG_LEVEL = 'DEBUG'` for them.

### Thinking and sampling on the wire

| `GenerationParams` field | Request field | Sent when |
|---|---|---|
| `max_tokens`, `temperature`, `top_p` | same names, plus `stream` | always |
| `stop` | `stop` (first 4) | non-empty. Kumaru also enforces every stop sequence locally, on the answer only. llama-server itself matches stops on the raw generated text, thinking included, so a stop string that appears in the reasoning ends the reply. Leave `stop` empty for thinking models. |
| `seed` | `seed` | not null |
| `presence_penalty` | `presence_penalty` | not null. It is a standard OpenAI field, so it is sent even in strict mode. |
| `top_k` | `top_k` | > 0 |
| `min_p` | `min_p` | not null. llama.cpp's own default is 0.05, so profiles send `0.0` explicitly. |
| `repetition_penalty` | `repeat_penalty` (llama.cpp's name) | ≠ 1.0 |
| `thinking` | `chat_template_kwargs: {"enable_thinking": <bool>}` | not null |
| `thinking_budget` | `reasoning_budget_tokens` + `reasoning_budget_message` | not null **and** `thinking` is not `false` (so also when `thinking` is null) |

The budget works like this. Once the model has produced `reasoning_budget_tokens` thinking tokens, llama-server injects `llama_server.reasoning_budget_message` before the end-of-thinking tag, and the model goes on to answer. Every row from `top_k` down is skipped when `backend.extra.strict_openai: true`. Never set that flag against llama-server, because it silently disables the thinking toggle and the budget.

Inspect a request body offline. Constructing the backend starts nothing:

```powershell
@'
import json
from kumaru.agent.chat import ChatAgent
from kumaru.backends.llamacpp_server import LlamaServerBackend
from kumaru.core.config import load_config
from kumaru.core.types import Message
cfg = load_config("qwen36")
backend = LlamaServerBackend(cfg.backend)
params = ChatAgent(backend, cfg).resolve_params(thinking=True, thinking_budget=512)
print(json.dumps(backend._payload([Message.user("hi")], params, stream=True), indent=1))
'@ | .\.venv\Scripts\python.exe -
```

The output contains the thinking preset (`temperature` 1.0, `top_p` 0.95, `top_k` 20, `min_p` 0.0, `presence_penalty` 1.5), `"chat_template_kwargs": {"enable_thinking": true}`, `"reasoning_budget_tokens": 512` and the budget message.

### Read `info()`

```powershell
(Invoke-RestMethod http://127.0.0.1:8080/v1/info).agent.backend | ConvertTo-Json -Depth 4
```

| Field | Value |
|---|---|
| `model_id` | `backend.model` |
| `device` | `"external server"` in attach mode. Managed: `"cpu"` for `n_gpu_layers: 0`, `"gpu (auto-fit)"` for `-1`, `"gpu (N layers)"` for any other N. |
| `dtype` | `"gguf"` |
| `context_length` | `default_generation_settings.n_ctx` from `/props`, falling back to `max_context_length` |
| `quantization` | the spec's quant, e.g. `UD-IQ4_XS`, otherwise `"gguf"` |
| `extra` | `engine`, `base_url`, `managed`, `build` (`/props` `build_info`), `slots` (`total_slots`), `model_file`, `pid`, `log_file`. Keys with null values are dropped. |

Before `load()` completes, `extra.base_url` shows the inherited placeholder `http://127.0.0.1:11434/v1`; ignore it. Kumaru's history budget uses `backend.max_context_length`, not `info().context_length`.

### Attach to a server you run yourself

```powershell
# terminal 1: your flags, your lifetime
.\tools\llama.cpp\llama-server.exe --model D:\models\x.gguf --port 8081 --ctx-size 32768 --jinja
# terminal 2
$env:KUMARU_BACKEND__BASE_URL = 'http://127.0.0.1:8081/v1'
kumaru serve -c qwen36 --context 32768
```

Use attach mode to:
- keep a model loaded while you restart Kumaru during development;
- share one server between tools;
- use flags that Kumaru does not expose;
- reach a server on another machine.

Rules for attach mode:
- **Point it at the `/v1` root**, e.g. `http://127.0.0.1:8081/v1`. Kumaru strips a trailing `/v1` to reach `/props`.
- **Kumaru checks `/v1/models` at load.** If the server is not up, the error is "Cannot reach OpenAI-compatible server at …".
- **Only `reasoning_budget_message` still applies.** The other `llama_server.*` options, `model_path` and `models_dir` are ignored.
- **Keep `max_context_length` within the server's window.** It must be ≤ the server's `--ctx-size`, because Kumaru sizes the history from it.
- **Add an API key if the server needs one.** `backend.api_key` sends `Authorization: Bearer <api_key>`, for a server started with `--api-key`.

### Logs

- **Location.** The default is `<repo>\runs\logs\llama-server-<port>.log`, under the repo root, not the current directory. Set `llama_server.log_file` to override it.
- **Appending.** The file is opened in append mode. Free ports usually give a new file per start, while a fixed port gathers every start in one file. Nothing rotates or deletes logs.
- **Kumaru's own log.** It records `starting llama-server on port N (log: …)` at INFO and the full command at DEBUG. `kumaru serve` takes the level from `--log-level` (`telemetry.log_level`); `chat` and `bench run` take it from `$env:KUMARU_LOG_LEVEL`.
- **Crashes.** When the child exits during startup or during a request, the error embeds the last 25 lines of its log, read from at most its final 16 KB.

```powershell
Get-ChildItem runs\logs\llama-server-*.log | Sort-Object LastWriteTime | Select-Object -Last 1 | Get-Content -Wait -Tail 50
```

### Share the GPU, or run on a smaller card

When VRAM is overcommitted, Windows (WDDM driver model) does not fail the allocation. It silently moves memory into shared system RAM, and generation speed collapses: ~110 tok/s fell to 7 tok/s with a 131k context while a game was running. The cause is that fit sizes the offload once, at load time; a game that grows its VRAM use afterwards pushes llama-server out. Task Manager → Performance → GPU → "Shared GPU memory" climbing while Kumaru generates is the sign. The recipe below was measured on an RTX 3090 next to World of Warcraft (4.2 GiB): 11.9 GiB VRAM, ~27 tok/s, no spill.

```powershell
$env:KUMARU_BACKEND__LLAMA_SERVER__N_CPU_MOE = '18'
kumaru serve -c qwen36 --context 65536
```

| Lever | VRAM effect | Cost |
|---|---|---|
| `--context` / `backend.max_context_length` | Qwen3.6's KV cache is 20 KiB/token, so 131,072 → 65,536 frees 1.25 GiB | Smaller history budget: 32,640 tokens at 65,536 with `qwen36`'s `max_tokens` 32,768 |
| `llama_server.n_cpu_moe: N` | ~0.37 GiB freed per layer whose experts move to system RAM | Speed: the 18-layer recipe ran at ~27 tok/s |
| `llama_server.cache_type_k` / `cache_type_v: q8_0` | Roughly halves the KV cache (8.5 instead of 16 bits per value) | Small quality cost. A quantized V cache needs flash attention, which is already `on`. |
| `extra_args: ["--fit-target", "8192"]` | Fit leaves this many MiB free at load time | Placement is decided once, at load time |
| NVIDIA Control Panel → Manage 3D settings → CUDA - Sysmem Fallback Policy → Prefer No Sysmem Fallback | none | An overflow becomes an out-of-memory error instead of a silent slowdown |

### Upgrade llama.cpp

1. Stop Kumaru. The script deletes `tools\llama.cpp\` first, and a running `llama-server.exe` locks those files.
2. Pick a release tag from https://github.com/ggml-org/llama.cpp/releases and run `.\scripts\get-llama-server.ps1 -Version bNNNNN`.
3. Restart Kumaru and confirm `extra.build` in `/v1/info`.

Upgrade when a model's log says its architecture is unknown, or for new kernels and fixes. A later run **without** `-Version` reinstalls the pinned default (`b11541`), which is a downgrade. See [get-llama-server.ps1.md](../../../scripts/get-llama-server.ps1.md).

### From Python

```python
import asyncio
from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS
from kumaru.core.types import Message

async def main() -> None:
    cfg = load_config("qwen25-7b-gguf")
    backend = BACKENDS.create(cfg.backend.name, cfg.backend)
    await backend.load()                       # spawns llama-server; downloads on first use
    try:
        params = cfg.generation.merged(max_tokens=64)
        async for chunk in backend.stream([Message.user("Name three primes.")], params):
            print(chunk.delta, end="", flush=True)
    finally:
        await backend.aclose()                 # always: this is what frees the VRAM

asyncio.run(main())
```

The `atexit` hook and the Job Object are safety nets. Always call `aclose()` yourself.

## How to extend it

- **A new llama-server flag.** Use `llama_server.extra_args` first; that needs no code. Add a typed `LlamaServerOptions` field only if the value needs validation or Kumaru must reason about it. In that case, add the mapping to `build_server_args` and a case to `tests\test_llamacpp_server.py::test_every_knob_reaches_the_command_line`.
- **A new request field.** Override `_extended_fields`, call `super()`, and add keys. Keep the condition explicit so that `None` still means "engine default":

```python
from typing import Any
from kumaru.backends.llamacpp_server import LlamaServerBackend
from kumaru.core.registry import BACKENDS
from kumaru.core.types import GenerationParams

@BACKENDS.register("llamacpp_server_dry")
class DryLlamaServerBackend(LlamaServerBackend):
    name = "llamacpp_server_dry"

    def _extended_fields(self, params: GenerationParams) -> dict[str, Any]:
        fields = super()._extended_fields(params)
        if (dry := self.config.extra.get("dry_multiplier")) is not None:
            fields["dry_multiplier"] = float(dry)   # llama.cpp DRY repetition sampler
        return fields
```

  Import the module before the backend is created, or add a `BACKENDS.register_lazy(...)` line in `src\kumaru\core\registry.py`.

- **Another engine with the same lifecycle** (for example vLLM). Copy the pattern: subclass `OpenAICompatBackend`, start the process in `_load` before `super()._load()`, wrap `_load` in `try/except BaseException` that stops the process, and stop it in `aclose`. Override `_connection_lost(exc)` to report the process's exit code when it has died, and defer to `super()` otherwise.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `llama-server was not found.` | No binary in `tools\llama.cpp\` and none on `PATH` | `.\scripts\get-llama-server.ps1`, or set `backend.llama_server.binary` |
| `backend.llama_server.binary does not exist: …` | Wrong explicit path. There is no fallback to the bundled copy. | Fix the path or unset it |
| `llama-server exited during startup (code N)`, and the log tail mentions allocation or out-of-memory | Weights plus KV cache plus buffers do not fit | Lower `--context`, set `llama_server.n_cpu_moe`, use a smaller quant, or set `cache_type_k`/`cache_type_v: q8_0` |
| Same error, and the tail names an unknown architecture | llama.cpp build is older than the model | Upgrade with `get-llama-server.ps1 -Version bNNNNN` |
| `Port <port> on <host> is already in use.` | A fixed `llama_server.port` that something already accepts connections on, often an earlier llama-server. Checked after the model is resolved and before anything starts. | Stop that program, choose another `backend.llama_server.port`, or leave it unset. To use a llama-server you started yourself, set `backend.base_url` ([attach mode](#attach-to-a-server-you-run-yourself)). |
| `llama-server exited during startup (code N)`, and the tail shows a bind error | The fixed port cannot be bound although nothing answers on it, e.g. it falls in a range Windows reserves (`netsh interface ipv4 show excludedportrange protocol=tcp`) | Choose another port, or leave `port: null` |
| `llama-server did not become healthy within 900s.` | The model loads very slowly (slow disk, or a large model on CPU) | Read the log (the path is in the hint). Raise `llama_server.startup_timeout_s`. |
| `Could not list files of '<repo>' on Hugging Face: …` (401, 403, 429 or a connection error) | Gated or private repo, anonymous rate limit, or offline | `$env:HF_TOKEN = '<token>'` (huggingface_hub reads it), or `hf auth login`. Or download the file yourself and set `backend.model_path`. |
| `huggingface_hub is required to download models.` | A broken or partial environment: huggingface_hub is a base dependency of Kumaru | As the hint says: run `pip install -e "."` in the Kumaru folder, or download the `.gguf` yourself and set `backend.model_path` |
| `No GGUF matching quant 'X' in <repo>.` | Typo, or a quant the repo does not ship | Pick one from the `Available:` hint |
| `<repo> contains several GGUF variants; name one as '<repo>:QUANT'.` | Spec has no quant | Add `:QUANT` |
| `GGUF model not found: <spec>` | Not an existing file and not a valid `org/repo[:QUANT]` | Fix the path or the spec |
| Warning `cached GGUF <path> is missing parts; fetching them` | A split download was interrupted between parts | Nothing: the missing parts are downloaded. Offline, copy them into the folder yourself. |
| CPU-like speed; the script warned "No CUDA device detected" | CPU or Vulkan build, or a driver older than 580 with the CUDA 13.4 build | Reinstall with the right `-Flavor` (e.g. `cuda-12.4`) |
| Speed falls to single digits while another app runs | WDDM VRAM spill | [Share the GPU](#share-the-gpu-or-run-on-a-smaller-card) |
| `llama-server exited (code N) during the request. Last log lines: …` | The child died after startup: out of VRAM (for example a game grew its share), a driver reset, a manual kill. Nothing restarts it, so every later request fails the same way. | Restart `kumaru serve`, or reload with `POST /v1/backend` `{"name": "llamacpp_server"}`. If it recurs, read the log tail; lower `backend.max_context_length` or set `llama_server.n_cpu_moe`. |
| `http://127.0.0.1:<port>/v1 sent nothing for 300s.` | No byte within `backend.timeout_s`: a very long prefill, or a request queued behind `parallel` long generations | Raise `backend.timeout_s` or `llama_server.parallel` |
| `Lost the connection to http://127.0.0.1:<port>/v1 (<Type>: …).` | The connection dropped while the child was alive, or still exiting. In attach mode, your server is down or unreachable. | Retry. If the next request reports an exit code, see the row above. |
| Long chats fail or truncate with `kv_unified: false` | Each slot only gets `ctx/parallel` tokens | Keep `kv_unified: true` |
| `Upstream returned 400: …` about the context size | The request does not fit the server's window by llama.cpp's own token count. The agent budgets the history, and trims `max_tokens` to the room left, with the `count_tokens` heuristic (~3.6 characters per token, tuned for English), which undercounts dense text. `sliding_window` also always keeps the two newest messages, however long. In attach mode, `max_context_length` may exceed the server's `--ctx-size`. | Lower `agent.max_history_tokens` to leave slack. In attach mode, set `--context` to the server's window. |
| `Input should be 'on', 'off' or 'auto'` for `flash_attn` | Another value, e.g. `1` or `enabled`, in YAML or `KUMARU_BACKEND__LLAMA_SERVER__FLASH_ATTN`. Booleans are fine: they map to `on`/`off`. | `on`, `off` or `auto` |
| A `llama-server.exe` is left running after Kumaru is gone | Job Object assignment failed (warning `could not attach llama-server to a job object`) | `Get-Process llama-server`, then `Stop-Process -Id <pid>` |

## Related files

- `src\kumaru\backends\openai_compat.py` — the parent class: protocol, SSE parsing, `_extended_fields`, `_connection_lost`.
- `src\kumaru\core\config.py` — `LlamaServerOptions`, `BackendConfig.models_dir`, env override parsing.
- `scripts\get-llama-server.ps1` — installs the binary into `tools\llama.cpp\`.
- `configs\qwen36.yaml`, `configs\qwen25-7b-gguf.yaml` — the profiles that use this backend.
- `src\kumaru\server\routes_chat.py` — `/v1/info` and `POST /v1/backend` (reload or hot-swap).
- `tests\test_llamacpp_server.py` — spec parsing, matching, shard completeness, exact command line, payload, `info()` and its device labels, wildcard addresses and ports, a taken fixed port, a server that dies mid-request, binary search order, resolution (including a partly downloaded split).
- `docs\BENCHMARKS.md` — scores measured with this backend.
