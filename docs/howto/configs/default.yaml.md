# `configs\default.yaml`

> The reference profile: every Kumaru setting at its code default, with a comment on each. As a profile, it serves Qwen2.5-7B-Instruct through Hugging Face transformers with an 8,192-token window on `127.0.0.1:8080`.

**Read this when:**
- you need to know what a setting does, its default and valid range, or which backend reads it;
- a setting seems to be ignored and you need the precedence rules;
- you are writing a profile, or overriding a field from the environment;
- you want the plain transformers setup for a GPU that holds a 7B model in 16-bit.

---

## What it does

`configs\default.yaml` sets every field of `KumaruConfig` (`src\kumaru\core\config.py`) to its code default. It omits only `backend.extra`, a free-form dict that defaults to `{}`. `kumaru config -c default` and a bare `kumaru config` print the same configuration with one exception: `backend.models_dir` is `~/.cache/kumaru/models` in the file and the expanded `C:\Users\<you>\.cache\kumaru\models` in the code. `llamacpp_server` is the only reader of that field and expands `~`, so the two behave the same.

Three consequences:
- **No `-c` means the code defaults, not this file.** Kumaru reads `default.yaml` only with `-c default`. `kumaru serve` without `-c` runs the field defaults in `config.py`, which hold the same values. An edit to this file therefore does not change a run without `-c`.
- **Profiles are not layered on this file.** `load_config` merges a profile onto an empty dict, and Pydantic fills every missing field from the code. Editing `default.yaml` therefore changes only runs that pass `-c default`.
- **Don't start a profile by copying this file.** A copy pins every value, including the ones you didn't mean to choose. Write only the fields you change ([Write a profile](#write-a-profile)).

### Precedence

At startup four layers are merged, and the later one wins:

| Layer | Example |
|---|---|
| 1. Field defaults in `config.py`; for `generation`, `GenerationParams` in `src\kumaru\core\types.py` | `temperature: 0.7` |
| 2. The YAML profile named by `-c` | `configs\test.yaml` sets `temperature: 0.0` |
| 3. Environment variables `KUMARU_<SECTION>__<FIELD>` | `$env:KUMARU_GENERATION__TEMPERATURE = '0.3'` |
| 4. CLI flags that set config fields ([table below](#cli-flags)) | `kumaru serve -c qwen36 --context 65536` |

`deep_merge` merges mappings key by key and replaces everything else whole. A list in a higher layer replaces the lower list and is never appended to it. That matters for `generation.stop`, `server.cors_origins`, `agent.enabled_tools`, `agent.workspace.allowed_commands` and `backend.llama_server.extra_args`.

`-c` takes a profile name or a file path. A bare name, with no folder and no `.yaml` or `.yml` suffix, always becomes `<repo>\configs\<value>.yaml`, even when the current directory holds a file of that name. So profile names work from any directory, and a file in a project you run Kumaru in cannot pose as one. Any other value is a path, resolved against the current directory: `.\my-profile`, `my-profile.yaml`.

Each request then resolves the decoding parameters once more (`ChatAgent.resolve_params` in `src\kumaru\agent\chat.py`):

```text
generation  <  thinking_mode (only while thinking is on)  <  the request's own fields
```

Thinking is on when the request turns it on, or when the request leaves it unset and `generation.thinking` is `true`. Request fields that are unset (`null`) never override, so a request cannot reset a value to `null`. Three consequences that surprise people:
- **The sampling flags of `chat` and `bench run` are request fields.** `-t`, `--max-tokens`, `--think`/`--no-think` and `--thinking-budget` are sent with every request, as the web UI sends its settings, so they beat both `generation` and `thinking_mode`. `kumaru chat -c qwen36 -b echo --think -q hi` reports the preset's `max_tokens=32768, temperature=1.0`; adding `-t 0.2 --max-tokens 99` reports `max_tokens=99, temperature=0.2`. Without `--max-tokens`, `bench run` chooses the output cap per benchmark ([runner.md](../src/kumaru/eval/benchmarks/runner.md)).
- **While thinking, `thinking_mode` beats `generation`.** A `generation` value from YAML or `KUMARU_GENERATION__*` loses to the same field of the preset: with `$env:KUMARU_GENERATION__TEMPERATURE = '0.3'`, the command above still reports `temperature=1.0`, and `0.3` applies only without `--think`. To change the thinking sampling for every request, set the preset: `$env:KUMARU_THINKING_MODE__TEMPERATURE = '0.6'` gives `temperature=0.6`.
- **`generation.thinking: null` skips the preset.** A request that doesn't mention thinking gets no `thinking_mode` preset, even when the model's chat template thinks by default. Profiles for thinking models set the field explicitly (`qwen36` sets `false`). The web UI always sends an explicit `thinking`.

### CLI flags

| Flag | Sets | Commands |
|---|---|---|
| `-c`, `--config` | the profile (layer 2) | `serve`, `chat`, `config`, `eval`, `bench run` |
| `-b`, `--backend` | `backend.name` | `serve`, `chat`, `config`, `eval`, `bench run` |
| `-m`, `--model` | `backend.model` | `serve`, `chat`, `config`, `eval`, `bench run` |
| `--device` | `backend.device` | `serve`, `chat` |
| `--adapter` | `backend.adapter_path` | `serve`, `chat`, `eval`, `bench run` |
| `--context` (≥ 512) | `backend.max_context_length` | `serve`, `chat`, `bench run` |
| `--host`, `-p`/`--port` | `server.host`, `server.port` | `serve` |
| `--log-level` | `telemetry.log_level` | `serve` |
| `-w`, `--workspace` | `agent.workspace.root`, and appends the project-folder tools to the resolved `agent.enabled_tools` instead of replacing the list | `chat` |
| `-t`/`--temperature`, `--max-tokens`, `--think`/`--no-think`, `--thinking-budget` | no config field: the request's `temperature`, `max_tokens`, `thinking`, `thinking_budget` | `chat`, `bench run` |

The last row is request-level, as described above, so `kumaru config` cannot show it. Every other field is set by YAML or an environment variable.

### Validation

Every section except `generation` rejects unknown keys (`extra="forbid"`), so a typo fails at load. The same applies to `KUMARU_*` variables:

```text
error: Invalid configuration: 1 validation error for KumaruConfig
backend.modle
  Extra inputs are not permitted
```

Exceptions and gaps:
- **`generation` ignores unknown keys.** A misspelt `temprature:` is silently dropped.
- **`backend.extra` is free-form.** Backends read the keys they know (table below) and ignore the rest.
- **Ranges are checked at load.**
  - `generation`: `max_tokens` 1–131,072, `temperature` 0–2, `top_p` 0–1, `top_k` ≥ 0, `repetition_penalty` 0.5–2, `min_p` 0–1, `presence_penalty` −2–2, `thinking_budget` ≥ 0. `thinking_mode` has the same ranges.
  - `llama_server`: `parallel` 1–64, `n_cpu_moe` ≥ 0, `batch_size`/`ubatch_size` ≥ 1, `startup_timeout_s` > 0.
  - `agent`: `max_tool_iterations` 1–100. In `agent.workspace`: `command_timeout_s` 1–3,600, `max_output_chars` 1,000–200,000, `max_file_bytes` 1,024–8,388,608, `max_list_entries` 10–10,000. `allowed_commands` takes bare program names only, and refuses shells and launchers: `Value error, 'cmd' is a shell or launcher, which could run any command`.
- **`telemetry.log_level` is checked at load.** It must be `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`, in any case and with any surrounding spaces; `WARN` becomes `WARNING` and `FATAL` becomes `CRITICAL`. Any other name fails before the server starts: `Invalid configuration: … telemetry.log_level  Input should be 'DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'`. Earlier builds accepted any string, and `kumaru serve` then died inside uvicorn with `KeyError: 'warn'`. The bare `KUMARU_LOG_LEVEL` variable is not a config field and is not checked; an unknown name there falls back to INFO.
- **An empty section fails.** A header whose keys are all commented out is YAML `null`, which fails with `Input should be a valid dictionary or instance of ThinkingModeConfig` (or the respective section class). Delete the header too.
- **Some values are checked only when used:**
  - `backend.name`, when the backend is created. Names are case-sensitive (`error: Unknown backend: 'Echo'.`, followed by the list). Aliases: `hf` → `transformers`, `gguf` → `llamacpp`, `llama-server`/`llama_server` → `llamacpp_server`, `ollama`/`vllm`/`remote` → `openai_compat`.
  - `agent.enabled_tools`, when the agent is built: `Unknown tool: 'x'. (Available tools: current_time, edit_file, list_files, read_file, run_command, write_file)`. A non-empty list also fails there on a backend that cannot call tools, which includes this profile's `transformers` (see the [`agent`](#agent) table).
  - `agent.workspace.root`, when `kumaru chat` builds the agent: it must be an existing folder, not a whole drive, not one that contains your home folder, and not inside a `.git` folder.
  - `backend.dtype`, when transformers loads: `Unknown dtype 'x'`.
  - `agent.memory_policy` is never checked. An unknown name logs `unknown memory_policy 'x'; using sliding_window`.

## When to use this profile

Use `default` in three cases:
- **As the reference.** The comments in the file are the short form of this page.
- **To serve Qwen2.5-7B-Instruct with transformers** on a GPU that holds it in 16-bit. The first run downloads the weights (~15 GB, according to `setup.ps1`'s hint) into the Hugging Face cache.
- **As the code-defaults baseline when you debug a profile.** Compare `kumaru config -c <profile>` with `kumaru config`.

For other situations a profile exists:

| Profile | Backend | Model | Context | For |
|---|---|---|---|---|
| `default` | `transformers` | `Qwen/Qwen2.5-7B-Instruct` | 8,192 | Reference; 16-bit 7B |
| [`local-gpu`](local-gpu.yaml.md) | `transformers` | `Qwen/Qwen2.5-7B-Instruct` | 16,384 | 24 GB GPU: bfloat16, `max_tokens` 2,048 |
| [`low-vram`](low-vram.yaml.md) | `transformers` | `Qwen/Qwen2.5-7B-Instruct` | 8,192 | 6–12 GB GPU: 4-bit NF4 |
| [`cpu-only`](cpu-only.yaml.md) | `llamacpp` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | 4,096 | No GPU: 1.5B GGUF in-process |
| [`vllm`](vllm.yaml.md) | `openai_compat` | `Qwen/Qwen2.5-7B-Instruct` | 16,384 | A vLLM server at `http://127.0.0.1:8000/v1` |
| [`qwen36`](qwen36.yaml.md) | `llamacpp_server` | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS` | 131,072 | 24 GB GPU: best local quality, thinking |
| [`qwen25-7b-gguf`](qwen25-7b-gguf.yaml.md) | `llamacpp_server` | `Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` | 32,768 | ~8 GB GPU; benchmark baseline |
| [`test`](test.yaml.md) | `echo` | `echo` (a label) | 8,192 | No model: install, UI and API checks |

## Every field explained

### `backend`

| Field | Default | Effect and notes |
|---|---|---|
| `name` | `transformers` | The engine: `transformers`, `llamacpp_server`, `llamacpp`, `openai_compat` or `echo` (aliases above). CLI `-b`. |
| `model` | `Qwen/Qwen2.5-7B-Instruct` | Its meaning depends on the backend. `transformers`: a Hugging Face repo id or a local directory. `llamacpp_server`: `org/repo:QUANT`, `org/repo` with a single GGUF variant, or a `.gguf` path ([model specs](../src/kumaru/backends/llamacpp_server.md#model-specs)); also passed as `--alias`. `llamacpp`: an existing `.gguf` path, or a repo id whose file is chosen by `extra.gguf_filename`. `openai_compat`: the model name sent upstream. `echo`: a label. CLI `-m`. |
| `device` | `auto` | Read by `transformers` only. `auto` picks `cuda`, then `mps`, then `cpu`. On CUDA the model is placed with `device_map` (`extra.device_map`, default `"auto"`), so an index such as `cuda:1` does not choose the GPU; use `$env:CUDA_VISIBLE_DEVICES = '1'`. CLI `--device`. |
| `dtype` | `auto` | `transformers` only. `auto` gives `bfloat16` on CUDA GPUs that support it, else `float16`, and `float32` on CPU and MPS. Also accepts `bf16`, `fp16` and `fp32`. |
| `load_in_4bit` / `load_in_8bit` | `false` / `false` | `transformers` only, via bitsandbytes (`.\scripts\setup.ps1 -Quant`, on top of the `-Gpu` install that the transformers backend needs; `-Gpu -Quant` does both, in that order). 4-bit is NF4 with double quantization and computes in `dtype`. 4-bit wins if both are set. |
| `attn_implementation` | `null` | `transformers` only. Passed to `from_pretrained` only when set, e.g. `flash_attention_2` if installed. |
| `max_context_length` | `8192` | The context window. `transformers` clamps it to the checkpoint's maximum. `llamacpp` uses it as `n_ctx`. `llamacpp_server` passes it as `--ctx-size`, which sizes the KV cache. `openai_compat` and `echo` only report it. For every backend it also bounds the history budget (`agent.max_history_tokens`). CLI `--context` (≥ 512). |
| `trust_remote_code` | `false` | `transformers` only. Runs Python code shipped in the model repo; enable it only for repos you trust. |
| `max_batch_size` | `1` | `transformers` only. Greedy (temperature 0) requests that arrive together are decoded as one batch of up to this many (1–512); `1` turns batching off. Each decoding step reads the weights once for the whole batch: a benchmark of Qwen2.5-0.5B ran 4.6× faster at 8. It halves itself, for the rest of the load, after running out of GPU memory. In `bfloat16` a batched answer can differ from a one-at-a-time one where two tokens are near-tied; `float32` matches exactly. Streamed and sampled requests never batch. `bench run` keeps this many greedy questions in flight unless `-j` is given. [Details](../src/kumaru/backends/transformers_backend.md#mental-model). |
| `adapter_path` | `null` | `transformers` backend only. A LoRA adapter directory from `kumaru train`, attached with peft (installed by `setup.ps1 -Gpu`). `extra.merge_adapter: true` folds it into the weights. Every other backend refuses to load while it is set: `The <name> backend cannot attach a LoRA adapter (backend.adapter_path is '<path>').` CLI `--adapter`. |
| `base_url` | `null` | `openai_compat`: the API root; `null` means `http://127.0.0.1:11434/v1` (Ollama). `llamacpp_server`: setting it attaches to a server you started instead of spawning one ([attach mode](../src/kumaru/backends/llamacpp_server.md#attach-to-a-server-you-run-yourself)). |
| `api_key` | `null` | Sent as `Authorization: Bearer <api_key>` by `openai_compat` and `llamacpp_server`. Prefer `$env:KUMARU_BACKEND__API_KEY` to a committed file. |
| `timeout_s` | `300` | HTTP timeout of `openai_compat` and `llamacpp_server` requests. The connect timeout is fixed at 10 s. |
| `model_path` | `null` | A local `.gguf` file. It wins over `model`. Read by `llamacpp` and `llamacpp_server`; both expand `~`. |
| `n_gpu_layers` | `-1` | Shared by `llamacpp` and `llamacpp_server`, but `-1` means different things. `llamacpp`: the number of layers to offload; `-1` offloads every layer and the load fails if they do not fit (on Windows it may instead spill into shared memory and crawl); `0` is CPU only. `llamacpp_server`: `-1` omits `--n-gpu-layers`, so llama.cpp fits as many layers as free VRAM allows; any other value is passed as is. |
| `n_threads` | `null` | CPU threads, as `n_threads` for `llamacpp` and `--threads` for `llamacpp_server`. Passed only when set and non-zero. |
| `models_dir` | `~/.cache/kumaru/models` | `llamacpp_server` only. Where `org/repo:QUANT` specs are looked up and downloaded to, as `<models_dir>\<org>\<repo>\`. Point it at a large, fast disk. `llamacpp` downloads into the Hugging Face cache instead. |

### `backend.llama_server`

Read only by `llamacpp_server`. The full flag order is in [llamacpp_server.md](../src/kumaru/backends/llamacpp_server.md#from-config-to-command-line-build_server_args).

| Field | Default | Flag | Effect and notes |
|---|---|---|---|
| `binary` | `null` | — | `null` searches `<repo>\tools\llama.cpp\` (installed by [get-llama-server.ps1](../scripts/get-llama-server.ps1.md)), then `PATH`. |
| `host` | `127.0.0.1` | `--host` | Kumaru connects to this address as well, except that a wildcard bind is dialled on loopback: `0.0.0.0` as `127.0.0.1` and `::` as `::1` (Windows refuses to connect to `0.0.0.0`). A wildcard exposes the unauthenticated llama-server to the LAN; to share the model, expose Kumaru (`server.host`, with `server.api_key`) instead. |
| `port` | `null` | `--port` | `null` picks a free port at start. A fixed port that something already listens on fails the load with `Port <port> on <host> is already in use.`, so a leftover llama-server cannot answer in place of the new one. |
| `parallel` | `4` | `--parallel` | Slots: concurrent requests batched into one forward pass. |
| `kv_unified` | `true` | `--kv-unified` | One KV pool for all slots. Without it each slot gets `ctx-size / parallel` tokens. |
| `flash_attn` | `"on"` | `--flash-attn` | `on`, `off` or `auto`. Quotes are optional: YAML reads a bare `on`/`off` (or `true`/`false`, `yes`/`no`) as a boolean, and a validator maps it back to `"on"`/`"off"`. Any other value fails validation. |
| `cache_type_k` / `cache_type_v` | `null` | `--cache-type-k` / `--cache-type-v` | `null` is f16. `q8_0` roughly halves the KV cache. A quantized V cache needs flash attention. |
| `n_cpu_moe` | `null` | `--n-cpu-moe` | Mixture-of-experts models: keep the experts of the first N layers in system RAM. Each layer frees ~0.37 GiB of VRAM on Qwen3.6. `0` is passed. |
| `batch_size` / `ubatch_size` | `null` | `--batch-size` / `--ubatch-size` | Logical and physical prompt batch sizes; `null` keeps llama.cpp's defaults. |
| `reasoning_budget_message` | `"\n\nI have used my thinking budget, …"` | request field | Sent with `reasoning_budget_tokens` whenever a thinking budget is set and thinking is not `false`. The model is told this when the budget runs out, so it answers instead of stopping mid-thought. Also used in attach mode. |
| `startup_timeout_s` | `900` | — | The deadline from launch to a healthy server. A model download happens before the process starts, so it does not count against this limit. Raise it for slow disks or CPU loads. |
| `log_file` | `null` | — | `null` is `<repo>\runs\logs\llama-server-<port>.log`. A relative path resolves against the current directory. The file is opened in append mode. |
| `extra_args` | `[]` | appended last | Any other llama-server flag, e.g. `["--fit-target", "8192"]`. |

### `backend.extra`

Not in `default.yaml`. These are all the keys any built-in backend reads:

| Key | Read by | Default | Effect |
|---|---|---|---|
| `delay_s` | `echo` | `0.01` | Sleep before each streamed piece ([Choosing `delay_s`](test.yaml.md#choosing-delay_s)). |
| `device_map` | `transformers` | `"auto"` | `device_map` for `from_pretrained`, on CUDA only. |
| `merge_adapter` | `transformers` | `false` | Fold the LoRA adapter into the base weights at load. |
| `gguf_filename` | `llamacpp` | `"*Q4_K_M.gguf"` | Filename pattern for repo downloads. The `cpu-only` profile sets `*q4_k_m.gguf`. |
| `strict_openai` | `openai_compat`, `llamacpp_server` | `false` | Send only standard OpenAI fields. This drops `top_k`, `min_p`, `repetition_penalty`, `chat_template_kwargs` (the thinking toggle) and the thinking budget; `presence_penalty`, `stop` and `seed` are still sent. Use it for strict hosted APIs, never for llama-server. |

### `generation`

The decoding defaults. `null` means "not sent; the engine's own default applies".

| Field | Default | Effect and notes |
|---|---|---|
| `max_tokens` | `1024` | Output cap per reply, thinking included. The agent reserves it, up to half the window, when it trims history, so raising it can shrink the history budget. If the trimmed prompt leaves less room than `max_tokens`, the request's cap is cut to the room left (at least 1), so prompt plus reply stay within `max_context_length` by the backend's token count. |
| `temperature` | `0.7` | `0` is greedy. With transformers, greedy decoding skips every sampling knob below, `repetition_penalty` included. |
| `top_p` | `0.95` | Nucleus sampling. |
| `top_k` | `50` | `0` disables it. The OpenAI-protocol backends send it only when > 0. llama.cpp's own default is 40, so this default changes llama-server's sampling. |
| `repetition_penalty` | `1.05` | Penalizes repeated tokens; `1.0` is off. Quality degrades above ~1.2. The OpenAI-protocol backends send it only when ≠ 1.0, as `repetition_penalty` (`repeat_penalty` for llama-server). |
| `min_p` | `null` | Drops tokens below `min_p` × p(top token). `null` is not sent, and llama.cpp then uses 0.05, so set `0.0` explicitly to turn it off. transformers applies it only when sampling and > 0. |
| `presence_penalty` | `null` | A flat penalty on tokens already in the output. Sent by the OpenAI-protocol backends (even in strict mode) and by `llamacpp`; transformers ignores it. |
| `stop` | `[]` | Stop strings; the OpenAI-protocol backends send the first 4. llama-server and `llamacpp` match them against the raw output, thinking included; transformers matches only the answer. |
| `seed` | `null` | Reproducible sampling where the engine supports it. transformers seeds torch globally; `llamacpp` uses `-1` (random) for `null`. |
| `thinking` | `null` | Reasoning on or off for hybrid models; `null` leaves it to the chat template and skips the `thinking_mode` preset. transformers passes `enable_thinking` to the chat template. `llamacpp_server` and `openai_compat` send `chat_template_kwargs.enable_thinking`. `echo` emulates thinking. `llamacpp` ignores the setting, but still splits inline `<think>` text into reasoning. |
| `thinking_budget` | `null` | Maximum thinking tokens; after that the model is told to answer. `null` is unlimited. Enforced only by `llamacpp_server` and `echo`. |

### `thinking_mode`

A sampling preset that is applied over `generation` only while thinking is on. Reasoning models are tuned with different settings per mode (Qwen3.6: temperature 1.0 thinking, 0.7 not) and need far more output tokens when thinking. A single flat preset cannot serve both.

- **Fields.** `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty` and `thinking_budget`, with the same ranges as in `generation`.
- **Defaults.** All are `null`, which falls through to `generation`, so with the defaults thinking changes no sampling value.
- **Not in the preset.** `thinking`, `stop` and `seed`. Unknown keys are rejected.

[`qwen36.yaml`](qwen36.yaml.md) fills it in with the vendor's thinking preset.

### `agent`

| Field | Default | Effect and notes |
|---|---|---|
| `system_prompt` | `"You are Kumaru, a precise and helpful assistant. …"` | The first part of the system message. Kumaru always appends its operating rules (`QUALITY_DIRECTIVES` in `src\kumaru\agent\prompt.py`) and today's date. Enabled tools are not listed here: their declarations go with each request, and the model's chat template renders them. An empty prompt still gets the rules and the date. Every nonblank client system message is appended after all of this, in the order sent, separated by blank lines. |
| `max_history_tokens` | `6144` | History budget per request: `min(max_history_tokens, max(512, context − min(max_tokens, context // 2) − 128 − tools))`, using the request's effective `max_tokens`. `tools` estimates the tool declarations (their JSON length ÷ 3, plus 150), and is 0 with no tools enabled. The reserve is capped at half the window because `max_tokens` is a ceiling, not a forecast. With the defaults that is `min(6144, 7040)` = 6,144. Tokens are counted by the backend: a real tokenizer for `transformers` and `llamacpp`, and a characters ÷ 3.6 estimate for the others. |
| `memory_policy` | `sliding_window` | What to drop when the budget is exceeded. `sliding_window` keeps the system message plus the newest messages that fit, and always at least the newest two. `drop_oldest` removes messages from the front with no floor, and can drop the current question. `none` fails the request with HTTP 413 (`Conversation needs N tokens but the budget is B.`). |
| `enabled_tools` | `[]` | Tools the model may call. Built in: `current_time` (read-only) and the project-folder tools `list_files`, `read_file`, `write_file`, `edit_file` and `run_command`, which `kumaru chat --workspace DIR` adds for you. Their declarations go with each request and the agent runs the calls, so the backend must call tools: `llamacpp_server`, `openai_compat` or `echo`. On this profile's `transformers` (and on `llamacpp`) a non-empty list fails before the model loads: `The transformers backend cannot call tools, and tools are enabled (...)`. Only `kumaru chat` asks you before each change and command; `serve`, the web UI, `eval` and `bench` never offer the tools that change things or use the project folder ([tools.md](../src/kumaru/agent/tools.md)). |
| `max_tool_iterations` | `10` | Rounds of tool calls per message, 1–100. Calls after the last round are not run, and the reply ends with `[Stopped after 10 tool rounds (agent.max_tool_iterations). Reply 'continue' to let it keep going.]`. |
| `workspace.root` | `null` | The project folder for the file and command tools; `kumaru chat --workspace DIR` sets it. The file tools refuse every path outside it ([workspace.md](../src/kumaru/agent/workspace.md)). |
| `workspace.allowed_commands` | `[python, node, npm, npx, dotnet]` | Programs `run_command` may start, by bare name. Case and `.exe`, `.com`, `.cmd` or `.bat` are ignored, and duplicates dropped. A list replaces the default whole, so repeat the names you keep ([commands.md](../src/kumaru/agent/commands.md)). |
| `workspace.command_timeout_s` | `300` | Time limit per command. At the limit the program and everything it started are stopped. The model can ask for less, never more. |
| `workspace.max_output_chars` | `16000` | Most characters one tool call returns to the model. Longer command output keeps its start and end; `read_file` stops there and says which `offset` continues. A result whose chat control tokens were escaped gets a note of about 400 characters after that ([untrusted.md](../src/kumaru/agent/untrusted.md)). |
| `workspace.max_file_bytes` | `262144` | Largest file `read_file` reads or `write_file` and `edit_file` write (256 KB). |
| `workspace.max_list_entries` | `500` | Most entries per `list_files` call. |

### `server`

| Field | Default | Effect and notes |
|---|---|---|
| `host` | `127.0.0.1` | Bind address. `0.0.0.0` exposes the API and UI to your network, so set `api_key` and `cors_origins` first. CLI `--host`. |
| `port` | `8080` | CLI `-p`/`--port`. |
| `cors_origins` | `["*"]` | Browser origins allowed to call the API, with credentials allowed. `["*"]` lets any web page open in your browser drive the local API, `POST /v1/backend` included. The bundled UI is same-origin and needs no entry, so `[]` is safe for it. |
| `max_concurrent_requests` | `8` | Generations admitted at once, through an asyncio semaphore around every completion and stream. Further requests wait; there is no 429. Keep it ≥ `llama_server.parallel`. It must be ≥ 1, but nothing enforces that: `0` makes every generation wait forever, and a negative value fails at startup. `bench run` uses `--concurrency` instead. |
| `request_timeout_s` | `600` | Not read. The wait for a loading backend is fixed at 600 s, after which requests get 503 `Backend is still loading.` |
| `serve_ui` | `true` | Serves the web UI at `/` and `/static`. Set `false` for an API-only server. |
| `api_key` | `null` | When set, `POST /v1/chat/completions` and `POST /v1/backend` require `Authorization: Bearer <api_key>` and otherwise answer 401 `Invalid or missing API key.`. `GET /v1/models`, `/v1/info`, `/healthz` and `/metrics` stay open; `/v1/info` shows the `backend` (without its key), `generation`, `thinking_mode` and `agent` settings, the system prompt and `agent.workspace` included. The bundled UI sends no key, so it stops working when a key is set. |

### `telemetry`

| Field | Default | Effect and notes |
|---|---|---|
| `log_level` | `INFO` | Applied by `kumaru serve` only (CLI `--log-level`). Every other command reads `$env:KUMARU_LOG_LEVEL`. One of `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`, in any case; `WARN` and `FATAL` are accepted as aliases. It is validated at load, so a typo fails every command that loads the config, not only `serve` ([Validation](#validation)). |
| `log_format` | `console` | `json` writes one JSON object per line to stderr; any other value gives console lines. `kumaru serve` only; other commands read `$env:KUMARU_LOG_FORMAT`. |
| `metrics_enabled` | `true` | Not read. `/metrics` is always served. |
| `trace_file` | `null` | Not read. No request trace is written. |

## How to use it

Commands assume the venv is active; otherwise call `.\.venv\Scripts\kumaru.exe`.

### Inspect the resolved configuration

```powershell
kumaru config                        # code defaults
kumaru config -c default             # this file: the same except models_dir
kumaru config -c qwen36 -b echo      # another profile with its backend swapped
```

`kumaru config` takes only `-c`, `-b` and `-m`, and never loads a model. It prints both API keys in clear text, so redact the output before you share it.

### Run it

```powershell
.\scripts\setup.ps1 -Gpu             # CUDA torch, transformers, peft
kumaru chat -c default -q "Explain KV caching in two paragraphs."
kumaru serve -c default              # API and UI on http://127.0.0.1:8080
```

### Check precedence without a model

The echo backend prints the values that reached it:

```powershell
kumaru chat -c test -q hi                    # ... temperature=0.0  (profile)
$env:KUMARU_GENERATION__TEMPERATURE = '0.3'
kumaru chat -c test -q hi                    # ... temperature=0.3  (environment beats profile)
kumaru chat -c test -t 0.5 -q hi             # ... temperature=0.5  (flag beats environment)
Remove-Item Env:KUMARU_GENERATION__TEMPERATURE
```

### Override with environment variables

The form is `KUMARU_<SECTION>__<FIELD>`. Use a double underscore between levels; single underscores belong to the names. Names are case-insensitive.

```powershell
$env:KUMARU_BACKEND__MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
$env:KUMARU_THINKING_MODE__TEMPERATURE = '0.6'
$env:KUMARU_BACKEND__LLAMA_SERVER__N_CPU_MOE = '18'
$env:KUMARU_BACKEND__EXTRA__DELAY_S = '0.05'
kumaru config -c test                        # check what won
Get-ChildItem Env:KUMARU_* | Remove-Item     # clear them all
```

Rules:
- **Some names are skipped silently.** That covers names without `__` (such as `KUMARU_LOG_LEVEL`, which the logger reads directly) and unknown sections (`KUMARU_BACKNED__MODEL`).
- **A misspelt field in a known section fails** like a YAML typo, except in `generation`, which ignores it.
- **`setx` persists a variable for new shells only**, not for the current one.

Values are parsed as YAML:

| You set | Kumaru gets | Consequence |
|---|---|---|
| `'0.3'`, `'18'` | float, int | |
| `'true'`/`'yes'`/`'on'`, `'false'`/`'no'`/`'off'` | `True`, `False` | `flash_attn` maps them back to `on`/`off`. A string field such as `api_key` fails unless quoted: `"'yes'"` |
| `'null'` | `None` | Clears an optional field |
| `'["--fit-target", "8192"]'`, `'["###", "</s>"]'` | list | The only way to set a list |
| `'###'` | the string `###` | `stop` fails: `Input should be a valid list` |
| `'12345'` | int | `api_key` fails: `Input should be a valid string`. Set `'"12345"'` |
| `'Be terse. #1 rule'` | `'Be terse.'` | ` #` starts a YAML comment. Wrap the value in quotes: `"'Be terse. #1 rule'"` |
| `'Rule: be terse'` | a mapping | `system_prompt` fails. Quote it as above |
| `D:\kumaru\models` | unchanged | Windows paths are safe |

### Write a profile

Keep only the fields that differ from the code defaults:

```yaml
# configs\my-profile.yaml
backend:
  model: Qwen/Qwen2.5-1.5B-Instruct
  max_context_length: 16384
generation:
  temperature: 0.3
agent:
  max_history_tokens: 12288
```

Run `kumaru config -c my-profile`, then `kumaru serve -c my-profile`. Every omitted field, here `generation.max_tokens` (1,024) for example, comes from the code.

### Override per request

The request body of `POST /v1/chat/completions` overrides `generation` and `thinking_mode` for one reply:
- `max_tokens` (or `max_completion_tokens`), `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`;
- `stop` (a string or a list), `seed`, `thinking_budget`;
- thinking, as `thinking`, `chat_template_kwargs.enable_thinking` or `reasoning_effort` (`"none"` means off, any other value on), checked in that order.

The ranges match the config's, except that the request's `max_tokens` has no upper bound. `model` is ignored.

```powershell
$body = @{ messages = @(@{ role = 'user'; content = 'hi' }); temperature = 0.2; thinking = $false } | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/chat/completions -ContentType application/json -Body $body
```

The web UI sends `thinking` with every message, plus `thinking_budget` when Think is on and a budget is chosen. Its Think toggle starts on only when `generation.thinking` is `true`. With **Use the model's tuned settings** checked, the default, it sends no sampling values, so the profile's presets decide. Unchecked, its temperature, top-p and max-tokens sliders override them.

## Tuning it

1. **The model doesn't fit (transformers).** Lower `backend.max_context_length`, then set `load_in_4bit: true` ([`low-vram`](low-vram.yaml.md)), then use a smaller `model` or a GGUF through `llamacpp_server` ([`qwen25-7b-gguf`](qwen25-7b-gguf.yaml.md)).
2. **History is forgotten too early.** Raise `agent.max_history_tokens` and `backend.max_context_length` together. The budget never exceeds `context − min(max_tokens, context // 2) − 128`.
3. **Thinking models.** Set `generation.thinking` explicitly, put the vendor's thinking preset in `thinking_mode`, and cap long thoughts with `thinking_budget` on `llamacpp_server` ([`qwen36`](qwen36.yaml.md)).
4. **Reproducible output.** Use `temperature: 0`, or a fixed `seed` on engines that honour it.
5. **Loops on quantized models.** Use `presence_penalty` (1.5 in both of Qwen's presets in `qwen36`) rather than pushing `repetition_penalty` past ~1.2.
6. **Serving on the LAN.** Set `server.host: 0.0.0.0` together with `server.api_key` and a restricted `cors_origins`. The bundled UI cannot send the key.
7. **Concurrency with llama-server.** Keep `server.max_concurrent_requests` ≥ `llama_server.parallel` so that every slot can be busy. Extra admitted requests queue inside llama-server.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Invalid configuration: … Extra inputs are not permitted` | A misspelt key in YAML or in a `KUMARU_*` name | The error names the path, e.g. `backend.modle` |
| A `generation` value has no effect | `generation` ignores misspelt keys, and a misspelt section in a variable name (`KUMARU_GENERATON__…`) is skipped | Compare with `kumaru config` |
| `Input should be a valid dictionary or instance of …` | A section header with every key commented out is `null` | Delete the header, or keep one key |
| `Input should be a valid string` for `api_key` or another string field | YAML read a digits-only value as an int, or `yes`/`no`/`on`/`off` as a boolean; variables are parsed the same way | Quote it: `"12345"` in YAML, `'"12345"'` in a variable |
| `Input should be 'on', 'off' or 'auto'` for `flash_attn` | Another value, e.g. `1` or `enabled` | `on`, `off` or `auto` |
| `error: Config file not found: x` | Neither an existing path nor a profile in `configs\` | Use a name from the `Available profiles:` hint |
| `error: Unknown backend: 'Echo'.` | Backend names are case-sensitive | Use a name from the hint |
| `generation.temperature` or `.max_tokens` is ignored while thinking | `thinking_mode` beats `generation` when thinking is on | Set the field in `thinking_mode` (`$env:KUMARU_THINKING_MODE__TEMPERATURE`), or pass `-t`/`--max-tokens`, which are request fields |
| The model thinks, but with the non-thinking sampling | `generation.thinking: null`: the template thinks by default, but the preset needs an explicit `true` | Set `generation.thinking`, or send `thinking: true` |
| `telemetry.log_level: DEBUG` changes nothing for `chat` or `bench run` | Only `kumaru serve` applies `telemetry` | `$env:KUMARU_LOG_LEVEL = 'DEBUG'` |
| `Invalid configuration: … telemetry.log_level … Input should be 'DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'` | A level name Kumaru doesn't know, such as `verbose`, in YAML, `KUMARU_TELEMETRY__LOG_LEVEL` or `--log-level` | One of the five names, in any case; `WARN` and `FATAL` also work |
| `request_timeout_s`, `metrics_enabled` or `trace_file` has no effect | Not implemented | — |
| `The transformers backend cannot call tools, and tools are enabled (agent.enabled_tools, or kumaru chat --workspace).` | `agent.enabled_tools` or `--workspace` on a backend without tool calling: this profile's `transformers`, or `llamacpp` | `-c qwen36` (`llamacpp_server`), or `openai_compat` with a server that parses tool calls |
| The log says `not offering write_file, ...: they need approval, and nobody here can give it` | `serve`, the web UI, `eval` and `bench` cannot ask you before a change or a command, so they withhold those tools | `kumaru chat --workspace DIR` |
| `Value error, 'C:/tools/node.exe' is not a bare program name (no paths or spaces)`, or `... is a shell or launcher, which could run any command` | A path, a space or a shell in `agent.workspace.allowed_commands` | List each program by bare name (`node`), and never a shell |
| HTTP 413 `Conversation needs N tokens but the budget is B.` | `memory_policy: none` and the history is over budget | `sliding_window`, or a larger budget |
| HTTP 503 `Backend is still loading.` | The backend took more than 600 s to load, e.g. during the first download | Retry once it has loaded; the wait is fixed |
| Every generation hangs, with nothing in the log | `server.max_concurrent_requests: 0` | ≥ 1, and ≥ `llama_server.parallel` |
| CUDA out of memory (transformers) | 16-bit 7B weights plus the KV cache exceed VRAM | Lower `max_context_length`, set `load_in_4bit: true`, or switch to `low-vram` or `qwen25-7b-gguf` |
| `device: cuda:1` still loads on GPU 0 | CUDA placement uses `device_map`, which ignores the index | `$env:CUDA_VISIBLE_DEVICES = '1'`, or `backend.extra.device_map` |
| The UI gets 401 `Invalid or missing API key.` | `server.api_key` is set, and the UI sends no `Authorization` header | Call the API with the header, or run without a key on localhost |
| Another web page can call your local API | `cors_origins: ["*"]` with credentials allowed | `cors_origins: []`, or list your own origins |
| API keys appear in `kumaru config` output | It prints the resolved config verbatim | Redact before you share it |

## Related files

- `src\kumaru\core\config.py`: the schema, `load_config` and `env_overrides` ([config.md](../src/kumaru/core/config.md)).
- `src\kumaru\core\types.py`: `GenerationParams` and its `merged()` ([types.md](../src/kumaru/core/types.md)).
- `src\kumaru\agent\chat.py`: `resolve_params` and the history budget ([chat.md](../src/kumaru/agent/chat.md)). `memory.py` holds the policies ([memory.md](../src/kumaru/agent/memory.md)), `prompt.py` the system prompt ([prompt.md](../src/kumaru/agent/prompt.md)), `tools.py` the tools ([tools.md](../src/kumaru/agent/tools.md)), `workspace.py` the project-folder tools ([workspace.md](../src/kumaru/agent/workspace.md)) and `commands.py` `run_command` ([commands.md](../src/kumaru/agent/commands.md)).
- `src\kumaru\cli.py`: the flags ([cli.md](../src/kumaru/cli.md)).
- `src\kumaru\server\app.py`, `routes_chat.py` and `schemas.py`: CORS, admission, auth and the request fields ([app.md](../src/kumaru/server/app.md), [routes_chat.md](../src/kumaru/server/routes_chat.md), [schemas.md](../src/kumaru/server/schemas.md)).
- `src\kumaru\core\logging.py`: `KUMARU_LOG_LEVEL` and `KUMARU_LOG_FORMAT` ([logging.md](../src/kumaru/core/logging.md)).
- The backends: [transformers_backend.md](../src/kumaru/backends/transformers_backend.md), [llamacpp_server.md](../src/kumaru/backends/llamacpp_server.md), [llamacpp_backend.md](../src/kumaru/backends/llamacpp_backend.md), [openai_compat.md](../src/kumaru/backends/openai_compat.md) and [echo.md](../src/kumaru/backends/echo.md).
- `ui\app.js`: what the UI sends ([app.js.md](../ui/app.js.md)).
- `docs\BENCHMARKS.md`: scores measured with these profiles ([BENCHMARKS.md](../../BENCHMARKS.md)).
