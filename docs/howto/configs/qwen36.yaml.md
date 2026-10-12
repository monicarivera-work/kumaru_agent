# `configs\qwen36.yaml`

> Kumaru's default "smart" profile: Qwen3.6-35B-A3B (a mixture-of-experts model) as a ~4-bit GGUF, served by a managed `llama-server`. It runs on one 24 GB GPU with a 131,072-token window and two vendor sampling presets, one for plain answers and one for thinking.

**Read this when:**
- you run Kumaru on an RTX 3090/4090-class card;
- you want thinking (reasoning) mode;
- you need to fit the model next to a game or onto a smaller GPU;
- you want to know why a sampling value is what it is.

---

## What it does

The profile selects the `llamacpp_server` backend and the GGUF `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS`. On first use that file (17.73 GB) is downloaded into `backend.models_dir`. The profile then sets:
- a 131,072-token context;
- 4 batched request slots sharing one KV-cache pool;
- Qwen's two recommended sampling presets: `generation` for answers without thinking, which is the default, and `thinking_mode` for when thinking is on.

Every field it does not set falls back to the code defaults, which `configs\default.yaml` documents. For example, `n_gpu_layers: -1` lets llama.cpp fit the GPU offload automatically.

### Why this model

- **Mixture of experts.** The model has 35B parameters in total. For each token, a router picks 8 of 256 routed experts plus 1 shared expert, so only ~3B parameters do the work. Knowledge scales with total parameters and speed with active ones. Measured on an RTX 3090 with nothing else on the GPU: ~110 tok/s single-stream.
- **131k context on one card.** Only 10 of the 40 layers use full attention, which needs a KV cache that grows with context. The other 30 are Gated DeltaNet linear-attention layers whose state has a fixed size. The KV cache is therefore 20 KiB/token (10 layers × 2 KV heads × 256 dims × 2 for K and V × 2 bytes), or 2.5 GiB at 131,072 tokens.
- **Why llama.cpp.** Gated DeltaNet is a new architecture. llama.cpp ships support for new architectures and kernels weeks before the Python bindings do, hence the [`llamacpp_server`](../src/kumaru/backends/llamacpp_server.md) backend.

### VRAM budget (RTX 3090, 24 GB)

| Component | Size |
|---|---|
| Weights, UD-IQ4_XS (17.73 GB file, ~4 bits/param) | 16.5 GiB |
| KV cache, 20 KiB/token × 131,072 | 2.5 GiB |
| Compute buffers, DeltaNet state, CUDA context | ~0.7 GiB |
| **llama-server total, measured** | **~19.7 GiB** |
| Left for Windows and other apps | ~4 GiB |

The KV cache scales linearly with `--context`: 1.25 GiB at 65,536 and 0.625 GiB at 32,768. Each layer whose experts move to system RAM (`n_cpu_moe`) frees ~0.37 GiB. A first estimate is therefore `19.7 − 0.37 × n_cpu_moe − KV saved` GiB. For the GPU-sharing recipe below this gives 11.8 GiB; 11.9 GiB was measured.

## When to use this profile

Use it when:
- you have a 24 GB GPU and want the best answers Kumaru can run locally;
- the work is maths, code or logic, where thinking mode pays off;
- you need long documents or long conversations (65,536 tokens of history by default);
- several clients or benchmark workers share the model (4 batched slots).

Use something else in these cases:
- **An 8 GB GPU, or an A/B baseline on the same engine:** [`qwen25-7b-gguf`](qwen25-7b-gguf.yaml.md).
- **Install or UI checks without a model:** [`test`](test.yaml.md).
- **No CUDA GPU:** a 35B model on CPU alone is not what this profile was tuned or measured for.

## Every field explained

| Field | Value | Code default | Why this value |
|---|---|---|---|
| `backend.name` | `llamacpp_server` | `transformers` | Needs llama.cpp for Gated DeltaNet. Kumaru starts, health-checks and stops the server itself. |
| `backend.model` | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS` | `Qwen/Qwen2.5-7B-Instruct` | Unsloth Dynamic IQ4_XS: the largest quant that still leaves room for a 131k KV cache on 24 GB. Spell the `UD-` prefix: a bare `IQ4_XS` would choose a plain `…-IQ4_XS.gguf` if the repo has one, because the shortest matching name wins. |
| `backend.max_context_length` | `131072` | `8192` | Qwen advises ≥ 128k so long reasoning traces are never cut off. The cost is 2.5 GiB of KV cache. Becomes `--ctx-size`. CLI: `--context`. |
| `backend.llama_server.parallel` | `4` | `4` | Four slots. Continuous batching runs concurrent requests in one forward pass. |
| `backend.llama_server.kv_unified` | `true` | `true` | One KV pool for all slots. Without it each slot would get 131,072 / 4 = 32,768 tokens. |
| `backend.llama_server.flash_attn` | `"on"` | `"on"` | Flash attention. Quotes are optional: YAML reads a bare `on`/`off` as a boolean, and the field maps it back to `"on"`/`"off"`. Flash attention is also required if you quantize the V cache. |
| `generation.thinking` | `false` | `null` | Fast by default. Turn it on per message (see below). |
| `generation.max_tokens` | `32768` | `1024` | Vendor recommendation. It is a cap, not a target: the model stops at end-of-turn. It also reserves room in the window (see `agent.max_history_tokens`). |
| `generation.temperature` / `top_p` / `top_k` | `0.7` / `0.8` / `20` | `0.7` / `0.95` / `50` | Qwen's recommended instruct (non-thinking) preset. `top_k` must be set: the code default 50 would be sent, and llama.cpp's own default is 40. |
| `generation.min_p` | `0.0` | `null` | Must be explicit. `null` means "not sent", and llama.cpp's own default is 0.05, not the vendor's 0. |
| `generation.presence_penalty` | `1.5` | `null` | Curbs the repetition loops that quantized models fall into. A standard OpenAI field, so it is always sent when set. |
| `generation.repetition_penalty` | `1.0` | `1.05` | Off. The vendor preset uses `presence_penalty` instead. Kumaru sends `repeat_penalty` only when the value is ≠ 1.0, so nothing is sent. |
| `thinking_mode.max_tokens` | `32768` | `null` | Qwen's thinking preset: thinking plus answer share this cap. |
| `thinking_mode.temperature` / `top_p` / `top_k` / `min_p` | `1.0` / `0.95` / `20` / `0.0` | `null` | Qwen's thinking preset. It is applied over `generation` only while thinking is on. Precedence: `generation` < `thinking_mode` < per request. |
| `thinking_mode.presence_penalty` / `repetition_penalty` | `1.5` / `1.0` | `null` | Same rationale as in `generation`. |
| `agent.max_history_tokens` | `65536` | `6144` | History budget. The effective budget is `min(65536, max(512, context − min(max_tokens, context // 2) − 128))`: 65,536 at 131,072, 32,640 at `--context 65536` and 16,256 at `--context 32768`. |
| `agent.memory_policy` | `sliding_window` | `sliding_window` | Keeps the system prompt and the newest turns. |
| `server.host` / `server.port` | `127.0.0.1` / `8080` | same | Local only. Add `server.api_key` before binding to a LAN address. |
| `server.max_concurrent_requests` | `8` | `8` | Kumaru admits 8 generations at a time and makes the rest wait. llama-server runs 4 and queues the other 4 internally. |
| `telemetry.log_level` | `INFO` | `INFO` | `DEBUG` also logs the full llama-server command line. Applied by `kumaru serve` only (`--log-level DEBUG`); for `chat` and `bench run`, set `$env:KUMARU_LOG_LEVEL = 'DEBUG'`. |

`thinking_mode.thinking_budget` is not set, so thinking is unlimited unless the request, `--thinking-budget`, or `generation.thinking_budget` sets a budget.

The resolved llama-server command line (`build_server_args`; the model path depends on `models_dir`, and the port is free-picked):

```text
llama-server.exe --model <models_dir>\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --host 127.0.0.1 --port <free> --ctx-size 131072 --parallel 4 --flash-attn on --jinja --alias unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS --no-webui --log-colors off --kv-unified
```

## How to use it

### First run

```powershell
.\scripts\get-llama-server.ps1                      # once: installs tools\llama.cpp\
setx KUMARU_BACKEND__MODELS_DIR D:\kumaru\models    # optional, new shells only; default ~\.cache\kumaru\models
kumaru chat -c qwen36 -q "hi"                       # downloads 17.73 GB, starts llama-server, answers
kumaru serve -c qwen36                              # UI at http://127.0.0.1:8080
```

Download with `kumaru chat` first. `kumaru serve` opens its port at once and loads in the background, and a chat request waits only 600 s for the backend before it gets 503 "Backend is still loading." The CLI waits as long as the download takes. An interrupted download resumes on the next start. Gated repos or rate limits need `$env:HF_TOKEN`; see [llamacpp_server.md](../src/kumaru/backends/llamacpp_server.md#failure-modes).

Inspect without loading anything:

```powershell
kumaru config -c qwen36          # resolved config, env overrides included
```

### Turn thinking on

Thinking is off by default. These are the ways to turn it on:

| Where | How |
|---|---|
| Web UI | The **Think** button under the message box. It starts at the profile's `generation.thinking`. The **thinking budget** slider in the Thinking panel ranges from 256 to 24,576 tokens, or unlimited. With "Use the model's tuned settings" checked (the default), the UI sends only the toggle and the budget, so the profile's thinking preset applies. Unchecked, the UI also sends temperature, top-p and max tokens. |
| CLI | `kumaru chat -c qwen36 --think -q "How many primes are below 100?"`, optionally with `--thinking-budget 2048`. `--no-think` forces it off. |
| Benchmarks | `kumaru bench run gsm8k -c qwen36 --think --thinking-budget 4096` |
| API, per request | `"thinking": true`, or `"chat_template_kwargs": {"enable_thinking": true}`, or `"reasoning_effort": "<anything but none>"`, checked in that order. Add `"thinking_budget": N` for a budget. |
| Profile default | Set `generation.thinking: true` in a copy of the profile, or `$env:KUMARU_GENERATION__THINKING = 'true'`. |

```powershell
$body = @{
  messages        = @(@{ role = 'user'; content = 'How many primes are below 100?' })
  thinking        = $true
  thinking_budget = 2048
} | ConvertTo-Json -Depth 4
$r = Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
$r.choices[0].message.reasoning_content    # the thinking
$r.choices[0].message.content              # the answer
```

How thinking behaves:
- **Where it appears.** Streaming responses carry the thinking in `delta.reasoning_content`.
- **Not kept in history.** It is never sent back to the model in later turns.
- **Budget.** When the budget runs out, llama-server injects `llama_server.reasoning_budget_message`, and the model answers.
- **`max_tokens` covers both parts.** It caps thinking plus answer, so a long trace can use up the cap before the answer starts. The UI then shows a note.
- **The preset beats `generation`, not your flags.** With thinking on, `thinking_mode` overrides `generation` from YAML or `KUMARU_GENERATION__*`. The `-t` and `--max-tokens` flags of `kumaru chat` and `kumaru bench run` are sent per request, like the UI's settings, so they beat the preset. To change the thinking defaults for every request, set the preset: `$env:KUMARU_THINKING_MODE__MAX_TOKENS = '8192'`, `$env:KUMARU_THINKING_MODE__TEMPERATURE = '0.6'`.

### Share the GPU with a game, or use a smaller card

Windows (WDDM) does not fail an allocation when VRAM is full. It spills into shared system memory, and speed collapses. Measured: ~110 tok/s fell to 7 tok/s with the 131k window while a game was running. This recipe was measured next to World of Warcraft (4.2 GiB): 11.9 GiB VRAM, ~27 tok/s, no spill:

```powershell
$env:KUMARU_BACKEND__LLAMA_SERVER__N_CPU_MOE = '18'
kumaru serve -c qwen36 --context 65536
```

For smaller cards, the profile's own starting points (not measured) are `n_cpu_moe: 20` for 16 GB and `30` for 12 GB, combined with a smaller `--context`. To make a setting permanent, copy the profile (e.g. to `configs\qwen36-shared.yaml`), add `n_cpu_moe: 18` under `backend.llama_server`, and run `kumaru serve -c qwen36-shared`.

## Tuning it

1. **Watch for spill.** If Task Manager → GPU → "Shared GPU memory" grows while Kumaru generates, VRAM has spilled. First lower `--context` (131,072 → 65,536 frees 1.25 GiB), then raise `n_cpu_moe` in steps (~0.37 GiB each).
2. **Below `--context 65536`, trade reply room for history.** At most half the window is reserved for the reply, and each reply's `max_tokens` is trimmed to the room the prompt leaves. At `--context 32768` the history budget is 32,768 − 16,384 − 128 = 16,256 tokens. Setting `generation.max_tokens` and `thinking_mode.max_tokens` to 8192 (`$env:KUMARU_GENERATION__MAX_TOKENS = '8192'`, `$env:KUMARU_THINKING_MODE__MAX_TOKENS = '8192'`) raises it to 24,448; add a thinking budget so long traces still end with an answer.
3. **Thinking too slow?** Set a thinking budget (UI slider or `--thinking-budget`) rather than lowering `max_tokens`. A budget ends the thinking cleanly; a low `max_tokens` cuts the answer off.
4. **Keep the sampling values.** `min_p: 0.0`, `top_k: 20` and `presence_penalty: 1.5` are vendor-tuned. Removing a value does not mean "off"; it hands control to llama.cpp's defaults.
5. **Compare models.** Scores and the A/B against `qwen25-7b-gguf` are in [docs\BENCHMARKS.md](../../BENCHMARKS.md).

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `llama-server was not found.` | Binary not installed | `.\scripts\get-llama-server.ps1` |
| `llama-server exited during startup`, with allocation errors in the log tail | Does not fit in free VRAM | Lower `--context`, set `n_cpu_moe`, close other GPU apps |
| Same error, with an unknown-architecture message | llama.cpp build too old for Qwen3.6 | `.\scripts\get-llama-server.ps1 -Version bNNNNN` (newer) |
| 503 "Backend is still loading." on the first run | The 17.73 GB download takes longer than the 600 s request wait | Download first with `kumaru chat -c qwen36 -q "hi"` |
| ~7 tok/s instead of ~110 | VRAM spill next to another GPU app | The GPU-sharing recipe above |
| `Input should be 'on', 'off' or 'auto'` for `flash_attn` | Another value, e.g. `1` or `enabled` | `on`, `off` or `auto` |
| Answer missing after a long thinking trace | `max_tokens` used up while thinking | Set a thinking budget, or raise `max_tokens` |
| Repetition loops | `presence_penalty` removed or set to 0 | Restore `1.5` |
| Think toggle has no effect | `backend.extra.strict_openai: true`, or another backend that ignores `thinking` (e.g. `llamacpp`) | Remove `strict_openai`; use `llamacpp_server` |
| Multi-turn chat forgets earlier turns at a small `--context` | The history budget shrinks with the window: 16,256 tokens at `--context 32768` | Lower `max_tokens` to trade reply room for history (Tuning step 2), or raise `--context` |
| `Could not list files of 'unsloth/Qwen3.6-35B-A3B-GGUF'` | Offline, rate-limited, or no token | `$env:HF_TOKEN`, or download manually and set `backend.model_path` |

## Related files

- `src\kumaru\backends\llamacpp_server.py` — the backend: download, process lifecycle, flag mapping, thinking fields. Its how-to page has the full details.
- `configs\qwen25-7b-gguf.yaml` — 7B baseline on the same engine.
- `configs\default.yaml` — every setting and its default.
- `src\kumaru\agent\chat.py` — `resolve_params` (preset precedence) and the history budget.
- `scripts\get-llama-server.ps1` — installs or upgrades llama-server.
- `docs\BENCHMARKS.md` — measured scores.
