# `configs\qwen25-7b-gguf.yaml`

> Qwen2.5-7B-Instruct as a Q4_K_M GGUF on the same managed `llama-server` engine as `qwen36`. It is the benchmark baseline, and the profile for GPUs with about 8 GB.

**Read this when:**
- you need an A/B baseline against `qwen36`;
- your GPU has ~8 GB of VRAM;
- you want to exercise the real llama-server path with a 4.7 GB download instead of 17.73 GB.

---

## What it does

The profile runs `Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` through the [`llamacpp_server`](../src/kumaru/backends/llamacpp_server.md) backend. That is the model v0.1 used by default (in bf16 via transformers, `configs\local-gpu.yaml`), now as a 4-bit GGUF on the v0.2 engine. The rest of the setup:
- a 32,768-token window, the model's native one;
- 4 batched slots sharing one KV pool;
- Qwen2.5's own `generation_config.json` sampling values;
- no thinking mode.

The Hugging Face repo stores the quant as two shards, `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` and `-00002-of-00002.gguf`, 4.7 GB in total. Both are downloaded into `<models_dir>\Qwen\Qwen2.5-7B-Instruct-GGUF\`, and llama-server receives the first shard.

### Why a baseline on the same engine

A score difference between two runs is attributable to the model only if everything else is equal. That means the same engine, server flags, harness, prompts and question sample. This profile changes only the model and its vendor sampling preset. A Qwen3.6 run against v0.1's transformers profile would also change the engine, quantization and numerics. Results are in [docs\BENCHMARKS.md](../../BENCHMARKS.md).

### VRAM budget

| Component | Size |
|---|---|
| Weights, Q4_K_M (4.7 GB in 2 files) | 4.4 GiB |
| KV cache, 56 KiB/token × 32,768 | 1.75 GiB |
| Total including compute buffers | ~7 GiB (the profile's estimate; not measured) |

The KV cost per token is 2.8 × that of Qwen3.6 (56 vs 20 KiB). The reason: all 28 layers of this dense model use full attention (28 × 4 KV heads × 128 dims × 2 × 2 bytes). Qwen3.6 has only 10 such layers. On an 8 GB card, lower the window: at `--context 16384` the KV cache is 0.875 GiB.

## When to use this profile

Use it when:
- you need a benchmark baseline or A/B partner for `qwen36`;
- the GPU has 8–12 GB of VRAM;
- you want a fast check of the real download, spawn and health path;
- latency matters more than answer quality.

Do not use it when:
- you need thinking. Qwen2.5 has none, so `--think` changes nothing (see below).
- you need context beyond 32,768 tokens;
- you have a 24 GB card and want the best answers. Use [`qwen36`](qwen36.yaml.md).

## Every field explained

| Field | Value | Code default | Why this value |
|---|---|---|---|
| `backend.name` | `llamacpp_server` | `transformers` | Same engine as `qwen36`, so A/B comparisons are fair |
| `backend.model` | `Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` | `Qwen/Qwen2.5-7B-Instruct` | Q4_K_M, the common 4-bit quant. The spec matches both `-0000N-of-00002` shards. |
| `backend.max_context_length` | `32768` | `8192` | The model's native window. Becomes `--ctx-size`. |
| `backend.llama_server.parallel` / `kv_unified` / `flash_attn` | `4` / `true` / `"on"` | same | Identical to `qwen36`, so the server setup is not a variable. The quotes around `"on"` are optional: a bare `on` is a YAML boolean, which the field maps back to `"on"`. |
| `generation.max_tokens` | `4096` | `1024` | Enough for long answers without a thinking phase |
| `generation.temperature` / `top_p` / `top_k` | `0.7` / `0.8` / `20` | `0.7` / `0.95` / `50` | Qwen2.5's `generation_config.json` |
| `generation.min_p` | `0.0` | `null` | Explicit, because llama.cpp's own default is 0.05 |
| `generation.repetition_penalty` | `1.05` | `1.05` | Qwen2.5's value. Sent as llama.cpp's `repeat_penalty`. |
| `generation.presence_penalty` | not set | `null` | Not part of Qwen2.5's preset, so it is not sent |
| `generation.thinking` | not set | `null` | No `chat_template_kwargs` is sent unless a request asks |
| `agent.max_history_tokens` | `24576` | `6144` | `min(24576, max(512, 32768 − min(4096, 32768 // 2) − 128))` = 24,576 tokens of history |
| `agent.memory_policy` | `sliding_window` | same | Keeps the system prompt and the newest turns |
| `server.*`, `telemetry.log_level` | `127.0.0.1:8080`, 8 concurrent, `INFO` | same | Same as `qwen36` |

The resolved command line (verified with `build_server_args`):

```text
llama-server.exe --model <models_dir>\Qwen\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --host 127.0.0.1 --port <free> --ctx-size 32768 --parallel 4 --flash-attn on --jinja --alias Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --no-webui --log-colors off --kv-unified
```

## How to use it

```powershell
.\scripts\get-llama-server.ps1                 # once
kumaru serve -c qwen25-7b-gguf                 # UI at http://127.0.0.1:8080
kumaru chat  -c qwen25-7b-gguf -q "Explain KV caching in two sentences."
kumaru config -c qwen25-7b-gguf                # resolved settings, no model load
```

An 8 GB card:

```powershell
kumaru serve -c qwen25-7b-gguf --context 16384     # KV 0.875 GiB; history budget becomes 12,160
```

A/B against the smart profile. Use the same benchmark, `--limit` and `--seed`, so both runs get the same questions:

```powershell
kumaru bench run gsm8k -c qwen25-7b-gguf --limit 100
kumaru bench run gsm8k -c qwen36 --limit 100
```

**Thinking.** `--think`, the UI's Think button and request-level `thinking` are accepted but change nothing. The flag reaches the chat template as `enable_thinking`, which Qwen2.5's template does not use, and the profile has no `thinking_mode` preset to switch to. Treat this model as non-thinking in comparisons.

## Tuning it

1. **Fitting 8 GB.** Lower `--context` (16384 or 8192) before anything else. The weights already fit, and the KV cache is the variable cost.
2. **Running on CPU.** Set `backend.n_gpu_layers: 0`, which sends `--n-gpu-layers 0`; `info().device` then reports `cpu`. Expect much lower speed. No CPU numbers have been measured for this profile.
3. **Longer answers.** Raise `generation.max_tokens`. The history budget stays 24,576 up to `max_tokens` 8,064. Above that, every token added comes out of it (32,640 − `max_tokens`), down to 16,256 at 16,384; larger values reserve no more, because at most half the window is reserved for the reply.
4. **Keep the sampling values for baselines.** Changing them makes the A/B measure your settings, not the model.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `llama-server was not found.` | Binary not installed | `.\scripts\get-llama-server.ps1` |
| Warning `cached GGUF … is missing parts; fetching them` | A first download was interrupted between the two shards | Nothing: the next start downloads the missing shard (it needs the network) |
| `llama-server exited during startup`, with allocation errors in the log tail, on an 8 GB card | 32k KV cache plus weights plus buffers do not fit | `--context 16384` or lower |
| `--think` makes no difference | Qwen2.5 has no thinking mode | Use `qwen36` |
| A/B differences look like noise | Different `--limit`, `--seed` or settings between runs | Re-run both with identical flags |
| `Input should be 'on', 'off' or 'auto'` for `flash_attn` | Another value, e.g. `1` or `enabled`, in a copied profile | `on`, `off` or `auto` |

## Related files

- `configs\qwen36.yaml` — the smart profile this one is compared against.
- `src\kumaru\backends\llamacpp_server.py` — download, shard matching, process lifecycle.
- `configs\local-gpu.yaml` — the same model in bf16 on transformers (v0.1 path).
- `docs\BENCHMARKS.md` — baseline and comparison scores.
