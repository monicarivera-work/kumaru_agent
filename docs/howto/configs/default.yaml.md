# `configs/default.yaml`

> Fully annotated baseline inference profile whose values mirror the Pydantic defaults in `src/kumaru/core/config.py`.

**Read this when:** you need to understand every serving knob, debug config precedence, or create a new inference profile.

---

## What it does

`configs/default.yaml` describes the default Kumaru serving runtime: `transformers` backend, Qwen2.5 7B Instruct, local device autodetection, 8k context, chat-oriented generation defaults, localhost HTTP serving, and console telemetry.

Configuration precedence is:

1. field defaults in `src/kumaru/core/config.py`;
2. this YAML or another `-c/--config` profile;
3. `KUMARU_*` environment variables such as `KUMARU_BACKEND__MODEL`;
4. CLI flags such as `--model`.

Inspect the resolved result with:

```powershell
kumaru config -c default
```

## When to use this profile

Use it as the reference profile or as a development starting point on a machine that can run `Qwen/Qwen2.5-7B-Instruct` through Hugging Face Transformers. For constrained hardware, prefer `low-vram`, `cpu-only`, `local-gpu`, `vllm`, or `test`.

## Every field explained

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `backend.name` | string | `transformers` | Inference engine: `transformers`, `llamacpp`, `openai_compat`, or `echo`. | Use `transformers` for local HF dev, `llamacpp` for GGUF/CPU, `openai_compat` for vLLM/Ollama/TGI, `echo` for tests. |
| `backend.model` | string | `Qwen/Qwen2.5-7B-Instruct` | Hugging Face repo id, local model directory, or model id sent to a remote API. | Pick the checkpoint you intend to serve; for vLLM it must match the server model. |
| `backend.device` | string | `auto` | Local placement for backends that load weights. | `auto` for normal use, `cuda`/`cuda:0` for a GPU, `cpu` for CPU, `mps` on Apple. |
| `backend.dtype` | string | `auto` | Compute dtype for local model execution. | Use `bfloat16` on Ampere+ GPUs, `float16` on older CUDA, `float32` for CPU/debug, `auto` when unsure. |
| `backend.load_in_4bit` | bool | `false` | Enables 4-bit NF4 quantization. | Set `true` to fit 7B models in 6-12 GB VRAM; install `kumaru[quant]`. |
| `backend.load_in_8bit` | bool | `false` | Enables 8-bit quantization. | Use only if supported by the backend and 4-bit is too aggressive. |
| `backend.max_context_length` | int | `8192` | Hard cap on prompt plus generation context. | Lower to save KV-cache memory; raise only when model/backend and VRAM support it. |
| `backend.adapter_path` | string/null | `null` | LoRA adapter directory loaded over the base model. | Set to a trained `runs/.../adapter` directory from `kumaru train`. |
| `backend.trust_remote_code` | bool | `false` | Allows custom model code from model repos. | Keep `false` unless the checkpoint requires it and you trust the source. |
| `backend.attn_implementation` | string/null | `null` | Transformers attention implementation, for example `flash_attention_2`. | Set only after installing the required backend package and testing compatibility. |
| `backend.base_url` | string/null | `null` | OpenAI-compatible API root. | Required for `openai_compat`, for example `http://127.0.0.1:8000/v1`. |
| `backend.api_key` | string/null | `null` | API key for a remote OpenAI-compatible backend. | Use env vars for secrets; do not commit real keys. |
| `backend.timeout_s` | float | `300` | Remote backend request timeout. | Increase for slow long-context generations; lower to fail faster. |
| `backend.model_path` | string/null | `null` | Absolute `.gguf` file path for llama.cpp. | Use when serving a local GGUF file directly. |
| `backend.n_gpu_layers` | int | `-1` | Number of llama.cpp layers to offload to GPU. | `-1` offloads as many as fit; `0` is CPU-only. |
| `backend.n_threads` | int/null | `null` | llama.cpp CPU thread count. | Leave `null` for physical-core default; tune if sharing the host. |
| `generation.max_tokens` | int | `1024` | Maximum generated tokens per response. | Lower for latency/cost; raise for long answers while staying under context. |
| `generation.temperature` | float | `0.7` | Sampling randomness. | `0` for deterministic extraction/evals; `0.7` for chat; higher for brainstorming. |
| `generation.top_p` | float | `0.95` | Nucleus sampling probability mass. | Keep `0.9-0.95` for general chat; lower for stricter outputs. |
| `generation.top_k` | int | `50` | Maximum candidate token count before sampling. | Leave default unless tuning repetition/creativity. |
| `generation.repetition_penalty` | float | `1.05` | Penalty for repeating tokens. | Use `1.05-1.15` for loops; avoid values above about `1.2`. |
| `generation.stop` | list[string] | `[]` | Stop sequences. | Add explicit delimiters for structured protocols. |
| `generation.seed` | int/null | `null` | Sampling seed where backend supports it. | Set for repeatable experiments; leave null for normal chat. |
| `agent.system_prompt` | string | Kumaru helpful-assistant prompt | Base instruction prepended to conversations. | Change for deployment persona or policy; keep direct and testable. |
| `agent.max_history_tokens` | int | `6144` | Token budget for retained conversation history. | Keep `max_history_tokens + max_tokens <= max_context_length`. |
| `agent.memory_policy` | string | `sliding_window` | History trimming behavior: `sliding_window`, `drop_oldest`, or `none`. | Use `sliding_window` generally; `none` only when you want errors instead of trimming. |
| `agent.enabled_tools` | list[string] | `[]` | Reserved tool list. | v0.1 is chat-only; leave empty unless tool layer is enabled. |
| `agent.max_tool_iterations` | int | `4` | Reserved cap for future tool loops. | Leave default while tools are disabled. |
| `server.host` | string | `127.0.0.1` | Bind address. | Use localhost by default; use `0.0.0.0` only with `server.api_key`. |
| `server.port` | int | `8080` | HTTP port. | Change if 8080 is occupied. |
| `server.cors_origins` | list[string] | `["*"]` | Allowed browser origins. | Restrict in production. |
| `server.max_concurrent_requests` | int | `8` | Admission limit around model work. | Lower to avoid OOM; raise for remote/batched backends. |
| `server.request_timeout_s` | float | `600` | HTTP request timeout. | Raise for long generations; lower for strict service SLOs. |
| `server.serve_ui` | bool | `true` | Whether the server serves static UI files. | Set false for API-only deployments. |
| `server.api_key` | string/null | `null` | Optional bearer-token gate for protected endpoints. | Set before exposing beyond localhost. |
| `telemetry.log_level` | string | `INFO` | Logging verbosity. | Use `DEBUG` for troubleshooting, `INFO` normally. |
| `telemetry.log_format` | string | `console` | Log format: console or JSON. | Use `json` for aggregation. |
| `telemetry.metrics_enabled` | bool | `true` | Enables in-process counters exposed at `/metrics`. | Leave enabled unless embedding in a constrained environment. |
| `telemetry.trace_file` | string/null | `null` | JSONL request trace path. | Set for offline analysis; protect traces because prompts may be sensitive. |

## How to use it

Prerequisites:

```powershell
pip install kumaru
```

For quantization or GGUF profiles, install extras named in those profile docs.

Run the server and UI:

```powershell
kumaru serve -c default
```

Open:

```text
http://127.0.0.1:8080
```

Run one terminal prompt without the server:

```powershell
kumaru chat -c default -q "Explain KV caching in two paragraphs."
```

Print the fully merged config:

```powershell
kumaru config -c default
```

Override a field once:

```powershell
kumaru serve -c default --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --port 8081
```

## Tuning it

1. If the model does not fit, lower `backend.max_context_length`, then set `backend.load_in_4bit: true`, then choose a smaller `backend.model`.
2. If replies are too long or slow, lower `generation.max_tokens`.
3. If output is too random, lower `generation.temperature`; if it loops at `0`, raise temperature and use a small repetition penalty.
4. If multi-turn chats lose context too early, raise `agent.max_history_tokens` while keeping room for `generation.max_tokens`.
5. If the server OOMs under load, lower `server.max_concurrent_requests`.
6. If exposing over LAN, set `server.host: 0.0.0.0` and set `server.api_key`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Startup says an extra field is forbidden | YAML key is not in `KumaruConfig`. | Fix the typo; schema uses `extra="forbid"`. |
| CUDA OOM on first request | Model, context, batch/concurrency, or dtype exceeds VRAM. | Use `low-vram`, lower context, or reduce concurrency. |
| `kumaru config -c default` differs from the file | Environment variables or CLI flags overrode it. | Check `KUMARU_*` variables and command flags. |
| Remote backend ignores settings | You selected `openai_compat`; some local-only fields do not apply. | Configure the remote engine and use `base_url`, `model`, and timeouts here. |
| UI returns 401/invalid API key | `server.api_key` is set but browser client does not send a bearer token. | Add auth handling to the UI or keep localhost-only with no key. |
| Model repeats itself | Temperature is too low, stop sequences are missing, or penalty is too weak. | Raise temperature, set appropriate `stop`, or tune `repetition_penalty`. |

## Related files

- `src/kumaru/core/config.py` — authoritative serving config schema and merge rules.
- `src/kumaru/core/types.py` — generation parameter schema.
- `src/kumaru/cli.py` — `serve`, `chat`, and `config` flags.
- `configs/local-gpu.yaml`, `configs/low-vram.yaml`, `configs/cpu-only.yaml`, `configs/vllm.yaml`, `configs/test.yaml` — concrete profiles.

