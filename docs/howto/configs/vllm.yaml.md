# `configs/vllm.yaml`

> OpenAI-compatible Kumaru profile that forwards inference to an external vLLM server for high-throughput serving.

**Read this when:** one Transformers process is not enough and you want continuous batching, PagedAttention, or multi-GPU serving through vLLM.

---

## What it does

This profile sets `backend.name: openai_compat` and sends requests to `http://127.0.0.1:8000/v1`. Kumaru keeps the UI, API compatibility, config, auth, and metrics surface while vLLM owns model loading, batching, KV-cache paging, and GPU utilization.

## When to use this profile

Use it for service-style deployments or multi-user workloads where throughput matters. The source comments state that vLLM does not have a native Windows build; start vLLM on Linux/WSL or another host first.

Do not use this profile until the vLLM OpenAI-compatible endpoint is running and serving the same model id.

## Every field explained

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `backend.name` | string | `openai_compat` | Uses an OpenAI-compatible remote API. | Required for vLLM/Ollama/TGI-like endpoints. |
| `backend.model` | string | `Qwen/Qwen2.5-7B-Instruct` | Model name sent in requests and shown in metadata. | Must match the vLLM model. |
| `backend.base_url` | string | `http://127.0.0.1:8000/v1` | Remote API root. | Change to the vLLM host/port; keep `/v1`. |
| `backend.api_key` | string/null | `null` | Remote API bearer key if required. | Use environment variables for real secrets. |
| `backend.timeout_s` | float | `300` | Timeout for forwarded requests. | Raise for long generations; lower for strict latency. |
| `backend.max_context_length` | int | `16384` | Kumaru-side context cap. | Match or stay below vLLM `--max-model-len`. |
| `generation.max_tokens` | int | `2048` | Reply cap sent to vLLM. | Lower for latency; keep within model length. |
| `agent.max_history_tokens` | int | `12288` | Conversation history budget. | Leave room for generated tokens under 16k. |
| `server.port` | int | `8080` | Kumaru HTTP/UI port. | Change if occupied. |
| `server.max_concurrent_requests` | int | `64` | Kumaru admission limit before forwarding. | Higher than local GPU profiles because vLLM handles batching. |
| `telemetry.log_format` | string | `json` | Structured logs. | Good default for service deployment. |

## How to use it

Start vLLM first. On Linux/WSL:

```powershell
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.90
```

Then run Kumaru:

```powershell
kumaru serve -c vllm
```

Test through Kumaru:

```powershell
kumaru chat -c vllm -q "Explain PagedAttention in one paragraph."
```

Inspect config:

```powershell
kumaru config -c vllm
```

Scaling is a vLLM concern:

```powershell
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --tensor-parallel-size 4
```

## Tuning it

1. Match `backend.max_context_length` to vLLM `--max-model-len`.
2. Tune vLLM `--gpu-memory-utilization` and batching before changing Kumaru.
3. Raise or lower `server.max_concurrent_requests` to protect the vLLM server.
4. Lower `generation.max_tokens` if tail latency is high.
5. Use `telemetry.log_format: json` with centralized log collection.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Connection refused | vLLM is not running at `base_url`. | Start vLLM or correct host/port. |
| Model mismatch errors | `backend.model` differs from vLLM's served model. | Make the names match. |
| Long prompts fail | Kumaru context exceeds vLLM `--max-model-len`. | Lower `backend.max_context_length` or restart vLLM with a larger limit. |
| Authentication failure | vLLM endpoint requires a key but `backend.api_key` is null. | Set the key via config/env. |
| High throughput but slow first token | Large batches improve throughput at latency cost. | Tune vLLM batching/admission and lower Kumaru concurrency if needed. |

## Related files

- `configs/local-gpu.yaml` — simpler one-request-at-a-time local GPU profile.
- `src/kumaru/server/routes_chat.py` — Kumaru API surface that forwards chat calls.
- `src/kumaru/core/config.py` — `openai_compat` backend fields.

