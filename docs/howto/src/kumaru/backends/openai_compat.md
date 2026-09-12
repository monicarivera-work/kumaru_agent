# `src\kumaru\backends\openai_compat.py`

> Delegates generation to any server that implements OpenAI-compatible chat completions.

**Read this when:** you are pointing Kumaru at vLLM, Ollama, SGLang, TGI, LM Studio, llama.cpp server, or a hosted OpenAI-compatible API.

---

## What it does

`OpenAICompatBackend` creates an `httpx.AsyncClient`, verifies the upstream `/models` endpoint during load, and streams `/chat/completions` Server-Sent Events into Kumaru `StreamChunk` objects. It converts Kumaru messages with `Message.to_openai()`, sends normalized generation params, requests streaming usage when supported, and locally enforces stop sequences. It is registered as `openai_compat` with aliases `ollama`, `vllm`, and `remote`.

## Why it exists

The local transformers backend is excellent for development but decodes one request at a time. Dedicated serving engines such as vLLM add continuous batching and PagedAttention, interleaving many requests into one decode loop and managing KV cache in pages. This adapter lets operators get that throughput by changing YAML instead of changing the agent, API, or UI.

## Mental model

1. Kumaru remains the agent/API/UI process; the upstream server owns model loading, batching, KV cache, and hardware scheduling.
2. `backend.model` must be a model id the upstream server accepts.
3. Stop sequences are enforced locally because OpenAI-compatible servers differ in stop support.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `DEFAULT_BASE_URL` | `str = "http://127.0.0.1:11434/v1"` | Default endpoint matching Ollama's OpenAI-compatible API. |
| `OpenAICompatBackend` | `class OpenAICompatBackend(config: BackendConfig)` | Remote backend registered as `openai_compat`, `ollama`, `vllm`, and `remote`. |
| `OpenAICompatBackend.name` | `str = "openai_compat"` | Registry and introspection name. |
| `OpenAICompatBackend.aclose` | `async def aclose(self) -> None` | Closes the async HTTP client and marks unloaded. |
| `OpenAICompatBackend.info` | `def info(self) -> BackendInfo` | Reports remote model id, remote device/dtype, context length, and base URL. |
| `OpenAICompatBackend.stream` | `async def stream(self, messages: Sequence[Message], params: GenerationParams) -> AsyncIterator[StreamChunk]` | POSTs streaming chat completions, parses SSE lines, maps deltas/usages, and emits final chunk. |
| `OpenAICompatBackend.list_remote_models` | `async def list_remote_models(self) -> list[str]` | Returns upstream model ids from `/models`. |

## How to use it

### Serve through vLLM

The shipped profile notes that vLLM has no native Windows build, so start it in Linux/WSL or on a Linux server:

```powershell
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.90
```

Then run Kumaru:

```powershell
kumaru serve -c vllm
```

```yaml
backend:
  name: openai_compat
  model: Qwen/Qwen2.5-7B-Instruct
  base_url: http://127.0.0.1:8000/v1
  api_key: null
  timeout_s: 300
  max_context_length: 16384

server:
  max_concurrent_requests: 64
```

### Use Ollama's default endpoint

```yaml
backend:
  name: openai_compat
  model: qwen2.5:7b-instruct
  base_url: http://127.0.0.1:11434/v1
  max_context_length: 8192
```

```powershell
ollama serve
kumaru chat -c configs\ollama.yaml -q "hello"
```

If `base_url` is omitted, this backend defaults to `http://127.0.0.1:11434/v1`.

### Set an upstream API key without committing it

```powershell
$env:KUMARU_BACKEND__API_KEY = "your-token"
kumaru serve -c vllm
```

```yaml
backend:
  name: openai_compat
  model: Qwen/Qwen2.5-7B-Instruct
  base_url: https://llm-gateway.example.com/v1
  api_key: null
```

### List upstream models from Python

```python
from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS

cfg = load_config("vllm")
backend = BACKENDS.create(cfg.backend.name, cfg.backend)
await backend.load()
print(await backend.list_remote_models())
await backend.aclose()
```

## Tuning / operational notes

| Knob | Effect | Trade-off / expected number |
|---|---|---|
| `backend.base_url` | Remote `/v1` API base. | Defaults to Ollama at `http://127.0.0.1:11434/v1`; vLLM profile uses `http://127.0.0.1:8000/v1`. |
| `backend.model` | Sent in every request. | Must match upstream model id or alias. |
| `backend.timeout_s` | `httpx` total timeout, with 10-second connect timeout. | Default 300s; raise for long generations or lower for fail-fast gateways. |
| `backend.api_key` | Adds an Authorization header when configured. | Prefer `KUMARU_BACKEND__API_KEY`; do not commit secrets. |
| `generation.stop` | Sent to upstream, capped to first 4 strings, and enforced locally. | Upstream support varies; local tracking protects downstream parsing. |
| `stream_options.include_usage` | Requests a final usage chunk. | vLLM and newer llama.cpp honor it; unsupported servers ignore it. |
| `server.max_concurrent_requests` | Kumaru admission limit before forwarding. | `configs\vllm.yaml` uses 64 because vLLM handles real batching/admission. |

Source/config expectations: vLLM continuous batching and PagedAttention are typically 10-24x the throughput of naive Hugging Face generation on identical hardware. PagedAttention reduces KV-cache fragmentation from about 60-80% waste to a few percent. Scaling beyond one GPU is a vLLM flag such as `--tensor-parallel-size 4`, not a Kumaru code change.

## How to extend it

Add small server-specific payload differences in `_payload()` and drive them from `BackendConfig.extra`:

```python
def _payload(self, messages, params, *, stream: bool):
    payload = {
        "model": self.config.model,
        "messages": [m.to_openai() for m in messages],
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "stream": stream,
    }
    if self.config.extra.get("send_top_k"):
        payload["top_k"] = params.top_k
    return payload
```

```yaml
backend:
  name: openai_compat
  model: my-model
  base_url: http://127.0.0.1:8000/v1
  extra:
    send_top_k: true
```

Do not remove local `StopSequenceTracker` enforcement; inconsistent upstream stop behavior is why the adapter has it.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Cannot reach OpenAI-compatible server` at load | Upstream is not running, wrong port/host, or `/models` fails. | Start `ollama serve` or `vllm serve ...`; set `backend.base_url` to the server `/v1` base. |
| HTTP 400/404 from chat completions | `backend.model` is not accepted by the upstream. | Check `/models` or `list_remote_models()` and update `backend.model`. |
| Stream hangs or times out | Upstream generation exceeds `timeout_s` or server stalled. | Raise `timeout_s`, lower `generation.max_tokens`, or inspect upstream logs. |
| Usage is zero | Upstream ignored `stream_options.include_usage`. | Use a server/version that emits streaming usage or treat usage as unavailable. |
| Text continues past a stop upstream | Server stop support is incomplete. | Keep `generation.stop` configured; local tracking truncates emitted text. |
| Works locally but fails in deployment | `127.0.0.1` points at the wrong host/container. | Use service DNS or the reachable host address for the Kumaru process. |
| Unauthorized response | Missing or wrong upstream API key. | Set `KUMARU_BACKEND__API_KEY` or `backend.api_key`. |

## Related files

- `src\kumaru\backends\base.py` — backend contract and stop tracker.
- `src\kumaru\core\config.py` — `base_url`, `api_key`, `timeout_s`, model, and context settings.
- `src\kumaru\core\registry.py` — lazy registration as `openai_compat`, `ollama`, `vllm`, and `remote`.
- `src\kumaru\core\types.py` — OpenAI-shaped `Message.to_openai()` and normalized generation params.
- `configs\vllm.yaml` — production-throughput profile for this backend.
- `configs\default.yaml` — documents remote backend fields.
- `src\kumaru\backends\transformers_backend.py` — local development backend this replaces at scale.
