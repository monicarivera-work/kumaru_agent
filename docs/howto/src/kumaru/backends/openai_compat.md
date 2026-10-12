# `src\kumaru\backends\openai_compat.py`

> Delegates generation to any server that implements OpenAI-compatible chat completions, and normalises its reasoning, sampling knobs and finish reasons into Kumaru's contract.

**Read this when:**
- you are pointing Kumaru at vLLM, Ollama, SGLang, TGI, LM Studio, a llama.cpp server you run yourself, or a hosted OpenAI-compatible API;
- a hosted API rejects Kumaru's requests;
- thinking output arrives in the wrong place;
- you are subclassing it for an engine-specific feature, as `llamacpp_server` does.

---

## What it does

`OpenAICompatBackend` creates an `httpx.AsyncClient` and verifies the upstream `GET /models` during load. It then streams `POST /chat/completions` Server-Sent Events into Kumaru `StreamChunk`s:
- messages are converted with `Message.to_openai()`, which drops earlier reasoning from the history;
- a *portable* payload is always sent, plus *extended* sampling and thinking fields unless `strict_openai` is set;
- streaming usage is requested;
- reasoning is read from `delta.reasoning_content` or `delta.reasoning`, with an inline `<think>` fallback;
- stop sequences are sent upstream (first 4) and also enforced locally, on the answer only;
- upstream finish reasons are normalised.

It is registered as `openai_compat` with aliases `ollama`, `vllm` and `remote`. `llamacpp_server` is a subclass: it reuses all of this and only adds process management plus llama.cpp-specific fields.

v0.2 changes:
- **Reasoning channel.** Separate reasoning fields and inline `<think>…</think>` both become `StreamChunk.reasoning`.
- **More request fields.** `presence_penalty` is sent as a standard field. `top_k`, `min_p`, `repetition_penalty` and `chat_template_kwargs.enable_thinking` are sent as extended fields, and the new `strict_openai` switch drops them. v0.1 sent none of these. So with the defaults, `top_k: 50` and `repetition_penalty: 1.05` now reach the server.
- **Error handling.** An in-stream `error` event now raises `BackendError`, and so does a connection that fails mid-request (refused, reset or silent for `timeout_s`), through the `_connection_lost` override point.
- **Finish reasons.** Unknown values such as `tool_calls` or `content_filter` map to `"stop"` instead of failing validation.

## Why it exists

The local transformers backend is excellent for development, but every request runs its own batch-of-one `generate` call on its own thread, so concurrent requests compete for the GPU instead of sharing a decode loop. Dedicated serving engines such as vLLM add continuous batching and PagedAttention, interleaving many requests into one decode loop and managing KV cache in pages. This adapter lets operators get that throughput by changing YAML instead of changing the agent, API or UI.

The extended/strict split exists because the two kinds of server want opposite things:
- Self-hosted engines accept extra knobs. Qwen's recommended sampling needs `top_k` and `min_p`, and thinking models need the template toggle.
- Hosted APIs reject unknown fields with HTTP 400.

A single flag keeps one adapter usable for both.

## Mental model

1. Kumaru remains the agent/API/UI process. The upstream server owns model loading, batching, KV cache and hardware scheduling.
2. `backend.model` must be a model id the upstream server accepts.
3. The payload has two layers:
   - `_payload()`, the portable core every OpenAI-compatible server accepts;
   - `_extended_fields()`, engine extensions that subclasses override.
4. Reasoning has three sources, merged in this order per delta:
   - `reasoning_content`, used by vLLM, llama.cpp and DeepSeek;
   - `reasoning`, used by Ollama and OpenRouter;
   - `<think>…</think>` at the start of `content`, via `ReasoningSplitter()` with its defaults.
5. Stop sequences are enforced locally because OpenAI-compatible servers differ in stop support. The local tracker sees answer text only. The same list is also sent upstream, and an engine matches it on its raw output. llama-server does, and that output includes the thinking (`find_stopping_strings` runs on the generated text before reasoning is parsed out). So a stop string the model *thinks* can still end the reply upstream.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `DEFAULT_BASE_URL` | `DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"` | Default endpoint; Ollama's OpenAI-compatible API. |
| `normalize_finish_reason` | `def normalize_finish_reason(value: Any) -> FinishReason` | `stop`/`eos` → `"stop"`. `length`, `error` and `cancelled` pass through. Anything else, including `None`, becomes `"stop"`. Shared with `llamacpp_backend`. |
| `OpenAICompatBackend` | `class OpenAICompatBackend(config: BackendConfig)` | Remote backend registered as `openai_compat`, `ollama`, `vllm` and `remote`. `__init__` strips a trailing `/` from `base_url or DEFAULT_BASE_URL`. |
| `OpenAICompatBackend.name` | `name = "openai_compat"` | Registry and introspection name. |
| `OpenAICompatBackend.aclose` | `async def aclose(self) -> None` | Closes the HTTP client and marks the backend unloaded. Safe to call twice. |
| `OpenAICompatBackend.info` | `def info(self) -> BackendInfo` | `device`/`dtype` = `"remote"`, `context_length` = `backend.max_context_length`, `extra={"base_url": ...}`. |
| `OpenAICompatBackend.strict_openai` | `property -> bool` | `bool(backend.extra.strict_openai)`, default `False`. When true, the extended fields are not sent. |
| `OpenAICompatBackend.stream` | `async def stream(self, messages: Sequence[Message], params: GenerationParams) -> AsyncIterator[StreamChunk]` | Lazy-loads, POSTs a streaming chat completion, parses SSE, splits reasoning, tracks stops, and ends with one `StreamChunk(finish_reason=..., usage=...)`. |
| `OpenAICompatBackend.list_remote_models` | `async def list_remote_models(self) -> list[str]` | Lazy-loads, then returns the `id`s from upstream `GET /models`. |

Protected extension points, documented because subclasses rely on them:
- `_payload(messages, params, *, stream: bool) -> dict[str, Any]` — the portable request body;
- `_extended_fields(params) -> dict[str, Any]` — the engine-specific additions;
- `_connection_lost(exc: Exception) -> BackendError` — the error raised when the HTTP connection fails during a request. `llamacpp_server` overrides it to report its child's exit code and log.

### The request body

| Field | Sent when | Layer |
|---|---|---|
| `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `stream` | always | portable |
| `stop` | `generation.stop` is non-empty. Only the first 4 strings are sent upstream (a common server cap); all of them are enforced locally. | portable |
| `seed` | `seed` is not `None` | portable |
| `presence_penalty` | `presence_penalty` is not `None` (a standard OpenAI field) | portable |
| `stream_options: {include_usage: true}` | streaming | portable |
| `top_k` | `top_k > 0` | extended |
| `min_p` | `min_p` is not `None`, so an explicit `0.0` *is* sent | extended |
| `repetition_penalty` | `repetition_penalty != 1.0` | extended |
| `chat_template_kwargs: {enable_thinking: <bool>}` | `thinking` is not `None`. This is the Qwen3/DeepSeek/GLM convention, honoured by vLLM, SGLang and llama.cpp. | extended |

`thinking_budget` has no portable wire field, so this class ignores it. `llamacpp_server` maps it to llama.cpp's `reasoning_budget_tokens`; see [`llamacpp_server.md`](llamacpp_server.md).

With every default (`GenerationParams()`) and `strict_openai` off, the body is `model, messages, max_tokens: 1024, temperature: 0.7, top_p: 0.95, stream, stream_options, top_k: 50, repetition_penalty: 1.05`.

### How the stream is parsed

- Lines that do not start with `data:` are ignored. These are keep-alives, comments and `event:` lines.
- `data: [DONE]` ends the stream. A malformed JSON payload is skipped with a debug log.
- An event with a truthy `error` raises `BackendError("Upstream error: ...")`.
- An event with `usage` updates usage, and the last one wins. If no event carries usage, the final chunk carries `Usage()` (zeros). `generate()` then estimates usage. The streaming stats frame does not: it reports `completion_tokens: 0` and `tokens_per_second: 0.0`.
- Per choice, the reasoning field plus any inline reasoning becomes `reasoning`. The remaining answer text goes through the stop tracker and becomes `delta`. Both travel in one chunk. `finish_reason` is normalised.
- After a stop sequence matches, the loop breaks and leaves the response context, which closes the HTTP stream early.
- Any `httpx.TransportError` while connecting or reading is re-raised as `self._connection_lost(exc)`, chained `from exc`. An `httpx.ReadTimeout` means the server is connected but silent, so it gets its own message; everything else (`ConnectError`, `ReadError`, `RemoteProtocolError`, `ConnectTimeout`, …) reports the exception type. The base `generate()` aggregates `stream()`, so non-streaming calls raise the same errors.

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

Save as `configs\ollama.yaml` (not shipped):

```yaml
backend:
  name: openai_compat
  model: qwen2.5:7b-instruct
  base_url: http://127.0.0.1:11434/v1
  max_context_length: 8192
```

```powershell
ollama serve          # in another terminal
kumaru chat -c ollama -q "hello"
```

If `base_url` is omitted, this backend defaults to `http://127.0.0.1:11434/v1`. `-c` accepts a profile name from `configs\` or a path to any YAML file.

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
import asyncio

from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS


async def main() -> None:
    cfg = load_config("vllm")
    backend = BACKENDS.create(cfg.backend.name, cfg.backend)
    try:
        print(await backend.list_remote_models())  # lazy-loads; raises BackendError if unreachable
    finally:
        await backend.aclose()


asyncio.run(main())
```

### Point at a hosted API (strict mode)

Hosted APIs reject `top_k`, `min_p`, `repetition_penalty` and `chat_template_kwargs` with HTTP 400. Strict mode sends only the portable fields:

```yaml
backend:
  name: openai_compat
  model: <hosted-model-id>
  base_url: https://llm-gateway.example.com/v1
  api_key: null            # set KUMARU_BACKEND__API_KEY instead of committing it
  extra:
    strict_openai: true
```

The cost of strict mode is that the thinking toggle is dropped silently, because `chat_template_kwargs` is an extended field. `presence_penalty` is still sent, since it is a standard OpenAI field.

### Turn thinking on or off on a self-hosted thinking model

The toggle reaches the server as `chat_template_kwargs.enable_thinking`. Leave `thinking` as `null` to keep the model's own default. Per request:

```powershell
kumaru chat -c vllm --think -q "Prove that sqrt(2) is irrational."
kumaru chat -c vllm --no-think -q "One-line answer: capital of France?"
```

Reasoning streams back as `delta.reasoning_content` in Kumaru's SSE and is shown dim by `kumaru chat`. Kumaru's `thinking_mode` preset still applies on top: generation < `thinking_mode` (only when thinking is on) < per-request. See [`default.yaml.md`](../../../configs/default.yaml.md).

## Tuning / operational notes

| Knob | Effect | Trade-off / expected number |
|---|---|---|
| `backend.base_url` | Remote `/v1` API base. | Defaults to Ollama at `http://127.0.0.1:11434/v1`; `configs\vllm.yaml` uses `http://127.0.0.1:8000/v1`. |
| `backend.model` | Sent in every request. | Must match the upstream model id or alias. |
| `backend.timeout_s` | `httpx` timeout for each connect, read, write and pool wait (not the whole request), with a 10-second connect timeout. | Default 300 s. It applies between SSE events, so a long generation that keeps streaming never trips it. A silent upstream does: `<base_url> sent nothing for 300s.` |
| `backend.api_key` | Adds `Authorization: Bearer <api_key>` when set. | Prefer `KUMARU_BACKEND__API_KEY`; do not commit secrets. |
| `backend.extra.strict_openai` | `true` drops `top_k`, `min_p`, `repetition_penalty` and `chat_template_kwargs`. | Required for hosted APIs. It also loses the thinking toggle. |
| `generation.stop` | Sent upstream (first 4) and enforced locally on the answer. | Upstream support varies; local tracking protects downstream parsing. |
| `generation.top_k`, `min_p`, `repetition_penalty` | Extended fields. | Sent by default since v0.2: `top_k: 50` and `repetition_penalty: 1.05` when unchanged. Set `top_k: 0` or `repetition_penalty: 1.0` to stop sending them. |
| `generation.presence_penalty` | Standard field, sent when not `null`. | The usual cure for repetition loops on long reasoning traces. |
| `generation.thinking` | `chat_template_kwargs.enable_thinking`. | `null` = model default. Ignored by models whose template has no such variable. |
| `stream_options.include_usage` | Requests a final usage chunk. | vLLM and newer llama.cpp honour it; unsupported servers ignore it. |
| `server.max_concurrent_requests` | Kumaru admission limit before forwarding. | `configs\vllm.yaml` uses 64 because vLLM does real batching and admission. |

Published numbers, for scale: the vLLM authors report 14–24× the serving throughput of plain Hugging Face Transformers (one completion per request, LLaMA-7B/13B on A10G/A100). They also report KV-cache waste of under 4 % with PagedAttention, against 60–80 % in earlier systems. Scaling beyond one GPU is a vLLM flag such as `--tensor-parallel-size 4`, not a Kumaru code change.

## How to extend it

Subclass and override `_extended_fields()`. Keep `_payload()` as the portable core, so that `strict_openai` keeps working for free. This example passes operator-supplied fields from YAML into every request:

```python
from typing import Any

from kumaru.backends.openai_compat import OpenAICompatBackend
from kumaru.core.registry import BACKENDS
from kumaru.core.types import GenerationParams


@BACKENDS.register("my_engine")
class MyEngineBackend(OpenAICompatBackend):
    name = "my_engine"

    def _extended_fields(self, params: GenerationParams) -> dict[str, Any]:
        fields = super()._extended_fields(params)
        fields.update(self.config.extra.get("extra_body", {}))  # operator-supplied fields
        return fields
```

```yaml
backend:
  name: my_engine
  model: my-model
  base_url: http://127.0.0.1:8000/v1
  extra:
    extra_body:
      ignore_eos: false
```

With defaults, the request body then ends in `"top_k": 50, "repetition_penalty": 1.05, "ignore_eos": false`.

The reference subclass is `LlamaServerBackend` in `llamacpp_server.py`. Its `_extended_fields()` renames `repetition_penalty` to llama.cpp's `repeat_penalty` and adds `reasoning_budget_tokens` and `reasoning_budget_message`, and its `_load()` spawns the server before calling the inherited load.

Rules for subclasses:
- Do not remove the local `StopSequenceTracker` enforcement; inconsistent upstream stop behaviour is why the adapter has it.
- Do not move standard fields into `_extended_fields()`, or strict mode will lose them.
- If the engine reports reasoning under a third field name, extend the `thought = ...` lookup rather than post-processing `content`.
- To explain a dropped connection in engine terms, override `_connection_lost(exc)` and return `super()._connection_lost(exc)` when you have nothing better. `llamacpp_server` does this: it reports the child's exit code and log tail when the process has died.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Cannot reach OpenAI-compatible server at <url>: ...` at load | Upstream not running, wrong host or port, or `GET /models` fails (including 401). | Start `ollama serve` or `vllm serve ...`; set `backend.base_url` to the server's `/v1` base. |
| `Upstream returned 400: ...` from a hosted API, naming `top_k`, `min_p`, `repetition_penalty` or `chat_template_kwargs` | Extended fields sent to a strict server. | Set `backend.extra.strict_openai: true`. |
| `Upstream returned 400/404: ...` naming the model | `backend.model` is not accepted upstream. | Check `list_remote_models()` and update `backend.model`. The error body is truncated to 500 characters. |
| `Upstream error: ...` mid-reply | The server reported an error inside the SSE stream after the response had started. | Read the upstream server's log; the message is passed through verbatim. |
| `<base_url> sent nothing for 300s.`, hint `The server may be overloaded or still reading a very long prompt. If that is expected, raise backend.timeout_s.` | `httpx.ReadTimeout`: connected, but no byte for `backend.timeout_s`. A very long prompt, a busy server, or a request queued behind others upstream. | If that is expected, raise `backend.timeout_s`; otherwise check the upstream server. |
| `Lost the connection to <base_url> (<Type>: <message>).`, hint `Check that the model server is still running and reachable.` | Any other transport error during a request: refused (`ConnectError`), reset (`ReadError`, `RemoteProtocolError`), or no connection within 10 s (`ConnectTimeout`). The server stopped or crashed after load. | Restart the upstream server, or fix `backend.base_url`. |
| Thinking toggle ignored | `strict_openai` is on, or the model or template has no `enable_thinking` variable, or the server does not support `chat_template_kwargs`. | Turn off strict mode for self-hosted servers; use a thinking model; or use `llamacpp_server`. |
| Thinking text, often followed by a stray `</think>`, shown as answer | The server returns thinking inline in `content`, but not starting with `<think>`: its chat template opens `<think>` inside the prompt, so the output starts mid-thought. Or the server uses a field name other than `reasoning_content`/`reasoning`. | Enable the server's reasoning parser (vLLM `--reasoning-parser`) so that thinking arrives in `reasoning_content`, or extend the field lookup in a subclass. |
| `thinking_budget` has no effect | No portable wire field; this class ignores it. | Use `llamacpp_server`, or cap thinking with `max_tokens`. |
| Usage is zero | Upstream ignored `stream_options.include_usage`. | Use a server or version that emits streaming usage, or treat usage as unavailable. |
| Text continues past a stop upstream | Server stop support is incomplete, or you configured more than 4 stops. | Keep `generation.stop` configured; local tracking truncates emitted text. |
| With `generation.stop` set, a thinking model's reply ends mid-thought or with an empty answer | The upstream engine matched a stop string inside the reasoning. llama-server checks stops on raw generated text, thinking included; the local answer-only tracking cannot prevent that. | Use stop strings that cannot occur in prose, or leave `generation.stop` empty for thinking models. |
| `content_filter`/`tool_calls` endings reported as `stop` | `normalize_finish_reason` maps unknown reasons to `"stop"`. | Expected; Kumaru's `FinishReason` has no such values. |
| Works locally but fails in deployment | `127.0.0.1` points at the wrong host or container. | Use service DNS or a host address reachable from the Kumaru process. |
| Unauthorized response | Missing or wrong upstream API key. | Set `KUMARU_BACKEND__API_KEY` or `backend.api_key`. |
| `The openai_compat backend cannot attach a LoRA adapter (backend.adapter_path is '<path>').` at load | `--adapter` or `adapter_path` is set. Kumaru attaches adapters only in the transformers backend; this class refuses rather than serve the base model under the adapter's name. | Let the upstream server load the adapter (vLLM: `--enable-lora --lora-modules <name>=<path>`) and set `backend.model` to `<name>`; or serve it with `backend.name: transformers`. |

## Related files

- `src\kumaru\backends\base.py` — backend contract, `StopSequenceTracker`, `ReasoningSplitter`.
- `src\kumaru\backends\llamacpp_server.py` — subclass that manages a local `llama-server`; see [`llamacpp_server.md`](llamacpp_server.md).
- `src\kumaru\core\config.py` — `base_url`, `api_key`, `timeout_s`, `extra`, model and context settings.
- `src\kumaru\core\registry.py` — lazy registration as `openai_compat`, `ollama`, `vllm` and `remote`.
- `src\kumaru\core\types.py` — `Message.to_openai()`, `GenerationParams`, `FinishReason`.
- `configs\vllm.yaml` — production-throughput profile for this backend.
- `configs\default.yaml` — documents remote backend fields.
- `src\kumaru\backends\transformers_backend.py` — local development backend this replaces at scale.
- `tests\test_backends.py` — strict-payload, finish-reason and transport-error tests.
