# `src/kumaru/backends/openai_compat.py`

> HTTP backend for any OpenAI-compatible `/chat/completions` server.

**Read this when:** connecting Kumaru to Ollama, llama.cpp server, LM Studio, vLLM, or OpenAI-compatible APIs.

---

## What it does
It builds chat-completion payloads, posts to `/chat/completions`, parses non-streamed and SSE streamed responses, reads optional bearer tokens from the environment, and checks `/models` for health.

## Why it exists
One wire format covers many local and remote inference servers. Operators can scale by moving the model server and changing `base_url`, without changing the agent or UI.

## Mental model
`backend.model` is required. `BackendConfig.extra` is merged into every payload for server-specific parameters.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `OpenAICompatBackend` | `OpenAICompatBackend(config: BackendConfig, *, client: httpx.Client | None = None)` | OpenAI-compatible HTTP backend. |
| `OpenAICompatBackend.name` | `str` | Always `openai_compat`. |
| `OpenAICompatBackend.generate` | `generate(self, request: ChatRequest) -> GenerationResult` | POST non-streamed chat completion. |
| `OpenAICompatBackend.stream` | `stream(self, request: ChatRequest) -> Iterator[StreamChunk]` | POST streamed chat completion and parse SSE. |
| `OpenAICompatBackend.health` | `health(self) -> dict[str, object]` | GET `/models` and report readiness. |
| `OpenAICompatBackend.close` | `close(self) -> None` | Close owned HTTP client. |

## How to use it
```python
from kumaru.core.config import BackendConfig
from kumaru.backends.openai_compat import OpenAICompatBackend
config = BackendConfig(name="openai_compat", model="llama3.1:8b", base_url="http://localhost:11434/v1")
backend = OpenAICompatBackend(config)
```

```python
from kumaru.core.types import ChatRequest, Message
reply = backend.generate(ChatRequest(messages=[Message.user("hello")]))
print(reply.text)
```

```yaml
backend:
  name: openai_compat
  model: llama3.1:8b
  base_url: http://localhost:11434/v1
  api_key_env: KUMARU_API_KEY
```

```bash
KUMARU_API_KEY=sk-... kumaru serve -c configs/openai-compat.yaml
```

## How to extend it
Add provider-specific options under `backend.extra`; `_payload()` merges them after standard fields. Example: `extra: {frequency_penalty: 0.2}` sends that key with every request.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| `backend.model must be set` | Model name is empty. | Set `backend.model` to the server's model id. |
| HTTP 404 model not found | Server rejected the model name. | Check loaded model names and `base_url`. |
| Connection failure | Model server is not reachable. | Start the server or fix `backend.base_url`. |
| No token usage in UI | Streaming server does not emit usage events. | Expect elapsed time only, or use a server that sends usage. |

## Related files
- [OpenAI config](../../../configs/openai-compat.yaml.md)
- [Backend base](./base.md)
- [Chat request types](../core/types.md)
