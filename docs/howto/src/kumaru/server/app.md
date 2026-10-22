# `src\kumaru\server\app.py`

> The FastAPI app factory and per-process runtime state: background model loading, readiness, hot-swap, the concurrency semaphore, metrics, the `KumaruError` → JSON handler, and the UI's static files with pinned MIME types.

**Read this when:**

- you change server startup or shutdown, concurrency limits, backend hot-swap, error-to-HTTP mapping, metrics, or UI serving;
- `/healthz` never leaves `loading`;
- the UI renders blank or unstyled on one Windows machine but not another.

---

## What it does

`create_app(config)` builds the ASGI app in this order:

1. `setup_logging(config.telemetry.log_level, fmt=config.telemetry.log_format)`.
2. A lifespan that calls `AppState.start_loading()` on startup and `AppState.shutdown()` on exit.
3. `app.state.kumaru = AppState(config)`.
4. CORS middleware.
5. The `KumaruError` exception handler.
6. The router from `src\kumaru\server\routes_chat.py`.
7. If `server.serve_ui` is true and `UI_DIR` exists: the UI, with `ui\` mounted at `/static` and `GET /` serving `ui\index.html`.

`run(config)` serves that app with uvicorn; `kumaru serve` calls it.

Importing the module pins two MIME types, process-wide:

```python
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
```

## Why it exists

**Non-blocking startup.** Loading takes a while:

- 30–60 s for a 7B transformers model;
- minutes for `llamacpp_server` on a first run, which may download a GGUF of about 20 GB. `backend.llama_server.startup_timeout_s` defaults to 900 s.

If loading ran inside a synchronous startup hook, the port would stay closed and clients would see connection-refused. Instead the port opens at once, the model loads in a background task, `/healthz` reports `loading`, and early requests simply wait.

**Bounded admission.** On one GPU, admitting every concurrent request makes all of them slower, then hits CUDA OOM, which kills the process. An `asyncio.Semaphore(server.max_concurrent_requests)` turns that crash into queueing.

**One error contract.** Domain errors carry their own `http_status` and `hint`, so a single handler maps every one to `{"error": {"message", "type", "hint"}}` without leaking stack traces. The full table is in `docs\howto\src\kumaru\core\errors.md`.

**Pinned MIME types.** On Windows, Python's `mimetypes` reads the registry (`HKEY_CLASSES_ROOT\.css` and `HKEY_CLASSES_ROOT\.js`, value `Content Type`). Some installers set those to `text/plain` or a legacy type. `StaticFiles` would then serve `styles.css` as `text/plain`, and browsers refuse to apply a stylesheet with a non-CSS MIME type, so the UI renders unstyled on that machine only. Scripts get the same strict check when loaded as modules or with `X-Content-Type-Options: nosniff`. `add_type` overrides the registry, so every machine serves the same headers.

## Mental model

**Lifecycle, one per process.** The state moves from `loading` to either `ok` or `error`.

| State | `/healthz` | `POST /v1/chat/completions` |
|---|---|---|
| `loading` | 200, `"status": "loading"` | Waits in `ensure_ready()` for up to 600 s, then 503 `Backend is still loading.` |
| `ok` | 200, `"status": "ok"` | Served. |
| `error` | 503, `"status": "error"`, `detail` = the exception text | 503 `Backend failed to load: <detail>` |

The `error` state is **sticky**. `start_loading()` creates the load task only once, so nothing retries automatically. To leave it, call `POST /v1/backend` (even with the same values) or restart the process.

**`AppState` is per process.** Each uvicorn worker or replica loads its own model, keeps its own metrics, and has its own semaphore. `ChatAgent` is stateless, so any replica can serve any request.

**The semaphore bounds generation, not HTTP.**

- Non-streaming requests hold a slot around `agent.complete()`.
- Streaming requests hold a slot for the stream's whole life, including thinking time.
- Introspection endpoints never wait for one.

**Hot-swap (`swap_backend`) runs these steps under `_lock`.**

1. Close the old backend, so its VRAM is freed before the new model allocates.
2. Clear `backend`, `agent`, `load_error`, and `_load_task`.
3. Apply the overrides to `config.backend` with `model_copy(update=...)`. This step is **not validated**.
4. Load the new backend.

It does **not** acquire the semaphore, so in-flight generations are not drained, and they may fail when their backend closes underneath them.

If the new backend fails:

- the call raises `BackendError("Failed to load new backend: ...")` (HTTP 503, no hint);
- the server is left with no backend;
- the next chat request re-attempts the same failing load.

**Errors outside `KumaruError` are not translated.**

- An unexpected exception in a non-streaming route becomes Starlette's plain-text `500 Internal Server Error`.
- A request-body validation failure is FastAPI's 422 `{"detail": [...]}`.
- Neither is counted in `metrics.errors`.
- Streaming errors after the first byte cannot change the status; they become SSE error frames (`docs\howto\src\kumaru\server\routes_chat.md`).

**Metrics are cheap and approximate.**

- `requests` counts successful generations only. Failures increment only `errors`.
- `avg_tokens_per_second` is `completion_tokens / total_latency`. It is latency-weighted and includes prompt processing and thinking time, so it is lower than the engine's decode speed.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `UI_DIR` | `Path` = `REPO_ROOT / "ui"` | Static UI directory. Exists only in a source checkout or editable install. |
| *(import side effect)* | `mimetypes.add_type("text/javascript", ".js")`; `mimetypes.add_type("text/css", ".css")` | Browser-safe content types regardless of the Windows registry. |
| `Metrics` | `Metrics()` | Counters `requests`, `errors`, `prompt_tokens`, `completion_tokens`, `total_latency_ms`, `started_at`. |
| `Metrics.record` | `record(self, *, prompt: int, completion: int, latency_ms: float) -> None` | Add one **successful** generation. |
| `Metrics.snapshot` | `snapshot(self) -> dict[str, Any]` | `uptime_s`, `requests`, `errors`, `prompt_tokens`, `completion_tokens`, `avg_latency_ms`, `avg_tokens_per_second`; served at `/metrics` and inside `/v1/info`. |
| `AppState` | `AppState(config: KumaruConfig)` | `config`, `backend: Backend \| None`, `agent: ChatAgent \| None`, `metrics`, `semaphore`, `load_error: str \| None`. |
| `AppState.ready` | `ready -> bool` (property) | `agent is not None and load_error is None`. |
| `AppState.start_loading` | `start_loading(self) -> None` | Schedule `_load()` as a task, once. |
| `AppState.ensure_ready` | `async ensure_ready(self, timeout_s: float = 600.0) -> ChatAgent` | Start the load if needed, wait (shielded) up to `timeout_s`, and return the agent. Otherwise raise `BackendError`: `Backend failed to load: ...` (hint: check the logs or try `--backend echo`) or `Backend is still loading.` (hint: `Retry in a few seconds.`). |
| `AppState.swap_backend` | `async swap_backend(self, **overrides: Any) -> None` | Close the old backend, update `config.backend` (unvalidated), and load the new one. Raises `BackendError` on failure. |
| `AppState.shutdown` | `async shutdown(self) -> None` | Cancel a pending load and close the backend, which stops a managed `llama-server` child. |
| `create_app` | `create_app(config: KumaruConfig \| None = None) -> FastAPI` | Build the app. `None` means `load_config()`: field defaults plus `KUMARU_*` env, no YAML. |
| `run` | `run(config: KumaruConfig \| None = None) -> None` | `uvicorn.run(create_app(config), host=server.host, port=server.port, log_level=telemetry.log_level.lower(), access_log=False)`. |

**Inside `create_app`, not importable:**

- `kumaru_error_handler`, which turns every `KumaruError` into `JSONResponse(status_code=exc.http_status, content={"error": {"message": exc.message, "type": type(exc).__name__, "hint": exc.hint}})` and increments `metrics.errors`.
- CORS with `allow_origins=server.cors_origins`, `allow_credentials=True`, and all methods and headers.
- FastAPI's own `/docs` and `/openapi.json`, which stay enabled.

## How to use it

### Serve a profile from Python

```python
from kumaru.core.config import load_config
from kumaru.server.app import run

run(load_config("qwen36"))      # same as: kumaru serve -c qwen36
```

### Build an isolated app for tests

```python
from fastapi.testclient import TestClient
from kumaru.core.config import load_config
from kumaru.server.app import create_app

cfg = load_config("test", use_env=False, overrides={"backend": {"extra": {"delay_s": 0}}})
with TestClient(create_app(cfg)) as client:          # `with` runs the lifespan, so loading starts
    assert client.get("/healthz").json()["status"] == "ok"
```

### Wait for readiness (PowerShell)

`curl.exe` prints the body even on 503, whereas `Invoke-RestMethod` would throw:

```powershell
do { Start-Sleep 2; $h = curl.exe -s http://127.0.0.1:8080/healthz | ConvertFrom-Json } while ($h.status -eq "loading")
$h.status    # ok | error   ($h.detail explains an error)
```

### Read metrics

```powershell
curl.exe -s http://127.0.0.1:8080/metrics
```

Captured against the `test` profile after one plain request, one streaming request with thinking, and one rejected request:

```json
{"uptime_s": 0.8, "requests": 2, "errors": 1, "prompt_tokens": 436, "completion_tokens": 47, "avg_latency_ms": 361.5, "avg_tokens_per_second": 65.01}
```

### Hot-swap the model

```powershell
$body = @{ model = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/backend -ContentType "application/json" -Body $body
```

Allowed keys and responses are listed in `docs\howto\src\kumaru\server\routes_chat.md`.

### Check what MIME types are served

```powershell
.\.venv\Scripts\python.exe -c "import mimetypes; print(mimetypes.guess_type('a.css'), mimetypes.guess_type('a.js'))"   # what the registry says
curl.exe -sI http://127.0.0.1:8080/static/styles.css    # content-type: text/css; charset=utf-8
curl.exe -sI http://127.0.0.1:8080/static/app.js        # content-type: text/javascript; charset=utf-8
```

## How to extend it

- **Process-wide state.** Add an attribute in `AppState.__init__` and read it in routes as `request.app.state.kumaru.<attr>`.
- **A metric.** Add the counter in `Metrics.__init__`, expose it in `snapshot()`, and increment it at the call site:

  ```python
  class Metrics:
      def __init__(self) -> None:
          ...
          self.client_disconnects = 0

      def snapshot(self) -> dict[str, Any]:
          return {..., "client_disconnects": self.client_disconnects}
  ```

  When you outgrow it, swap in `prometheus_client`; the call sites stay the same.
- **A route module.** Create an `APIRouter` in `src\kumaru\server\routes_<name>.py`, then add `app.include_router(...)` next to the chat router in `create_app()`. Import it inside the function, as the chat router is, because `routes_chat` imports `AppState` from this module.
- **A new error status.** Subclass `KumaruError` with an `http_status` (`docs\howto\src\kumaru\core\errors.md`); the existing handler picks it up. Add an `@app.exception_handler` only for exception types outside the `KumaruError` tree.
- **A new UI asset type.** Pin it next to the existing two, for example `mimetypes.add_type("application/wasm", ".wasm")`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `/healthz` stays `loading` for minutes | The load is still running: a first-run GGUF download, or `llama-server` warming up (up to `startup_timeout_s`, 900 s). | Watch the server log and `runs\logs\llama-server-<port>.log`. Gate traffic on `/healthz`. |
| `/healthz` returns 503 `"status": "error"`, and chat returns 503 `Backend failed to load: ...` | `BACKENDS.create()`, `ChatAgent(...)` (a bad knowledge pack or tool name, or tools enabled on a backend that cannot call them) or `backend.load()` raised. The state is sticky. | Read `detail`. Fix the cause, then `POST /v1/backend` (even with the same values) or restart. `kumaru serve -b echo` checks the rest of the stack. |
| Chat returns 503 `Backend is still loading.` | The load took longer than `ensure_ready`'s fixed 600 s. `server.request_timeout_s` is not read. | Wait for `/healthz` to report `ok` before sending traffic. |
| Chat returns 503 after a failed `POST /v1/backend` | A failed swap leaves no backend, and each chat request retries the same bad config. | Swap to a working config. |
| In-flight requests break during a swap | The swap does not drain the semaphore and closes the backend under them. | Swap while idle. |
| Swap runs out of memory | The old backend did not release memory, or the new model does not fit. | Check that `aclose()` frees resources; use a smaller quant or a shorter `max_context_length`. |
| UI renders unstyled or blank; the browser console says `Refused to apply style ... MIME type ('text/plain')` | The Windows registry maps `.css`/`.js` badly and the pin was bypassed, for example because the UI is served by another server. | Serve through `create_app`. Check with `curl.exe -sI .../static/styles.css`. |
| `GET /` returns 404 | `server.serve_ui` is false, or `UI_DIR` is missing (installed as a wheel). | Enable `serve_ui`, and run from a source checkout or `pip install -e .`. |
| Requests queue; time to first token climbs under load | All `max_concurrent_requests` slots are busy. Long thinking streams hold a slot for their whole duration. With `llamacpp_server`, admissions beyond `llama_server.parallel` queue inside `llama-server` instead. | Raise the limit only as far as VRAM allows, and keep it consistent with `parallel`. |
| Plain-text `500 Internal Server Error` | A non-`KumaruError` exception escaped a non-streaming route. | Read the server log, then raise or wrap the error as a `KumaruError` subclass. |
| The browser blocks calls from your own front end | Its origin is not in `server.cors_origins`. | Add it, for example `$env:KUMARU_SERVER__CORS_ORIGINS = '["http://localhost:3000"]'`. |
| Any web page open in your browser can call the local API | The default `cors_origins: ["*"]` with `allow_credentials=True` makes Starlette reflect every `Origin`. | Set `server.api_key` (required once bound beyond localhost) and narrow `cors_origins`. |
| `KUMARU_LOG_LEVEL` has no effect under `kumaru serve` | `create_app` calls `setup_logging(telemetry.log_level, ...)` explicitly. | Use `$env:KUMARU_TELEMETRY__LOG_LEVEL = "DEBUG"` or `kumaru serve --log-level DEBUG`. |

## Related files

- `src\kumaru\server\routes_chat.py` — endpoints that use `AppState`, the semaphore, and metrics.
- `src\kumaru\server\schemas.py` — wire models.
- `src\kumaru\core\errors.py` — `KumaruError.http_status` and `hint`, which the handler serializes.
- `src\kumaru\agent\chat.py` — `ChatAgent`, constructed before the load, so a bad knowledge pack or tool name fails before the model takes memory. It is built without an approver, so the tools that change things or use the project folder are never offered over HTTP (`docs\howto\src\kumaru\agent\tools.md`).
- `src\kumaru\backends\base.py` — `load()`/`aclose()` lifecycle used by load, swap, and shutdown.
- `src\kumaru\core\logging.py` — `setup_logging`.
- `ui\index.html`, `ui\app.js`, `ui\styles.css` — served from `/` and `/static`. `app.js` calls `/v1/chat/completions`, `/v1/info`, and `/v1/backend`.
- `tests\test_server.py` — `test_ui_assets_are_served_with_browser_safe_types` guards the MIME pin.
