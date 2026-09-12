# `src\kumaru\server\app.py`

> FastAPI app factory, runtime state, model-loading lifecycle, and in-process metrics.

**Read this when:** you are changing server startup, concurrency limits, backend hot-swap behavior, UI mounting, or shutdown cleanup.

---

## What it does
This file builds the FastAPI application and owns process-local runtime state. It starts backend loading in the background, exposes readiness through shared state, caps concurrent generations with a semaphore, mounts the UI, and maps `KumaruError` to JSON responses. `run()` starts uvicorn for `kumaru serve`.

## Why it exists
Large local models can take 30-60 seconds to load. If startup blocked on loading, clients would see connection refused and orchestrators might kill a healthy process. The explicit semaphore also prevents unlimited concurrent GPU work from turning graceful queueing into CUDA OOM and process death.

## Mental model
- The port opens before the model is ready; `/healthz` reports `loading` meanwhile.
- `AppState` is per process, not shared across replicas.
- Backend swaps close the old backend before allocating the new one to release VRAM.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `UI_DIR` | `Path` | Repository `ui` directory mounted at `/static` when `server.serve_ui` is true. |
| `Metrics` | `class Metrics()` | Minimal in-memory counters for requests, errors, tokens, latency, and uptime. |
| `Metrics.record` | `record(*, prompt: int, completion: int, latency_ms: float) -> None` | Adds one successful generation to counters. |
| `Metrics.snapshot` | `snapshot() -> dict[str, Any]` | Returns `/metrics` and `/v1/info` metrics payload. |
| `AppState` | `class AppState(config: KumaruConfig)` | Holds config, backend, agent, metrics, semaphore, load task, and load error. |
| `AppState.ready` | `property ready -> bool` | True when an agent exists and no load error is set. |
| `AppState.start_loading` | `start_loading() -> None` | Schedules backend loading once without blocking the event loop. |
| `AppState.ensure_ready` | `async ensure_ready(timeout_s: float = 600.0) -> ChatAgent` | Waits for readiness or raises `KumaruError` with an actionable hint. |
| `AppState.swap_backend` | `async swap_backend(**overrides: Any) -> None` | Closes the current backend, updates backend config, and loads the replacement. |
| `AppState.shutdown` | `async shutdown() -> None` | Cancels loading and closes the backend. |
| `create_app` | `create_app(config: KumaruConfig | None = None) -> FastAPI` | Builds the configured ASGI application. |
| `run` | `run(config: KumaruConfig | None = None) -> None` | Starts uvicorn with host, port, and logging from config. |

## How to use it

### Start the server from Python

```python
from kumaru.server.app import run

run()  # loads config with load_config(), then starts uvicorn
```

### Build an isolated test app

```python
from kumaru.core.config import load_config
from kumaru.server.app import create_app

config = load_config("echo", use_env=False)
app = create_app(config)
```

### Check readiness with curl

```powershell
curl.exe http://127.0.0.1:8080/healthz
```

Possible response while loading:

```json
{"status":"loading","version":"0.1.0","backend":"transformers","model":"Qwen/Qwen2.5-7B-Instruct","loaded":false,"detail":null}
```

### Inspect metrics

```python
import httpx

res = httpx.get("http://127.0.0.1:8080/metrics")
res.raise_for_status()
print(res.json()["avg_latency_ms"])
```

## How to extend it
Add process-wide state to `AppState`, initialize it in `create_app()`, and access it from routes through `request.app.state.kumaru`.

Worked example: add a request counter for rejected requests.

```python
class Metrics:
    def __init__(self) -> None:
        self.requests = 0
        self.rejected = 0
        # existing fields...

    def snapshot(self) -> dict[str, Any]:
        data = {
            "requests": self.requests,
            "rejected": self.rejected,
            # existing fields...
        }
        return data
```

Then increment it in a route or exception handler:

```python
request.app.state.kumaru.metrics.rejected += 1
```

For new route files, import and include their router inside `create_app()` after the app is constructed.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `/healthz` stays `loading` | Backend load is still running or blocked on model download. | Check logs and wait; first run can take several minutes. |
| `/healthz` returns 503 with `status: error` | `BACKENDS.create()` or `backend.load()` failed. | Read `detail`, check dependencies/model path, or try `--backend echo`. |
| Requests queue and feel slow | `server.max_concurrent_requests` is lower than demand or backend is slow. | Tune the semaphore for hardware capacity; do not set it high enough to OOM. |
| Hot-swap fails with out-of-memory | Old backend did not release device memory or new model is too large. | Ensure `aclose()` frees resources; choose quantization/smaller model. |
| Browser cannot call API from another origin | CORS origins do not include the UI origin. | Set `server.cors_origins` in YAML or env. |
| UI route `/` returns 404 | `server.serve_ui` is false or `UI_DIR` does not exist. | Enable `serve_ui` and run from a complete repo checkout. |

## Related files

- `src\kumaru\server\routes_chat.py` — route handlers use `AppState` and metrics.
- `src\kumaru\server\schemas.py` — response models returned by routes.
- `src\kumaru\agent\chat.py` — constructed after backend load.
- `src\kumaru\backends\base.py` — backend lifecycle contract used by `AppState`.
- `src\kumaru\core\config.py` — supplies server, backend, generation, and telemetry settings.
- `ui\app.js` — consumes `/v1/info`, `/metrics`, and streaming completions.

