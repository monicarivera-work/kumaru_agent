# `src/kumaru/server/app.py`

> FastAPI application factory that wires one process-wide agent, routes, errors, CORS, and static UI serving.

**Read this when:** starting the web/API server or injecting an agent for tests.

---

## What it does
It creates the FastAPI app, constructs or stores a single `ChatAgent` during lifespan, registers API routes, maps `KumaruError` to JSON, exposes health/config endpoints, optionally adds CORS, and mounts `ui/` at `/`.

## Why it exists
Loading a model per request would be slow and memory-heavy. Serving UI and API from the same app keeps same-origin requests simple and avoids a separate front-end build step.

## Mental model
One agent lives in `app.state.agent` for the process lifetime. `/api/*` routes are included before static UI is mounted so API paths win.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `UI_DIR` | `Path` | Resolved UI directory, overrideable with `KUMARU_UI_DIR`. |
| `create_app` | `create_app(config: KumaruConfig | None = None, *, agent: ChatAgent | None = None) -> FastAPI` | Build the ASGI app. |
| `health` | `GET /api/health -> HealthModel` | Report version/backend readiness. |
| `get_config` | `GET /api/config -> dict[str, object]` | Return non-secret runtime config. |

## How to use it
```python
from kumaru.server.app import create_app
app = create_app()
```

```python
from kumaru.core.config import load_config
from kumaru.server.app import create_app
app = create_app(load_config("configs/default.yaml"))
```

```python
from kumaru.server.app import create_app
# In tests, pass an object with ChatAgent-compatible methods.
app = create_app(agent=fake_agent)
```

```bash
kumaru serve -c configs/default.yaml
```

## How to extend it
Add new route modules with `app.include_router(...)` before the UI mount. Add new startup/shutdown resources inside the `lifespan` context so cleanup is guaranteed.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| UI 404s but API works | `UI_DIR` does not exist or `serve_ui` is false. | Set `KUMARU_UI_DIR` or fix packaging to include `ui/`. |
| CORS blocked from another origin | `server.cors_origins` is empty by default. | Configure allowed origins explicitly. |
| Backend loads during tests | Factory created a real `ChatAgent`. | Pass a fake `agent=` into `create_app`. |

## Related files
- [Routes](./routes_chat.md)
- [Schemas](./schemas.md)
- [UI index](../../../ui/index.html.md)
- [Config](../core/config.md)
