# `src\kumaru\server\__init__.py`

> Package entry point for Kumaru's HTTP server layer.

**Read this when:** you need to create the FastAPI app from another process, inspect the server package boundary, or keep imports stable while server internals move.

---

## What it does
This file re-exports `create_app` from `kumaru.server.app`. It marks the server package as the OpenAI-compatible HTTP layer plus the web UI. Callers can build an ASGI app without importing implementation modules directly.

## Why it exists
Compatibility is a package-level feature: clients should see one stable way to create the server. Without this boundary, scripts and tests would import `kumaru.server.app` directly and become sensitive to internal refactors. The rest of the server can change while `from kumaru.server import create_app` remains stable.

## Mental model
- `kumaru.server` is the HTTP/UI surface, not the agent itself.
- `create_app()` is the stable app factory.
- The OpenAI-compatible routes are mounted inside the factory.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `create_app` | `create_app(config: KumaruConfig \| None = None) -> FastAPI` | Builds the FastAPI application; re-exported from `kumaru.server.app`. With no `config` it calls `load_config()`: field defaults plus `KUMARU_*` environment variables, with no YAML profile. |
| `__all__` | `list[str] = ["create_app"]` | Defines the intentional package public surface. |

## How to use it

### Build an app for ASGI servers

```python
from kumaru.server import create_app

app = create_app()
```

With no argument the app uses field defaults plus `KUMARU_*` environment variables. The default backend is `transformers` with `Qwen/Qwen2.5-7B-Instruct`, which `from_pretrained` downloads from the Hugging Face Hub on first start if it is not cached. Pass a config, as below, unless that is what you want.

### Build an app with explicit config

```python
from kumaru.core.config import load_config
from kumaru.server import create_app

config = load_config("test")
app = create_app(config)
```

### Smoke-test with FastAPI's test client

```python
from fastapi.testclient import TestClient
from kumaru.core.config import load_config
from kumaru.server import create_app

with TestClient(create_app(load_config("test"))) as client:  # `with` runs the lifespan, which starts loading
    res = client.get("/healthz")
    assert res.status_code in {200, 503}
    print(res.json()["status"])  # "loading" or "ok"; "error" (with 503) if the load failed
```

Without `with`, the lifespan never runs: nothing loads and `/healthz` keeps answering 200 with `"status": "loading"`.

## How to extend it
Keep this file small. Add public server entry points here only when external callers should depend on them.

Worked example: exposing a future `create_admin_app` factory.

```python
from kumaru.server.app import create_app
from kumaru.server.admin import create_admin_app

__all__ = ["create_app", "create_admin_app"]
```

Do not re-export route handlers; clients should call HTTP endpoints, and tests can import route modules directly.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: cannot import name create_app` | Stale install or edited `__all__`. | Reinstall the package and keep `create_app` exported. |
| App starts without routes | A custom factory bypassed `kumaru.server.create_app`. | Use the package-level factory or include `routes_chat.router`. |
| Server imports trigger model load | Expensive work was added at import time. | Keep loading inside `AppState.start_loading()` and `Backend.load()`. |
| Tests share server state unexpectedly | A module-level app was introduced. | Keep using the factory to create isolated app instances. |

## Related files

- `src\kumaru\server\app.py` — defines `create_app` and server runtime state.
- `src\kumaru\server\routes_chat.py` — defines the HTTP endpoints mounted by the app.
- `src\kumaru\server\schemas.py` — defines the wire contract for those endpoints.
- `src\kumaru\agent\chat.py` — server routes call into the agent layer.

