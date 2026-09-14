# `src/kumaru/server/__init__.py`

> Server-layer facade exposing the FastAPI application factory.

**Read this when:** embedding or launching Kumaru as an ASGI app.

---

## What it does
It re-exports `create_app` from `kumaru.server.app`. This is the stable server import for ASGI runners and tests.

## Why it exists
Consumers should not rely on the internal module path for the application factory. This file keeps server startup imports short and stable.

## Mental model
The server layer owns HTTP concerns only. It receives an agent or builds one through the app factory.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `create_app` | `create_app(config: KumaruConfig | None = None, *, agent: ChatAgent | None = None) -> FastAPI` | Build a configured FastAPI app. |

## How to use it
```python
from kumaru.server import create_app
app = create_app()
```

```python
from kumaru.server import create_app
from kumaru.core.config import load_config
app = create_app(load_config("configs/default.yaml"))
```

## How to extend it
If the server exposes another stable factory, re-export it here. Keep concrete route organization inside `app.py` and route modules.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| ASGI import fails | Import path points at an internal renamed module. | Use `kumaru.server:create_app` or import from this facade. |
| Model loads in tests | No fake agent was injected into `create_app`. | Pass `agent=` to the factory. |

## Related files
- [Server app](./app.md)
- [Chat routes](./routes_chat.md)
- [Schemas](./schemas.md)
