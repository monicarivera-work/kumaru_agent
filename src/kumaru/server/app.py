"""FastAPI application factory.

Two decisions worth knowing about:

**One agent per process.** The model is loaded once during startup and shared
by every request. Loading per request would re-read gigabytes of weights and
exhaust memory immediately. Backends serialise themselves internally where the
underlying runtime is not thread-safe.

**The UI is served by this app.** ``ui/`` is mounted at ``/`` so the browser
and the API share an origin: no CORS, no second dev server, no build step. Open
http://127.0.0.1:8000 and it just works.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from kumaru import __version__
from kumaru.agent.chat import ChatAgent
from kumaru.core.config import KumaruConfig, load_config
from kumaru.core.errors import KumaruError
from kumaru.core.logging import get_logger
from kumaru.server.routes_chat import router as chat_router
from kumaru.server.schemas import ErrorModel, HealthModel

log = get_logger("server.app")

#: repository-root/ui - resolved relative to this file so it works from any cwd.
#: Override with KUMARU_UI_DIR when running from an installed wheel.
UI_DIR = Path(os.environ.get("KUMARU_UI_DIR") or Path(__file__).resolve().parents[3] / "ui")


def create_app(
    config: KumaruConfig | None = None, *, agent: ChatAgent | None = None
) -> FastAPI:
    """Build the application.

    ``agent`` can be injected by tests to avoid loading a real model.
    """
    config = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.agent = agent or ChatAgent(config)
        app.state.config = config
        log.info(
            "kumaru ready",
            extra={"backend": config.backend.name, "model": config.backend.model},
        )
        try:
            yield
        finally:
            app.state.agent.close()

    app = FastAPI(
        title="Kumaru",
        version=__version__,
        description="A local-first, scalable LLM chat agent.",
        lifespan=lifespan,
    )

    if config.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors_origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
        )

    @app.exception_handler(KumaruError)
    async def _kumaru_error(_: Request, exc: KumaruError) -> JSONResponse:
        """Turn deliberate failures into clean JSON the UI can display."""
        log.warning("%s: %s", type(exc).__name__, exc.message)
        return JSONResponse(
            status_code=exc.http_status,
            content=ErrorModel(**exc.to_dict()).model_dump(),
        )

    @app.get("/api/health", response_model=HealthModel, tags=["meta"])
    def health(request: Request) -> HealthModel:
        detail = dict(request.app.state.agent.health())
        ready = bool(detail.pop("ready", True))
        return HealthModel(
            status="ok" if ready else "degraded",
            version=__version__,
            backend=str(detail.pop("backend", config.backend.name)),
            model=detail.pop("model", None) or config.backend.model or None,
            ready=ready,
            detail=detail,
        )

    @app.get("/api/config", tags=["meta"])
    def get_config() -> dict[str, object]:
        """Non-secret runtime settings, used by the UI's settings panel."""
        data = config.to_dict()
        data["backend"].pop("extra", None)
        return data

    app.include_router(chat_router)

    if config.server.serve_ui and UI_DIR.is_dir():
        # Mounted last so /api/* always wins.
        app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
    elif config.server.serve_ui:  # pragma: no cover - packaging edge case
        log.warning("UI directory not found at %s; serving API only", UI_DIR)

    return app
