"""FastAPI application factory and shared runtime state.

Two decisions here shape how the server *feels* and how it scales.

**Non-blocking startup.** Loading a 7B model takes 30-60 seconds. If that
happened inside a synchronous startup hook, the port would not open until it
finished and every client would see connection-refused. Instead the model loads
in a background task, `/healthz` reports `loading`, and the UI shows a progress
state. Requests arriving early simply await readiness.

**Explicit concurrency limiting.** A semaphore caps in-flight generations. This
is not a nicety: on a single GPU, admitting unlimited concurrent requests makes
every one of them slower and eventually triggers a CUDA OOM that kills the
process. Bounded admission converts a hard crash into graceful queueing — the
difference between a service that degrades and one that dies.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from kumaru import __version__
from kumaru.agent.chat import ChatAgent
from kumaru.backends.base import Backend
from kumaru.core.config import REPO_ROOT, KumaruConfig, load_config
from kumaru.core.errors import KumaruError
from kumaru.core.logging import get_logger, setup_logging
from kumaru.core.registry import BACKENDS

logger = get_logger(__name__)
UI_DIR = REPO_ROOT / "ui"


class Metrics:
    """In-process counters. Deliberately trivial.

    Enough to answer "is it working, and how fast?" with zero dependencies.
    When you outgrow it, swap in `prometheus_client` — the call sites do not
    change.
    """

    def __init__(self) -> None:
        self.requests = 0
        self.errors = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_latency_ms = 0.0
        self.started_at = time.time()

    def record(self, *, prompt: int, completion: int, latency_ms: float) -> None:
        self.requests += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_latency_ms += latency_ms

    def snapshot(self) -> dict[str, Any]:
        uptime = time.time() - self.started_at
        return {
            "uptime_s": round(uptime, 1),
            "requests": self.requests,
            "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "avg_latency_ms": round(self.total_latency_ms / max(1, self.requests), 1),
            "avg_tokens_per_second": round(
                self.completion_tokens / max(0.001, self.total_latency_ms / 1000), 2
            ),
        }


class AppState:
    """Everything the request handlers need, created once per process."""

    def __init__(self, config: KumaruConfig) -> None:
        self.config = config
        self.backend: Backend | None = None
        self.agent: ChatAgent | None = None
        self.metrics = Metrics()
        self.semaphore = asyncio.Semaphore(config.server.max_concurrent_requests)
        self.load_error: str | None = None
        self._load_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self.agent is not None and self.load_error is None

    def start_loading(self) -> None:
        """Kick off model loading without blocking the event loop."""
        if self._load_task is None:
            self._load_task = asyncio.create_task(self._load())

    async def _load(self) -> None:
        async with self._lock:
            if self.ready:
                return
            try:
                cfg = self.config.backend
                logger.info("initializing backend '%s' -> %s", cfg.name, cfg.model)
                backend = BACKENDS.create(cfg.name, cfg)
                await backend.load()
                self.backend = backend
                self.agent = ChatAgent(backend, self.config)
                self.load_error = None
                logger.info("server ready at http://%s:%d", self.config.server.host, self.config.server.port)
            except Exception as exc:  # noqa: BLE001 - reported via /healthz
                self.load_error = str(exc)
                logger.error("backend failed to load: %s", exc)

    async def ensure_ready(self, timeout_s: float = 600.0) -> ChatAgent:
        """Await model readiness, raising a useful error if it failed."""
        self.start_loading()
        if self._load_task is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(self._load_task), timeout=timeout_s)

        if self.load_error:
            raise KumaruError(
                f"Backend failed to load: {self.load_error}",
                hint="Check the server logs, or try `--backend echo` to verify the rest of the stack.",
            )
        if self.agent is None:
            raise KumaruError("Backend is still loading.", hint="Retry in a few seconds.")
        return self.agent

    async def swap_backend(self, **overrides: Any) -> None:
        """Hot-swap the model without restarting the process.

        Powers the UI's model switcher. The old backend is closed first so its
        VRAM is released before the new one allocates — doing it the other way
        round reliably OOMs on a single GPU.
        """
        async with self._lock:
            if self.backend is not None:
                await self.backend.aclose()
            self.backend = None
            self.agent = None
            self.load_error = None
            self._load_task = None

            updated = self.config.backend.model_copy(update=overrides)
            self.config = self.config.model_copy(update={"backend": updated})

        await self._load()
        if self.load_error:
            raise KumaruError(f"Failed to load new backend: {self.load_error}")

    async def shutdown(self) -> None:
        if self._load_task is not None:
            self._load_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._load_task
        if self.backend is not None:
            await self.backend.aclose()


def create_app(config: KumaruConfig | None = None) -> FastAPI:
    """Build the ASGI application.

    A factory (not a module-level `app`) so tests can construct isolated
    instances with different configs in the same process.
    """
    config = config or load_config()
    setup_logging(config.telemetry.log_level, fmt=config.telemetry.log_format)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state: AppState = app.state.kumaru
        state.start_loading()  # background: does not delay the port opening
        try:
            yield
        finally:
            await state.shutdown()

    app = FastAPI(
        title="Kumaru",
        version=__version__,
        description="Local-first, scalable LLM agent with an OpenAI-compatible API.",
        lifespan=lifespan,
    )
    app.state.kumaru = AppState(config)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(KumaruError)
    async def kumaru_error_handler(request: Request, exc: KumaruError) -> JSONResponse:
        """Map domain errors onto HTTP without leaking stack traces."""
        request.app.state.kumaru.metrics.errors += 1
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": {
                    "message": exc.message,
                    "type": type(exc).__name__,
                    "hint": exc.hint,
                }
            },
        )

    from kumaru.server.routes_chat import router as chat_router

    app.include_router(chat_router)

    if config.server.serve_ui and UI_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(str(UI_DIR / "index.html"))

    return app


def run(config: KumaruConfig | None = None) -> None:
    """Start uvicorn. Used by `kumaru serve`."""
    import uvicorn

    config = config or load_config()
    uvicorn.run(
        create_app(config),
        host=config.server.host,
        port=config.server.port,
        log_level=config.telemetry.log_level.lower(),
        access_log=False,  # Kumaru logs its own, more useful, request lines
    )
