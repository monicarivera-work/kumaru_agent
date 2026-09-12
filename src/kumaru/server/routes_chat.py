"""Chat endpoints: OpenAI-compatible completion, streaming, health, and control.

Endpoints
---------
``POST /v1/chat/completions``  generation, streaming or not
``GET  /v1/models``            what is loaded
``GET  /v1/info``              full effective configuration (UI status panel)
``POST /v1/backend``           hot-swap model/backend at runtime
``GET  /healthz``              liveness + readiness
``GET  /metrics``              in-process counters

On the streaming format
-----------------------
Server-Sent Events, framed exactly as OpenAI frames them: one `data: {json}`
line per chunk, terminated by `data: [DONE]`. SSE is chosen over WebSockets
because generation is strictly one-directional, and SSE survives proxies, works
with plain `fetch`, and reconnects on its own. WebSockets would add a stateful
connection for no gain.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kumaru import __version__
from kumaru.core.errors import KumaruError
from kumaru.core.logging import get_logger
from kumaru.core.registry import BACKENDS, TOOLS
from kumaru.server.app import AppState
from kumaru.server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    ModelCard,
    ModelList,
    new_completion_id,
    sse_chunk,
)

logger = get_logger(__name__)
router = APIRouter()


def get_state(request: Request) -> AppState:
    return request.app.state.kumaru


async def require_auth(
    request: Request, authorization: str | None = Header(default=None)
) -> None:
    """Optional bearer-token gate.

    Disabled unless `server.api_key` is set, because the default deployment is
    localhost-only. The moment you bind to 0.0.0.0, set it — an open inference
    endpoint is an open compute account.
    """
    expected = request.app.state.kumaru.config.server.api_key
    if not expected:
        return
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if supplied != expected:
        raise KumaruError("Invalid or missing API key.", hint="Send 'Authorization: Bearer <key>'.")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
async def chat_completions(body: ChatCompletionRequest, request: Request):
    state = get_state(request)
    agent = await state.ensure_ready()

    if not body.messages:
        raise KumaruError("`messages` must contain at least one message.")

    overrides = body.generation_overrides()
    messages = body.to_domain()

    if body.stream:
        return StreamingResponse(
            _stream_sse(state, agent, messages, overrides, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Without this, nginx buffers the whole response and streaming
                # silently degrades to a single delayed blob.
                "X-Accel-Buffering": "no",
            },
        )

    # The semaphore bounds concurrent GPU work; see AppState for why.
    async with state.semaphore:
        result = await agent.complete(messages, **overrides)

    state.metrics.record(
        prompt=result.usage.prompt_tokens,
        completion=result.usage.completion_tokens,
        latency_ms=result.latency_ms,
    )
    return ChatCompletionResponse.from_result(result)


async def _stream_sse(
    state: AppState,
    agent: Any,
    messages: list,
    overrides: dict,
    request: Request,
) -> AsyncIterator[str]:
    """Render the agent's stream as Server-Sent Events."""
    completion_id = new_completion_id()
    model = state.config.backend.model
    started = time.perf_counter()
    ttft_ms: float | None = None
    usage = None

    def frame(payload: dict) -> str:
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    try:
        async with state.semaphore:
            # An initial role-only chunk, as OpenAI does: it lets clients render
            # the assistant bubble immediately, before any text exists.
            yield frame(sse_chunk(completion_id=completion_id, model=model, role="assistant"))

            async for chunk in agent.stream(messages, **overrides):
                # Stop generating the moment the client hangs up. Without this
                # check the GPU keeps working on an answer nobody will read.
                if await request.is_disconnected():
                    logger.info("client disconnected; aborting generation")
                    break

                if chunk.delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    yield frame(
                        sse_chunk(completion_id=completion_id, model=model, delta=chunk.delta)
                    )
                if chunk.usage:
                    usage = chunk.usage
                if chunk.finish_reason:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    completion_tokens = usage.completion_tokens if usage else 0
                    yield frame(
                        sse_chunk(
                            completion_id=completion_id,
                            model=model,
                            finish_reason=chunk.finish_reason,
                            extra={
                                "latency_ms": round(elapsed_ms, 1),
                                "time_to_first_token_ms": (
                                    round(ttft_ms, 1) if ttft_ms else None
                                ),
                                "prompt_tokens": usage.prompt_tokens if usage else 0,
                                "completion_tokens": completion_tokens,
                                "tokens_per_second": round(
                                    completion_tokens / max(0.001, elapsed_ms / 1000), 2
                                ),
                            },
                        )
                    )
                    state.metrics.record(
                        prompt=usage.prompt_tokens if usage else 0,
                        completion=completion_tokens,
                        latency_ms=elapsed_ms,
                    )
    except asyncio.CancelledError:
        raise
    except KumaruError as exc:
        state.metrics.errors += 1
        yield frame({"error": {"message": exc.message, "type": type(exc).__name__, "hint": exc.hint}})
    except Exception as exc:  # noqa: BLE001
        state.metrics.errors += 1
        logger.exception("stream failed")
        yield frame({"error": {"message": str(exc), "type": type(exc).__name__}})

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Introspection and control
# ---------------------------------------------------------------------------


@router.get("/v1/models", response_model=ModelList)
async def list_models(request: Request) -> ModelList:
    state = get_state(request)
    return ModelList(data=[ModelCard(id=state.config.backend.model)])


@router.get("/v1/info")
async def info(request: Request) -> dict:
    """Everything the UI's status panel needs, in one round trip."""
    state = get_state(request)
    payload: dict[str, Any] = {
        "version": __version__,
        "ready": state.ready,
        "loading": not state.ready and state.load_error is None,
        "error": state.load_error,
        "available_backends": BACKENDS.names(),
        "available_tools": TOOLS.names(),
        "config": {
            "backend": state.config.backend.model_dump(exclude={"api_key"}),
            "generation": state.config.generation.model_dump(),
            "agent": state.config.agent.model_dump(),
        },
        "metrics": state.metrics.snapshot(),
    }
    if state.agent is not None:
        payload["agent"] = state.agent.describe()
    return payload


@router.post("/v1/backend", dependencies=[Depends(require_auth)])
async def swap_backend(payload: dict, request: Request) -> dict:
    """Switch model or backend at runtime, without restarting."""
    state = get_state(request)
    allowed = {"name", "model", "device", "dtype", "load_in_4bit", "adapter_path", "base_url", "model_path"}
    overrides = {k: v for k, v in payload.items() if k in allowed and v is not None}
    if not overrides:
        raise KumaruError("No valid fields supplied.", hint=f"Allowed: {sorted(allowed)}")

    logger.info("swapping backend: %s", overrides)
    await state.swap_backend(**overrides)
    return {"ok": True, "backend": state.config.backend.model_dump(exclude={"api_key"})}


@router.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> JSONResponse:
    """Liveness and readiness in one probe.

    Returns 200 while loading so orchestrators do not kill a process that is
    merely slow to warm up; `status` distinguishes the states.
    """
    state = get_state(request)
    status = "error" if state.load_error else ("ok" if state.ready else "loading")
    body = HealthResponse(
        status=status,
        version=__version__,
        backend=state.config.backend.name,
        model=state.config.backend.model,
        loaded=state.ready,
        detail=state.load_error,
    )
    return JSONResponse(status_code=503 if status == "error" else 200, content=body.model_dump())


@router.get("/metrics")
async def metrics(request: Request) -> dict:
    return get_state(request).metrics.snapshot()
