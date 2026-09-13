"""Chat endpoints.

Streaming is exposed as Server-Sent Events because it is the simplest protocol
that survives proxies, needs no client library, and reconnects cleanly - the
browser's built-in ``EventSource`` semantics are reproduced in ``ui/app.js``
with ``fetch`` so that a POST body can be sent.

Each SSE frame carries one JSON object:

    {"delta": "..."}                      incremental text
    {"done": true, "usage": {...}}        final frame
    {"error": "...", "message": "..."}    the generation failed mid-stream

Backends are synchronous and blocking. Starlette runs a sync generator passed
to ``StreamingResponse`` in a worker thread, so one slow generation does not
block the event loop.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from kumaru.agent.chat import ChatAgent
from kumaru.core.errors import KumaruError
from kumaru.core.logging import get_logger
from kumaru.server.schemas import (
    ChatRequestModel,
    ChatResponseModel,
    HistoryResponseModel,
    MessageModel,
    UsageModel,
)

log = get_logger("server.routes")

router = APIRouter(prefix="/api", tags=["chat"])


def get_agent(request: Request) -> ChatAgent:
    """The single process-wide agent, created during app startup."""
    return request.app.state.agent


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat", response_model=ChatResponseModel)
def chat(body: ChatRequestModel, request: Request) -> ChatResponseModel:
    """Non-streamed chat. Simplest possible client integration."""
    agent = get_agent(request)
    reply, usage = agent.ask_with_usage(body.message, session_id=body.session_id)
    return ChatResponseModel(
        reply=reply,
        session_id=body.session_id,
        usage=UsageModel(**usage.to_dict()),
    )


@router.post("/chat/stream")
def chat_stream(body: ChatRequestModel, request: Request) -> StreamingResponse:
    """Streamed chat over SSE."""
    agent = get_agent(request)

    def events() -> Iterator[str]:
        try:
            for chunk in agent.stream(body.message, session_id=body.session_id):
                if chunk.delta:
                    yield _sse({"delta": chunk.delta})
                if chunk.done:
                    usage = chunk.usage.to_dict() if chunk.usage else {}
                    yield _sse({"done": True, "usage": usage})
        except KumaruError as exc:
            log.warning("stream failed: %s", exc.message)
            yield _sse(exc.to_dict())
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to the UI
            log.exception("unexpected stream failure")
            yield _sse({"error": "InternalError", "message": str(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stops nginx buffering the stream into one lump.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{session_id}", response_model=HistoryResponseModel)
def history(session_id: str, request: Request) -> HistoryResponseModel:
    """Replay a session, so a page reload does not lose the conversation."""
    agent = get_agent(request)
    return HistoryResponseModel(
        session_id=session_id,
        messages=[MessageModel.from_core(m) for m in agent.history(session_id)],
    )


@router.delete("/history/{session_id}")
def clear_history(session_id: str, request: Request) -> dict[str, str]:
    """Forget one session."""
    get_agent(request).reset(session_id)
    return {"status": "cleared", "session_id": session_id}
