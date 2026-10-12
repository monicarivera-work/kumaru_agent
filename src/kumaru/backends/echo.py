"""A dependency-free backend used for tests, CI, and install verification.

Why ship a fake model
---------------------
It gives you a *fast, deterministic, offline* way to exercise every layer above
inference: HTTP routing, SSE framing, memory trimming, UI rendering, error
paths. Those layers contain most of the bugs, and none of them need a GPU to
test. Without this, the test suite would need 15 GB of weights and a CUDA device
to answer "did I break the streaming protocol?" — so in practice nobody would
run it.

It is also the first thing to try when something breaks:
``kumaru chat --backend echo``. If that works, your problem is in the model
layer; if it does not, your problem is in the application layer. That single
bisection saves hours.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Sequence

from kumaru.backends.base import Backend, chunks_from_text
from kumaru.core.config import BackendConfig
from kumaru.core.registry import BACKENDS
from kumaru.core.types import BackendInfo, GenerationParams, Message, StreamChunk, Usage


@BACKENDS.register("echo")
class EchoBackend(Backend):
    """Returns a deterministic, structured reply without loading any model."""

    name = "echo"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        # Simulated inter-token delay. Non-zero by default so the UI's streaming
        # path is exercised realistically rather than completing instantly.
        self.delay_s: float = float(config.extra.get("delay_s", 0.01))

    async def _load(self) -> None:
        return None

    def info(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            model_id="echo",
            device="cpu",
            dtype="none",
            context_length=self.config.max_context_length,
            supports_streaming=True,
            extra={"purpose": "testing and install verification"},
        )

    async def stream(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> AsyncIterator[StreamChunk]:
        last_user = next(
            (m.content for m in reversed(messages) if m.role.value == "user"), ""
        )
        # A stable hash makes replies deterministic across runs, which is what
        # lets tests assert on them.
        digest = hashlib.sha256(last_user.encode("utf-8")).hexdigest()[:8]
        reply = (
            f"[echo:{digest}] You said: {last_user.strip() or '(nothing)'}\n"
            f"Context: {len(messages)} message(s), "
            f"~{self.count_tokens(messages)} prompt tokens, "
            f"max_tokens={params.max_tokens}, temperature={params.temperature}."
        )

        emitted = 0
        if params.thinking:
            # A stand-in "thought" so the reasoning channel (SSE field, UI
            # panel, budget accounting) can be tested without a model.
            thought = (
                f"The user asked: {last_user.strip()[:80] or '(nothing)'}. "
                "Echo cannot reason; this text only exercises the reasoning channel."
            )
            pieces = list(chunks_from_text(thought, size=8))
            if params.thinking_budget is not None:
                pieces = pieces[: params.thinking_budget]
            for piece in pieces[: params.max_tokens]:
                if self.delay_s:
                    await asyncio.sleep(self.delay_s)
                emitted += 1
                yield StreamChunk(reasoning=piece)

        # Like a real model, max_tokens caps the thought and the answer together,
        # so a long thought can leave no room for the answer.
        answer = list(chunks_from_text(reply, size=6))
        room = params.max_tokens - emitted
        for piece in answer[:room]:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            emitted += 1
            yield StreamChunk(delta=piece)

        yield StreamChunk(
            finish_reason="length" if len(answer) > room else "stop",
            usage=Usage(
                prompt_tokens=self.count_tokens(messages), completion_tokens=emitted
            ),
        )
