"""A dependency-free backend that answers without any model.

It exists for three concrete reasons:

1. ``git clone && pip install -e . && kumaru chat`` must work in under a
   minute, before any weights are downloaded.
2. Tests for the agent, server and UI need a deterministic generator.
3. When a real backend misbehaves, switching to ``echo`` instantly tells you
   whether the problem is the model or everything else.

It streams character-by-character with a small delay so the UI's streaming
path is genuinely exercised.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from kumaru.backends.base import Backend, GenerationResult, estimate_tokens
from kumaru.core.config import BackendConfig
from kumaru.core.types import ChatRequest, Role, StreamChunk, Usage

_BANNER = (
    "Kumaru is running on the 'echo' backend, which has no language model "
    "behind it, so it cannot actually answer questions yet. Point "
    "backend.name at 'openai_compat', 'transformers' or 'llamacpp' in your "
    "config to get real answers."
)


class EchoBackend(Backend):
    """Reflects the last user message back, with a short explanation."""

    name = "echo"

    def __init__(self, config: BackendConfig, *, delay_s: float = 0.01) -> None:
        super().__init__(config)
        self.delay_s = delay_s

    def _reply(self, request: ChatRequest) -> str:
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role is Role.USER),
            "",
        )
        if not last_user.strip():
            return _BANNER
        text = f'You said: "{last_user.strip()}"\n\n{_BANNER}'
        # Respect max_tokens so callers can test truncation behaviour.
        limit = request.max_tokens * 4
        return text[:limit]

    def generate(self, request: ChatRequest) -> GenerationResult:
        text = self._reply(request)
        return GenerationResult(
            text=text,
            usage=Usage(
                prompt_tokens=sum(estimate_tokens(m.content) for m in request.messages),
                completion_tokens=estimate_tokens(text),
            ),
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        text = self._reply(request)
        for index in range(0, len(text), 8):
            if self.delay_s:
                time.sleep(self.delay_s)
            yield StreamChunk(delta=text[index : index + 8])
        yield StreamChunk(
            done=True,
            usage=Usage(
                prompt_tokens=sum(estimate_tokens(m.content) for m in request.messages),
                completion_tokens=estimate_tokens(text),
            ),
        )

    def health(self) -> dict[str, object]:
        return {"backend": self.name, "model": "echo", "ready": True}
