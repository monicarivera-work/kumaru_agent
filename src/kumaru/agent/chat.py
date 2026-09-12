"""`ChatAgent` — orchestrates one conversational turn.

Its job, in order: normalize the system prompt, fit the history into the context
budget, call the backend, and measure what happened. It owns *policy*; the
backend owns *mechanism*.

Statelessness is a design decision, not an omission
---------------------------------------------------
`ChatAgent` stores no conversation history. Callers pass the full message list
on every turn, exactly like the OpenAI API. This is what makes horizontal
scaling trivial: any replica can serve any request, so you scale by starting
more processes behind a load balancer with no session affinity, no shared cache,
and no sticky routing. Persistence, when you want it, belongs in a store the
caller owns — not smuggled into the agent.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence

from kumaru.agent.memory import get_policy
from kumaru.agent.prompt import build_system_prompt, ensure_system_message
from kumaru.agent.tools import ToolBelt
from kumaru.backends.base import Backend
from kumaru.core.config import KumaruConfig
from kumaru.core.logging import get_logger
from kumaru.core.types import ChatResult, GenerationParams, Message, StreamChunk

logger = get_logger(__name__)


class ChatAgent:
    """Turns a message list into a reply, using a backend and a memory policy."""

    def __init__(self, backend: Backend, config: KumaruConfig) -> None:
        self.backend = backend
        self.config = config
        self.memory = get_policy(config.agent.memory_policy)
        self.tools = ToolBelt(config.agent.enabled_tools)

    # -- prompt assembly ----------------------------------------------------

    def prepare(self, messages: Sequence[Message], *, extra_context: str | None = None) -> list[Message]:
        """Normalize and budget the conversation. Pure and side-effect free.

        Kept public and separate from generation so tests can assert on the
        exact prompt without running a model — the fastest way to catch the
        class of bug where the model is fine but the prompt is wrong.
        """
        system_prompt = build_system_prompt(
            self.config.agent,
            tool_descriptions=self.tools.descriptions(),
            extra_context=extra_context,
        )
        normalized = ensure_system_message(messages, system_prompt)

        # Reserve room for the answer: the budget must cover prompt *and*
        # completion, or generation is truncated right after it starts.
        window = self.backend.context_length
        reserved = self.config.generation.max_tokens
        budget = min(self.config.agent.max_history_tokens, max(512, window - reserved - 128))

        return self.memory.apply(normalized, budget, self.backend.count_tokens)

    def _params(self, **overrides) -> GenerationParams:
        return self.config.generation.merged(**overrides)

    # -- generation ---------------------------------------------------------

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        extra_context: str | None = None,
        **overrides,
    ) -> ChatResult:
        """Produce a full reply."""
        prepared = self.prepare(messages, extra_context=extra_context)
        result = await self.backend.generate(prepared, self._params(**overrides))
        logger.info(
            "completed: %d prompt + %d completion tokens in %.0fms (%.1f tok/s)",
            result.usage.prompt_tokens,
            result.usage.completion_tokens,
            result.latency_ms,
            result.tokens_per_second,
        )
        return result

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        extra_context: str | None = None,
        **overrides,
    ) -> AsyncIterator[StreamChunk]:
        """Yield the reply incrementally.

        This is the path the UI uses. Time-to-first-token dominates *perceived*
        speed far more than total tokens/second does, so it is measured and
        logged on every request.
        """
        prepared = self.prepare(messages, extra_context=extra_context)
        params = self._params(**overrides)

        started = time.perf_counter()
        ttft_ms: float | None = None
        emitted = 0

        async for chunk in self.backend.stream(prepared, params):
            if chunk.delta and ttft_ms is None:
                ttft_ms = (time.perf_counter() - started) * 1000
            if chunk.delta:
                emitted += 1
            yield chunk

        total_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "streamed: %d chunks in %.0fms (ttft %.0fms)",
            emitted,
            total_ms,
            ttft_ms if ttft_ms is not None else -1,
        )

    # -- introspection ------------------------------------------------------

    def describe(self) -> dict:
        """Snapshot of the agent's effective configuration, shown in the UI."""
        return {
            "backend": self.backend.info().model_dump(),
            "memory_policy": self.memory.name,
            "max_history_tokens": self.config.agent.max_history_tokens,
            "tools": list(self.tools.tools),
            "generation": self.config.generation.model_dump(),
        }
