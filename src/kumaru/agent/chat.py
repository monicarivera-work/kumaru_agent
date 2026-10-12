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
from pathlib import Path

from kumaru.agent.knowledge import load_pack, reference_context
from kumaru.agent.memory import get_policy
from kumaru.agent.prompt import build_system_prompt, ensure_system_message
from kumaru.agent.tools import ToolBelt
from kumaru.backends.base import Backend
from kumaru.core.config import KumaruConfig
from kumaru.core.errors import ContextLengthExceededError
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
        self.knowledge = [load_pack(Path(path)) for path in config.agent.knowledge_files]

    # -- prompt assembly ----------------------------------------------------

    def prepare(
        self,
        messages: Sequence[Message],
        *,
        extra_context: str | None = None,
        reserve_tokens: int | None = None,
    ) -> list[Message]:
        """Normalize and budget the conversation. Pure and side-effect free.

        Kept public and separate from generation so tests can assert on the
        exact prompt without running a model — the fastest way to catch the
        class of bug where the model is fine but the prompt is wrong.

        `reserve_tokens` is the room kept free for the reply; it defaults to the
        configured `max_tokens`. Thinking replies are far longer, so the
        generation methods pass the effective value for the current request.
        At most half the window is reserved: `max_tokens` is a ceiling, not a
        forecast, and the generation methods trim it to the room actually left.
        """
        references = reference_context(self.knowledge, self.config.agent.max_knowledge_chars)
        context = "\n\n".join(part for part in (references, extra_context) if part)
        system_prompt = build_system_prompt(
            self.config.agent,
            tool_descriptions=self.tools.descriptions(),
            extra_context=context,
        )
        normalized = ensure_system_message(messages, system_prompt)

        # Reserve room for the answer: the budget must cover prompt *and*
        # completion, or generation is truncated right after it starts. Capped
        # at half the window: reserving qwen36's 32k cap in full would leave a
        # 32k window only the 512-token floor for history.
        window = self.backend.context_length
        reserved = reserve_tokens if reserve_tokens is not None else self.config.generation.max_tokens
        reserved = min(reserved, window // 2)
        budget = min(self.config.agent.max_history_tokens, max(512, window - reserved - 128))

        prepared = self.memory.apply(normalized, budget, self.backend.count_tokens)
        if (self.config.agent.expertise or self.knowledge) and self.backend.count_tokens(prepared) > budget:
            raise ContextLengthExceededError(
                "Engineering guidance, references and recent messages exceed the prompt budget.",
                hint="Reduce agent.max_knowledge_chars or expertise, shorten the request, "
                "or increase the context/history budget.",
            )
        return prepared

    def resolve_params(self, **overrides) -> GenerationParams:
        """Resolve the decoding parameters for one request.

        Precedence, lowest to highest: ``generation`` (config), then
        ``thinking_mode`` (config, applied only when thinking is on), then the
        per-request overrides. Reasoning models want different sampling when
        they think — Qwen recommends temperature 1.0 thinking vs 0.7 not — and
        many more output tokens, so one flat preset cannot serve both modes.

        Public so the benchmark runner decodes exactly the way chat does.
        """
        base = self.config.generation
        thinking = overrides.get("thinking")
        if thinking is None:
            thinking = base.thinking
        if thinking:
            base = base.merged(**self.config.thinking_mode.model_dump(exclude_none=True))
        return base.merged(**overrides)

    def _fit(
        self, messages: Sequence[Message], extra_context: str | None, overrides: dict
    ) -> tuple[list[Message], GenerationParams]:
        """Resolve the params, budget the history, then cap the reply to the room left.

        Strict servers (vLLM, OpenAI) reject a request whose prompt plus
        `max_tokens` exceeds the window instead of stopping at its end.
        """
        params = self.resolve_params(**overrides)
        prepared = self.prepare(messages, extra_context=extra_context, reserve_tokens=params.max_tokens)
        room = self.backend.context_length - self.backend.count_tokens(prepared)
        if params.max_tokens > room:
            params = params.merged(max_tokens=max(room, 1))
        return prepared, params

    # -- generation ---------------------------------------------------------

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        extra_context: str | None = None,
        **overrides,
    ) -> ChatResult:
        """Produce a full reply."""
        prepared, params = self._fit(messages, extra_context, overrides)
        result = await self.backend.generate(prepared, params)
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
        prepared, params = self._fit(messages, extra_context, overrides)

        started = time.perf_counter()
        ttft_ms: float | None = None
        emitted = 0

        async for chunk in self.backend.stream(prepared, params):
            # With thinking on, the first token is a thought, and that is what
            # the user perceives as "it started answering".
            if chunk.delta or chunk.reasoning:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000
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
            "expertise": self.config.agent.expertise,
            "knowledge_topics": [pack.topic for pack in self.knowledge],
            "generation": self.config.generation.model_dump(),
            "thinking_mode": self.config.thinking_mode.model_dump(exclude_none=True),
        }
