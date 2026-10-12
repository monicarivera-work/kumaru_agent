"""Context-window management: deciding what the model gets to remember.

The constraint
--------------
A model has a fixed context window. Attention cost grows ~O(n^2) with sequence
length and the KV cache grows ~O(n) in memory, so long conversations get slower
and more expensive *per token* and eventually stop fitting at all. Something
must be dropped. The only question is *what*, and that choice is a policy.

Policies implemented here
-------------------------
``none``
    Send everything. Correct until it is not; fails loudly when it overflows.
``drop_oldest``
    Remove from the front until it fits. Cheap, lossy, and surprisingly strong,
    because recency dominates relevance in most chats.
``sliding_window`` (default)
    Always keep the system prompt and the most recent turns; drop from the
    middle. Preserves both the instructions (which govern behaviour) and the
    immediate context (which governs the answer) — the two regions that
    actually matter.

Where this goes next
--------------------
The `MemoryPolicy` interface is the extension point for the standard upgrades:
*summarize* the dropped middle into a paragraph and re-inject it, or *retrieve*
only the relevant history from a vector store (RAG). Both slot in as new
policies without touching `ChatAgent`. See docs/SCALABILITY.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from kumaru.core.errors import ContextLengthExceededError
from kumaru.core.logging import get_logger
from kumaru.core.types import Message, Role

logger = get_logger(__name__)

TokenCounter = Callable[[Sequence[Message]], int]


class MemoryPolicy(ABC):
    """Decides which messages survive into the prompt."""

    name: str = "base"

    @abstractmethod
    def apply(
        self, messages: Sequence[Message], budget: int, count_tokens: TokenCounter
    ) -> list[Message]:
        """Return a sub-list of `messages` that fits within `budget` tokens."""


class NoTrimPolicy(MemoryPolicy):
    """Pass everything through; raise if it cannot fit."""

    name = "none"

    def apply(
        self, messages: Sequence[Message], budget: int, count_tokens: TokenCounter
    ) -> list[Message]:
        used = count_tokens(messages)
        if used > budget:
            raise ContextLengthExceededError(
                f"Conversation needs {used} tokens but the budget is {budget}.",
                hint="Switch agent.memory_policy to 'sliding_window', or raise the budget.",
            )
        return list(messages)


class DropOldestPolicy(MemoryPolicy):
    """Drop from the front (after the system message) until it fits."""

    name = "drop_oldest"

    def apply(
        self, messages: Sequence[Message], budget: int, count_tokens: TokenCounter
    ) -> list[Message]:
        system = [m for m in messages if m.role == Role.SYSTEM][:1]
        body = [m for m in messages if m.role != Role.SYSTEM]

        while body and count_tokens([*system, *body]) > budget:
            body.pop(0)
        return [*system, *body]


class SlidingWindowPolicy(MemoryPolicy):
    """Keep the system prompt plus the newest turns; drop the middle.

    `min_recent_turns` guarantees the immediate exchange survives even under a
    tiny budget, so the model never answers a question it can no longer see.
    """

    name = "sliding_window"

    def __init__(self, min_recent_turns: int = 2) -> None:
        self.min_recent_turns = min_recent_turns

    def apply(
        self, messages: Sequence[Message], budget: int, count_tokens: TokenCounter
    ) -> list[Message]:
        system = [m for m in messages if m.role == Role.SYSTEM][:1]
        body = [m for m in messages if m.role != Role.SYSTEM]

        if count_tokens([*system, *body]) <= budget:
            return [*system, *body]

        # Grow a window backwards from the newest message while it still fits.
        kept: list[Message] = []
        for msg in reversed(body):
            candidate = [msg, *kept]
            if count_tokens([*system, *candidate]) > budget and len(kept) >= self.min_recent_turns:
                break
            kept = candidate

        dropped = len(body) - len(kept)
        if dropped:
            logger.info(
                "memory: dropped %d older message(s) to fit %d-token budget", dropped, budget
            )
        return [*system, *kept]


POLICIES: dict[str, Callable[[], MemoryPolicy]] = {
    "none": NoTrimPolicy,
    "drop_oldest": DropOldestPolicy,
    "sliding_window": SlidingWindowPolicy,
}


def get_policy(name: str) -> MemoryPolicy:
    """Look up a policy by config name, defaulting to the safe one."""
    factory = POLICIES.get(name)
    if factory is None:
        logger.warning("unknown memory_policy '%s'; using sliding_window", name)
        factory = SlidingWindowPolicy
    return factory()
