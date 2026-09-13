"""The agent: turns "a user said X in session S" into a reply.

This is the only class the CLI and the HTTP layer need. It owns the loop

    history -> prompt -> backend -> reply -> history

and nothing else. Tool use, retrieval and planning are deliberately absent in
v1; they are added by wrapping or subclassing this class, not by growing it.

Memory is written *after* a successful generation. If the backend fails, the
history is left untouched so the user can simply retry.
"""

from __future__ import annotations

from collections.abc import Iterator

from kumaru.agent.memory import ConversationMemory
from kumaru.agent.prompt import build_messages
from kumaru.backends import Backend, load_backend
from kumaru.core.config import KumaruConfig
from kumaru.core.logging import get_logger
from kumaru.core.types import ChatRequest, Message, StreamChunk, Usage

log = get_logger("agent.chat")

DEFAULT_SESSION = "default"


class ChatAgent:
    """Stateful chat over a stateless backend."""

    def __init__(
        self,
        config: KumaruConfig,
        *,
        backend: Backend | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.config = config
        self.backend = backend or load_backend(config.backend)
        self.memory = memory or ConversationMemory(
            max_turns=config.agent.max_history_turns,
            max_chars=config.agent.max_history_chars,
        )

    def _request(self, session_id: str, user_input: str) -> ChatRequest:
        agent = self.config.agent
        return ChatRequest(
            messages=build_messages(
                agent.system_prompt, self.memory.history(session_id), user_input
            ),
            max_tokens=agent.max_tokens,
            temperature=agent.temperature,
            top_p=agent.top_p,
            stop=list(agent.stop),
        )

    def _remember(self, session_id: str, user_input: str, reply: str) -> None:
        self.memory.add(session_id, Message.user(user_input.strip()))
        self.memory.add(session_id, Message.assistant(reply))

    def ask_with_usage(
        self, user_input: str, *, session_id: str = DEFAULT_SESSION
    ) -> tuple[str, Usage]:
        """Send one turn and return ``(reply, usage)``."""
        request = self._request(session_id, user_input)
        result = self.backend.generate(request)
        self._remember(session_id, user_input, result.text)
        log.info(
            "chat turn",
            extra={"session": session_id, "tokens": result.usage.total_tokens},
        )
        return result.text, result.usage

    def ask(self, user_input: str, *, session_id: str = DEFAULT_SESSION) -> str:
        """Send one turn and return the complete reply."""
        return self.ask_with_usage(user_input, session_id=session_id)[0]

    def stream(
        self, user_input: str, *, session_id: str = DEFAULT_SESSION
    ) -> Iterator[StreamChunk]:
        """Send one turn and yield the reply as it arrives.

        The caller must consume the iterator to completion for the exchange to
        be written to memory - that is what keeps an aborted request from
        poisoning the history with a half-finished answer.
        """
        request = self._request(session_id, user_input)
        parts: list[str] = []
        usage = Usage()
        for chunk in self.backend.stream(request):
            if chunk.delta:
                parts.append(chunk.delta)
            if chunk.usage is not None:
                usage = chunk.usage
            yield chunk
        self._remember(session_id, user_input, "".join(parts))
        log.info(
            "chat turn (streamed)",
            extra={"session": session_id, "tokens": usage.total_tokens},
        )

    def reset(self, session_id: str | None = None) -> None:
        """Forget one session, or all of them when ``session_id`` is ``None``."""
        if session_id is None:
            self.memory.clear_all()
        else:
            self.memory.clear(session_id)

    def history(self, session_id: str = DEFAULT_SESSION) -> list[Message]:
        return self.memory.history(session_id)

    def health(self) -> dict[str, object]:
        info = dict(self.backend.health())
        info["sessions"] = len(self.memory)
        return info

    def close(self) -> None:
        self.backend.close()

    def __enter__(self) -> ChatAgent:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
