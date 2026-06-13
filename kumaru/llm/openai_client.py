"""
kumaru/llm/openai_client.py
---------------------------
OpenAI implementation of the BaseLLMClient interface.

Concept: API calls and retries
------------------------------
Production agents need to handle transient API failures gracefully.
We implement a simple exponential back-off retry loop here so you can see
the pattern even though the logic is kept minimal for readability.

Concept: Tool / function calling
---------------------------------
Modern OpenAI models can "call" tools.  When the model decides it needs a
tool it returns a response with ``finish_reason == "tool_calls"`` and a list
of calls it wants to make.  The agent is responsible for:
  1. Detecting that the model wants to call a tool.
  2. Running the tool with the given arguments.
  3. Appending a "tool" role message with the result.
  4. Sending the updated conversation back to the model.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from kumaru.config import LLMConfig
from kumaru.llm.base import BaseLLMClient, LLMResponse, Message
from kumaru.logger import get_logger

log = get_logger(__name__)


class OpenAIClient(BaseLLMClient):
    """Wraps the OpenAI Chat Completions API.

    Args:
        config: :class:`~kumaru.config.LLMConfig` with the model name,
                temperature, token limit, and API key.
    """

    def __init__(self, config: LLMConfig) -> None:
        # Lazy-import so users who don't have openai installed can still
        # import other parts of the package (e.g. for testing with mocks).
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use OpenAIClient.\n"
                "Install it with:  pip install openai"
            ) from exc

        self._config = config
        self._client = openai.OpenAI(
            api_key=config.api_key,
            timeout=config.timeout,
        )

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Call the OpenAI Chat Completions endpoint with retry logic."""
        wire_messages = [m.to_dict() for m in messages]

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": wire_messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._call_with_retry(kwargs)

        choice = response.choices[0]
        message = choice.message

        tool_calls: list[dict[str, Any]] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            log.debug("Token usage", extra={"usage": usage})

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage=usage,
            raw=response,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        kwargs: dict[str, Any],
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> Any:
        """Call the API, retrying on transient errors with exponential back-off.

        Concept: Exponential back-off
        ------------------------------
        When an API is temporarily overloaded it returns 429 (rate limit) or
        503 (server error).  Retrying immediately makes things worse.
        Instead we wait  base_delay * 2^attempt  seconds between retries,
        adding random jitter to avoid thundering herds in production.
        """
        import openai

        for attempt in range(max_retries):
            try:
                return self._client.chat.completions.create(**kwargs)
            except openai.RateLimitError as exc:
                if attempt == max_retries - 1:
                    raise
                wait = base_delay * (2 ** attempt)
                log.warning(
                    "Rate limit hit; retrying",
                    extra={"attempt": attempt + 1, "wait_seconds": wait},
                )
                time.sleep(wait)
            except openai.APIStatusError as exc:
                if attempt == max_retries - 1 or exc.status_code < 500:
                    raise
                wait = base_delay * (2 ** attempt)
                log.warning(
                    "API server error; retrying",
                    extra={
                        "attempt": attempt + 1,
                        "status": exc.status_code,
                        "wait_seconds": wait,
                    },
                )
                time.sleep(wait)
        # Unreachable, but makes type-checkers happy.
        raise RuntimeError("Exhausted retries")  # pragma: no cover
