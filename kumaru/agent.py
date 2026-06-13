"""
kumaru/agent.py
---------------
The Kumaru LLM Agent.

Concept: The ReAct pattern
---------------------------
ReAct (Reasoning + Acting) is the most widely-used agent architecture.
Its loop looks like this:

  ┌─────────────────────────────────────────┐
  │  1. REASON  – Ask the LLM: what should  │
  │               I do next?                │
  │  2. ACT     – If the LLM wants a tool,  │
  │               call it.                  │
  │  3. OBSERVE – Add the tool result to    │
  │               memory.                   │
  │  4. REPEAT  – Until the LLM produces a  │
  │               final answer (no tool     │
  │               call).                    │
  └─────────────────────────────────────────┘

This keeps the agent architecture simple and debuggable: every reasoning step
and every tool call is logged and stored in memory, so you can always trace
*why* the agent did what it did.

How to extend the agent
------------------------
* Add a new tool  → create a subclass of ``BaseTool`` and pass it to the
                    ``tools`` parameter of the constructor.
* Change the LLM  → pass a different ``BaseLLMClient`` implementation.
* Add persistence → subclass ``ConversationMemory`` to store history in a DB.
* Add streaming   → override ``_reason()`` and adapt ``OpenAIClient`` to use
                    the streaming API.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from kumaru.config import AgentConfig
from kumaru.llm.base import BaseLLMClient, Message
from kumaru.llm.ollama_client import OllamaClient
from kumaru.llm.openai_client import OpenAIClient
from kumaru.logger import get_logger
from kumaru.memory.conversation import ConversationMemory
from kumaru.tools.base import BaseTool, ToolError


class Agent:
    """The main agent that orchestrates the LLM + memory + tools.

    Args:
        config: :class:`~kumaru.config.AgentConfig` instance.
                Uses sensible defaults if not provided.
        llm:    A :class:`~kumaru.llm.base.BaseLLMClient` instance.
                Defaults to a provider-specific client built from ``config.llm``
                (Ollama by default, OpenAI when ``provider="openai"``).
        tools:  List of :class:`~kumaru.tools.base.BaseTool` instances the
                agent may call.  Empty by default.

    Example::

        from kumaru.agent import Agent
        from kumaru.tools import CalculatorTool

        agent = Agent(tools=[CalculatorTool()])
        reply = agent.run("What is 123 * 456?")
        print(reply)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm: Optional[BaseLLMClient] = None,
        tools: Optional[list[BaseTool]] = None,
    ) -> None:
        self._config = config or AgentConfig()
        self._llm = llm or self._build_default_llm(self._config)
        self._log = get_logger(__name__, verbose=self._config.verbose)

        # Build a lookup dict so we can dispatch tool calls by name quickly.
        self._tools: dict[str, BaseTool] = {}
        for tool in (tools or []):
            self._tools[tool.name] = tool

        # Memory is created fresh for each conversation.
        # Call ``reset()`` to start a new session without recreating the agent.
        self._memory = ConversationMemory(
            system_prompt=self._config.system_prompt
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> str:
        """Process *user_message* and return the agent's final reply.

        This is the main entry point.  It runs the full ReAct loop until:
          * The model produces a text response (not a tool call).
          * ``config.max_iterations`` is reached (safety limit).

        Args:
            user_message: The human's input.

        Returns:
            The agent's text response.

        Raises:
            RuntimeError: If the safety iteration limit is reached.
        """
        self._log.info("User message received", extra={"user_message": user_message})
        self._memory.add_user(user_message)

        tool_schemas = self._get_tool_schemas()

        for iteration in range(1, self._config.max_iterations + 1):
            self._log.debug(
                "Starting iteration", extra={"iteration": iteration}
            )

            # ── Step 1: REASON ──────────────────────────────────────────
            response = self._reason(tool_schemas)

            # ── Step 2: ACT (if the model wants to call a tool) ─────────
            if response.tool_calls:
                self._memory.add_assistant(
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
                self._act(response.tool_calls)
                # Continue the loop so the model can OBSERVE and reason again.
                continue

            # ── Final answer ─────────────────────────────────────────────
            final_answer = response.content or ""
            self._memory.add_assistant(final_answer)
            self._log.info(
                "Agent produced final answer",
                extra={"iterations": iteration},
            )
            return final_answer

        raise RuntimeError(
            f"Agent exceeded maximum iterations ({self._config.max_iterations}). "
            "This usually means the agent is stuck in a reasoning loop.  "
            "Increase AgentConfig.max_iterations or simplify the query."
        )

    def reset(self) -> None:
        """Clear the conversation history and start fresh."""
        self._memory.clear()
        self._log.debug("Conversation memory cleared")

    @property
    def memory(self) -> ConversationMemory:
        """Direct access to the conversation memory (useful for inspection)."""
        return self._memory

    @property
    def tools(self) -> dict[str, BaseTool]:
        """The registered tools, keyed by name."""
        return dict(self._tools)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_default_llm(self, config: AgentConfig) -> BaseLLMClient:
        """Create the default LLM client from configuration."""
        provider = config.llm.provider.lower()
        if provider == "ollama":
            return OllamaClient(config.llm)
        if provider == "openai":
            return OpenAIClient(config.llm)
        raise ValueError(
            f"Unsupported LLM provider '{config.llm.provider}'. "
            "Supported providers: ollama, openai."
        )

    def _reason(self, tool_schemas: list[dict[str, Any]]):
        """Send the current conversation to the LLM and return its response."""
        messages = self._memory.get_messages()
        return self._llm.chat(
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
        )

    def _act(self, tool_calls: list[dict[str, Any]]) -> None:
        """Execute each tool call and store the results in memory."""
        for tc in tool_calls:
            tool_call_id = tc["id"]
            function = tc["function"]
            tool_name = function["name"]
            raw_args = function.get("arguments", "{}")

            self._log.info(
                "Tool call",
                extra={"tool": tool_name, "tool_args": raw_args},
            )

            result = self._execute_tool(tool_name, raw_args)

            self._log.info(
                "Tool result",
                extra={"tool": tool_name, "result": result[:200]},
            )

            # Append the result so the model can OBSERVE it next iteration.
            self._memory.add_tool_result(
                tool_call_id=tool_call_id,
                name=tool_name,
                content=result,
            )

    def _execute_tool(self, tool_name: str, raw_args: str) -> str:
        """Look up and run a tool by name, returning a string result."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return (
                f"Error: Unknown tool '{tool_name}'. "
                f"Available tools: {list(self._tools.keys())}"
            )

        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            return f"Error: Could not parse tool arguments – {exc}"

        try:
            return tool.run(**args)
        except ToolError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001
            self._log.warning(
                "Unexpected tool error",
                extra={"tool": tool_name, "error": str(exc)},
            )
            return f"Error: Tool '{tool_name}' encountered an unexpected error – {exc}"

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return the OpenAI-formatted schemas for all registered tools."""
        return [t.to_openai_schema() for t in self._tools.values()]
