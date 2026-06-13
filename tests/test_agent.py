"""Tests for kumaru/agent.py

We mock the LLM client so these tests run without an API key and don't make
any real network calls.  This pattern is essential for fast, reliable CI.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from kumaru.agent import Agent
from kumaru.config import AgentConfig
from kumaru.llm.base import BaseLLMClient, LLMResponse, Message
from kumaru.tools.base import BaseTool
from kumaru.tools.calculator import CalculatorTool


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class EchoLLM(BaseLLMClient):
    """A deterministic mock LLM that returns a preset sequence of responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = iter(responses)

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        return next(self._responses)


def _text_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], usage={})


def _tool_call_response(tool_name: str, args: str, call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": args},
            }
        ],
        usage={},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Basic agent behaviour
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentBasics:
    def _make_agent(self, responses, tools=None):
        llm = EchoLLM(responses)
        config = AgentConfig(max_iterations=5)
        return Agent(config=config, llm=llm, tools=tools or [])

    def test_simple_text_response(self):
        agent = self._make_agent([_text_response("Hello, world!")])
        result = agent.run("Say hello")
        assert result == "Hello, world!"

    def test_memory_grows_after_run(self):
        agent = self._make_agent([_text_response("Hi")])
        agent.run("Hello")
        # system + user + assistant = 3
        assert len(agent.memory) == 3

    def test_reset_clears_conversation(self):
        agent = self._make_agent([_text_response("Hi"), _text_response("Hi again")])
        agent.run("Hello")
        agent.reset()
        assert len(agent.memory) == 1  # only system prompt remains

    def test_max_iterations_raises(self):
        # The LLM always requests a tool → agent never gets a text answer.
        responses = [
            _tool_call_response("calculator", '{"expression": "1+1"}')
        ] * 10
        agent = self._make_agent(responses, tools=[CalculatorTool()])
        with pytest.raises(RuntimeError, match="exceeded maximum iterations"):
            agent.run("Loop forever")

    def test_tools_property(self):
        agent = self._make_agent([], tools=[CalculatorTool()])
        assert "calculator" in agent.tools


# ──────────────────────────────────────────────────────────────────────────────
# Tool calling
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentToolCalling:
    def _make_agent(self, responses, tools=None):
        llm = EchoLLM(responses)
        config = AgentConfig(max_iterations=5)
        return Agent(config=config, llm=llm, tools=tools or [])

    def test_calculator_tool_called_and_result_returned(self):
        responses = [
            _tool_call_response("calculator", '{"expression": "6 * 7"}'),
            _text_response("6 * 7 = 42"),
        ]
        agent = self._make_agent(responses, tools=[CalculatorTool()])
        result = agent.run("What is 6 * 7?")
        assert result == "6 * 7 = 42"

    def test_tool_result_added_to_memory(self):
        responses = [
            _tool_call_response("calculator", '{"expression": "2+2"}'),
            _text_response("It is 4"),
        ]
        agent = self._make_agent(responses, tools=[CalculatorTool()])
        agent.run("What is 2+2?")
        # system + user + assistant(tool_call) + tool_result + assistant = 5
        assert len(agent.memory) == 5

    def test_unknown_tool_returns_error_to_model(self):
        """The agent should not crash when the model calls a non-existent tool."""
        responses = [
            _tool_call_response("nonexistent_tool", "{}"),
            _text_response("I could not use that tool."),
        ]
        agent = self._make_agent(responses)
        result = agent.run("Use nonexistent tool")
        assert result == "I could not use that tool."

    def test_malformed_args_returns_error_to_model(self):
        """The agent should not crash on malformed JSON tool arguments."""
        responses = [
            _tool_call_response("calculator", "NOT JSON"),
            _text_response("Sorry, I made a mistake."),
        ]
        agent = self._make_agent(responses, tools=[CalculatorTool()])
        result = agent.run("Calculate something")
        assert result == "Sorry, I made a mistake."

    def test_tool_error_propagates_as_message(self):
        """ToolError should be caught and sent back to the model."""
        responses = [
            _tool_call_response("calculator", '{"expression": "1/0"}'),
            _text_response("You cannot divide by zero."),
        ]
        agent = self._make_agent(responses, tools=[CalculatorTool()])
        result = agent.run("Divide by zero")
        assert result == "You cannot divide by zero."


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentConfig:
    def test_default_config_is_used_when_none_provided(self):
        """Agent should be constructable with only a mock LLM (no config needed)."""
        llm = EchoLLM([_text_response("ok")])
        agent = Agent(llm=llm)
        assert agent.run("hi") == "ok"

    def test_custom_system_prompt_is_in_memory(self):
        llm = EchoLLM([_text_response("ok")])
        config = AgentConfig(system_prompt="You are a pirate.")
        agent = Agent(config=config, llm=llm)
        assert agent.memory.get_messages()[0].content == "You are a pirate."

    def test_default_provider_is_ollama(self):
        config = AgentConfig()
        assert config.llm.provider == "ollama"
        assert config.llm.model == "llama3.1"

    def test_default_agent_uses_ollama_client(self, monkeypatch):
        class DummyLLM(BaseLLMClient):
            def __init__(self, config):
                self.config = config

            def chat(self, messages, tools=None):
                return _text_response("ok")

        monkeypatch.setattr("kumaru.agent.OllamaClient", DummyLLM)
        agent = Agent(config=AgentConfig())
        assert isinstance(agent._llm, DummyLLM)

    def test_openai_provider_uses_openai_client(self, monkeypatch):
        class DummyLLM(BaseLLMClient):
            def __init__(self, config):
                self.config = config

            def chat(self, messages, tools=None):
                return _text_response("ok")

        monkeypatch.setattr("kumaru.agent.OpenAIClient", DummyLLM)
        config = AgentConfig()
        config.llm.provider = "openai"
        agent = Agent(config=config)
        assert isinstance(agent._llm, DummyLLM)
