"""Protect prompt normalization, memory trimming, and echo-backed agent calls."""

import pytest

from kumaru.agent.chat import ChatAgent
from kumaru.agent.memory import DropOldestPolicy, NoTrimPolicy, SlidingWindowPolicy, get_policy
from kumaru.agent.prompt import QUALITY_DIRECTIVES, build_system_prompt, ensure_system_message
from kumaru.core.config import AgentConfig, BackendConfig, KumaruConfig
from kumaru.core.errors import ContextLengthExceededError
from kumaru.core.types import Message, Role


def count_words(messages: list[Message]) -> int:
    return sum(len(m.content.split()) + 1 for m in messages)


def test_ensure_system_message_inserts_exactly_one_system_at_index_zero() -> None:
    out = ensure_system_message([Message.user("hello")], "configured")
    assert [m.role for m in out].count(Role.SYSTEM) == 1
    assert out[0].role == Role.SYSTEM
    assert out[0].content == "configured"


def test_ensure_system_message_merges_client_system_instead_of_replacing() -> None:
    out = ensure_system_message(
        [Message.user("u"), Message.system("client"), Message.assistant("a")], "configured"
    )
    assert [m.role for m in out].count(Role.SYSTEM) == 1
    assert out[0].content == "configured\n\nclient"
    assert [m.role for m in out[1:]] == [Role.USER, Role.ASSISTANT]


def test_build_system_prompt_includes_quality_directives_and_tools() -> None:
    prompt = build_system_prompt(
        AgentConfig(system_prompt="Base"),
        tool_descriptions=["clock: tells time"],
        include_date=False,
    )
    assert "Base" in prompt
    assert QUALITY_DIRECTIVES in prompt
    assert "clock: tells time" in prompt


class TestMemoryPolicies:
    def test_no_trim_policy_raises_when_over_budget(self) -> None:
        with pytest.raises(ContextLengthExceededError):
            NoTrimPolicy().apply([Message.user("one two three")], 1, count_words)

    def test_drop_oldest_fits_budget_and_keeps_system_message(self) -> None:
        messages = [Message.system("sys"), *[Message.user(f"turn {i}") for i in range(6)]]
        out = DropOldestPolicy().apply(messages, 7, count_words)
        assert out[0].role == Role.SYSTEM
        assert count_words(out) <= 7
        assert out[-1].content == "turn 5"

    def test_sliding_window_keeps_system_message(self) -> None:
        messages = [Message.system("sys"), *[Message.user(f"turn {i}") for i in range(6)]]
        out = SlidingWindowPolicy(min_recent_turns=2).apply(messages, 8, count_words)
        assert out[0].role == Role.SYSTEM
        assert count_words(out) <= 8
        assert [m.content for m in out[-2:]] == ["turn 4", "turn 5"]

    def test_get_policy_unknown_name_falls_back_to_sliding_window(self) -> None:
        assert isinstance(get_policy("unknown"), SlidingWindowPolicy)


class TestChatAgent:
    def test_prepare_is_pure_and_respects_budget(self, loaded_echo_backend) -> None:
        cfg = KumaruConfig(
            backend=BackendConfig(name="echo", model="echo", max_context_length=256, extra={"delay_s": 0}),
            agent=AgentConfig(max_history_tokens=260, memory_policy="drop_oldest"),
        )
        agent = ChatAgent(loaded_echo_backend, cfg)
        messages = [Message.user("x" * 100) for _ in range(8)]
        original = list(messages)
        prepared = agent.prepare(messages)
        assert messages == original
        assert loaded_echo_backend.count_tokens(prepared) <= 260
        assert prepared[0].role == Role.SYSTEM

    @pytest.mark.asyncio
    async def test_complete_works_against_echo(self, chat_agent: ChatAgent) -> None:
        result = await chat_agent.complete([Message.user("hello")], max_tokens=8)
        assert result.message.role == Role.ASSISTANT
        assert "You said: hello" in result.message.content

    @pytest.mark.asyncio
    async def test_stream_works_against_echo(self, chat_agent: ChatAgent) -> None:
        chunks = [chunk async for chunk in chat_agent.stream([Message.user("hello")], max_tokens=8)]
        assert "You said: hello" in "".join(chunk.delta for chunk in chunks)
        assert chunks[-1].finish_reason == "stop"
