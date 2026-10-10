"""Protect prompt normalization, memory trimming, and echo-backed agent calls."""

import pytest

from kumaru.agent.chat import ChatAgent
from kumaru.agent.memory import DropOldestPolicy, NoTrimPolicy, SlidingWindowPolicy, get_policy
from kumaru.agent.prompt import QUALITY_DIRECTIVES, build_system_prompt, ensure_system_message
from kumaru.backends.echo import EchoBackend
from kumaru.core.config import AgentConfig, BackendConfig, KumaruConfig, ThinkingModeConfig
from kumaru.core.errors import ContextLengthExceededError
from kumaru.core.types import GenerationParams, Message, Role


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


class TestThinkingParams:
    """Precedence: `generation` < `thinking_mode` (only while thinking) < the request."""

    @staticmethod
    def make_agent(thinking: bool | None = None) -> ChatAgent:
        cfg = KumaruConfig(
            backend=BackendConfig(name="echo", model="echo", extra={"delay_s": 0}),
            generation=GenerationParams(max_tokens=100, temperature=0.7, thinking=thinking),
            thinking_mode=ThinkingModeConfig(max_tokens=4000, temperature=1.0, thinking_budget=1000),
        )
        return ChatAgent(EchoBackend(cfg.backend), cfg)

    def test_without_thinking_the_preset_is_ignored(self) -> None:
        params = self.make_agent().resolve_params()
        assert (params.max_tokens, params.temperature, params.thinking_budget) == (100, 0.7, None)

    def test_a_thinking_request_applies_the_preset(self) -> None:
        params = self.make_agent().resolve_params(thinking=True)
        assert (params.thinking, params.max_tokens, params.temperature) == (True, 4000, 1.0)
        assert params.thinking_budget == 1000

    def test_thinking_on_by_default_applies_the_preset(self) -> None:
        params = self.make_agent(thinking=True).resolve_params()
        assert (params.max_tokens, params.temperature) == (4000, 1.0)

    def test_a_request_can_turn_default_thinking_off(self) -> None:
        params = self.make_agent(thinking=True).resolve_params(thinking=False)
        assert (params.thinking, params.max_tokens, params.temperature) == (False, 100, 0.7)

    def test_request_values_beat_the_preset_and_none_means_unset(self) -> None:
        params = self.make_agent().resolve_params(thinking=True, temperature=0.2, max_tokens=None)
        assert (params.temperature, params.max_tokens) == (0.2, 4000)

    @pytest.mark.asyncio
    async def test_history_budget_reserves_the_resolved_answer_length(self, monkeypatch) -> None:
        agent = self.make_agent()
        reserved: list[int | None] = []
        prepare = agent.prepare

        def spy(messages, *, extra_context=None, reserve_tokens=None):
            reserved.append(reserve_tokens)
            return prepare(messages, extra_context=extra_context, reserve_tokens=reserve_tokens)

        monkeypatch.setattr(agent, "prepare", spy)
        await agent.backend.load()
        try:
            result = await agent.complete([Message.user("hi")], thinking=True)
            _ = [chunk async for chunk in agent.stream([Message.user("hi")])]
        finally:
            await agent.backend.aclose()
        # A thinking answer can be 40x longer, so history must shrink to make room.
        assert reserved == [4000, 100]
        assert "max_tokens=4000, temperature=1.0" in result.message.content  # echo reports params

    def test_a_huge_answer_cap_still_leaves_half_the_window_for_history(self, monkeypatch) -> None:
        # qwen36 at --context 32768: reserving its 32k cap in full would leave 512 tokens.
        cfg = KumaruConfig(
            backend=BackendConfig(name="echo", model="echo", max_context_length=32768, extra={"delay_s": 0}),
            agent=AgentConfig(max_history_tokens=65536),
        )
        agent = ChatAgent(EchoBackend(cfg.backend), cfg)
        budgets: list[int] = []
        monkeypatch.setattr(
            agent.memory, "apply", lambda messages, budget, count: budgets.append(budget) or list(messages)
        )
        agent.prepare([Message.user("hi")], reserve_tokens=32768)
        agent.prepare([Message.user("hi")], reserve_tokens=1000)
        assert budgets == [32768 // 2 - 128, 32768 - 1000 - 128]

    @pytest.mark.asyncio
    async def test_the_reply_is_capped_to_the_room_the_prompt_leaves(self) -> None:
        cfg = KumaruConfig(
            backend=BackendConfig(name="echo", model="echo", max_context_length=2048, extra={"delay_s": 0}),
            generation=GenerationParams(max_tokens=2048),
        )
        agent = ChatAgent(EchoBackend(cfg.backend), cfg)
        await agent.backend.load()
        try:
            result = await agent.complete([Message.user("word " * 60)])
        finally:
            await agent.backend.aclose()
        # Strict servers reject prompt + max_tokens > window, so the cap shrinks to fit.
        assert f"max_tokens={2048 - result.usage.prompt_tokens}," in result.message.content
