"""Agent layer: prompt assembly, bounded memory, and the chat loop."""

from __future__ import annotations

from datetime import datetime

from kumaru.agent.chat import ChatAgent
from kumaru.agent.memory import ConversationMemory
from kumaru.agent.prompt import build_messages, render_system_prompt
from kumaru.core.types import Message, Role


def test_system_prompt_placeholders() -> None:
    rendered = render_system_prompt("today is {date}", now=datetime(2030, 1, 2, 3, 4))
    assert rendered == "today is 2030-01-02"


def test_unknown_placeholders_are_left_alone() -> None:
    assert render_system_prompt("{ok} stays") == "{ok} stays"


def test_build_messages_orders_system_history_user() -> None:
    history = [Message.user("earlier"), Message.assistant("reply")]
    messages = build_messages("be helpful", history, "now")
    assert [m.role for m in messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
    ]
    assert messages[-1].content == "now"


def test_build_messages_drops_history_system_messages() -> None:
    history = [Message.system("injected"), Message.user("q")]
    messages = build_messages("real system", history, "next")
    assert sum(1 for m in messages if m.role is Role.SYSTEM) == 1
    assert messages[0].content == "real system"


def test_memory_keeps_sessions_separate() -> None:
    memory = ConversationMemory()
    memory.add("a", Message.user("one"))
    memory.add("b", Message.user("two"))
    assert [m.content for m in memory.history("a")] == ["one"]
    assert [m.content for m in memory.history("b")] == ["two"]


def test_memory_enforces_turn_budget() -> None:
    memory = ConversationMemory(max_turns=2)
    for i in range(5):
        memory.add("s", Message.user(str(i)))
    assert [m.content for m in memory.history("s")] == ["3", "4"]


def test_memory_enforces_character_budget() -> None:
    memory = ConversationMemory(max_turns=10, max_chars=20)
    memory.add("s", Message.user("x" * 15))
    memory.add("s", Message.user("y" * 15))
    history = memory.history("s")
    assert len(history) == 1 and history[0].content.startswith("y")


def test_memory_ignores_system_messages() -> None:
    memory = ConversationMemory()
    memory.add("s", Message.system("nope"))
    assert memory.history("s") == []


def test_memory_clear() -> None:
    memory = ConversationMemory()
    memory.add("s", Message.user("hi"))
    memory.clear("s")
    assert memory.history("s") == []


def test_agent_records_both_sides_of_a_turn(agent: ChatAgent) -> None:
    reply = agent.ask("hello", session_id="s")
    history = agent.history("s")
    assert [m.role for m in history] == [Role.USER, Role.ASSISTANT]
    assert history[1].content == reply


def test_agent_stream_matches_ask(agent: ChatAgent) -> None:
    streamed = "".join(c.delta for c in agent.stream("question", session_id="s"))
    assert "question" in streamed
    assert agent.history("s")[1].content == streamed


def test_agent_sends_prior_turns_to_the_backend(agent: ChatAgent) -> None:
    agent.ask("first", session_id="s")
    request = agent._request("s", "second")
    contents = [m.content for m in request.messages]
    assert contents[0].startswith("You are Kumaru")
    assert "first" in contents[1]
    assert contents[-1] == "second"


def test_agent_reset_clears_history(agent: ChatAgent) -> None:
    agent.ask("hi", session_id="s")
    agent.reset("s")
    assert agent.history("s") == []
