"""Tests for kumaru/memory/conversation.py"""

import pytest
from kumaru.llm.base import Message
from kumaru.memory.conversation import ConversationMemory


def test_initial_state_has_system_prompt():
    mem = ConversationMemory(system_prompt="You are a helpful assistant.")
    messages = mem.get_messages()
    assert len(messages) == 1
    assert messages[0].role == "system"
    assert messages[0].content == "You are a helpful assistant."


def test_add_user_message():
    mem = ConversationMemory(system_prompt="sys")
    mem.add_user("Hello!")
    messages = mem.get_messages()
    assert len(messages) == 2
    assert messages[1].role == "user"
    assert messages[1].content == "Hello!"


def test_add_assistant_message():
    mem = ConversationMemory(system_prompt="sys")
    mem.add_assistant("Hi there!")
    messages = mem.get_messages()
    assert messages[1].role == "assistant"
    assert messages[1].content == "Hi there!"


def test_add_tool_result():
    mem = ConversationMemory(system_prompt="sys")
    mem.add_tool_result(tool_call_id="call_1", name="calculator", content="42")
    msg = mem.get_messages()[1]
    assert msg.role == "tool"
    assert msg.content == "42"
    assert msg.tool_call_id == "call_1"
    assert msg.name == "calculator"


def test_max_messages_drops_oldest_non_system():
    mem = ConversationMemory(system_prompt="sys", max_messages=3)
    for i in range(5):
        mem.add_user(f"message {i}")

    messages = mem.get_messages()
    # System prompt is always kept; only 3 user messages are kept.
    assert len(messages) == 4  # system + 3
    assert messages[0].role == "system"
    # The oldest messages should have been evicted.
    assert messages[1].content == "message 2"
    assert messages[-1].content == "message 4"


def test_clear_resets_to_system_prompt_only():
    mem = ConversationMemory(system_prompt="sys")
    mem.add_user("hi")
    mem.add_assistant("hello")
    mem.clear()
    messages = mem.get_messages()
    assert len(messages) == 1
    assert messages[0].role == "system"


def test_len_includes_system_prompt():
    mem = ConversationMemory(system_prompt="sys")
    assert len(mem) == 1
    mem.add_user("test")
    assert len(mem) == 2


def test_get_messages_returns_copy():
    """Mutating the returned list must not affect internal state."""
    mem = ConversationMemory(system_prompt="sys")
    messages = mem.get_messages()
    messages.append(Message(role="user", content="injected"))
    assert len(mem) == 1  # unchanged
