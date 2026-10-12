"""Protect stable core data model helpers and derived metrics."""

from kumaru.core.types import ChatResult, GenerationParams, Message, Role, Usage


def test_message_to_openai_shape_includes_optional_fields() -> None:
    message = Message(
        role=Role.TOOL,
        content="result",
        name="lookup",
        tool_call_id="call-1",
        tool_calls=[{"id": "call-1"}],
    )
    assert message.to_openai() == {
        "role": "tool",
        "content": "result",
        "name": "lookup",
        "tool_call_id": "call-1",
        "tool_calls": [{"id": "call-1"}],
    }


def test_message_none_content_is_coerced_to_empty_string() -> None:
    assert Message(role=Role.ASSISTANT, content=None).content == ""  # type: ignore[arg-type]


def test_generation_params_merged_ignores_none_and_does_not_mutate_original() -> None:
    original = GenerationParams(max_tokens=10, temperature=0.5, stop=["x"])
    merged = original.merged(max_tokens=None, temperature=0.0, stop=None)
    assert merged.max_tokens == 10
    assert merged.temperature == 0.0
    assert merged.stop == ["x"]
    assert original.temperature == 0.5


def test_usage_total_tokens_sums_prompt_and_completion() -> None:
    assert Usage(prompt_tokens=3, completion_tokens=4).total_tokens == 7


def test_chat_result_tokens_per_second_and_zero_latency_guard() -> None:
    result = ChatResult(message=Message.assistant("x"), usage=Usage(completion_tokens=10), latency_ms=250)
    assert result.tokens_per_second == 40.0
    zero = ChatResult(message=Message.assistant("x"), usage=Usage(completion_tokens=10), latency_ms=0)
    assert zero.tokens_per_second == 0.0
