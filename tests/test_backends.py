"""Protect the offline echo backend and streaming stop-sequence helpers."""

import pytest

from kumaru.backends.base import Backend, StopSequenceTracker
from kumaru.backends.echo import EchoBackend
from kumaru.core.config import BackendConfig
from kumaru.core.types import GenerationParams, Message


@pytest.mark.asyncio
async def test_echo_backend_load_info_stream_and_generate(loaded_echo_backend: EchoBackend) -> None:
    assert loaded_echo_backend.is_loaded
    info = loaded_echo_backend.info()
    assert info.name == "echo"
    assert info.device == "cpu"

    messages = [Message.user("hello")]
    params = GenerationParams(max_tokens=8, temperature=0.0)
    chunks = [chunk async for chunk in loaded_echo_backend.stream(messages, params)]
    stream_text = "".join(chunk.delta for chunk in chunks)
    result = await loaded_echo_backend.generate(messages, params)
    assert result.message.content == stream_text
    assert chunks[-1].finish_reason == "stop"
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens == len([chunk for chunk in chunks if chunk.delta])


@pytest.mark.asyncio
async def test_echo_reply_is_deterministic_for_same_input() -> None:
    backend = EchoBackend(BackendConfig(name="echo", model="echo", extra={"delay_s": 0}))
    await backend.load()
    messages = [Message.user("same")]
    params = GenerationParams(max_tokens=8, temperature=0.0)
    first = await backend.generate(messages, params)
    second = await backend.generate(messages, params)
    assert first.message.content == second.message.content


def test_apply_stop_sequences_truncates_at_earliest_stop() -> None:
    assert Backend.apply_stop_sequences("abcSTOPdefEND", ["END", "STOP"]) == ("abc", True)
    assert Backend.apply_stop_sequences("abcdef", ["STOP"]) == ("abcdef", False)
    assert Backend.apply_stop_sequences("abcdef", [""]) == ("abcdef", False)


class TestStopSequenceTracker:
    def test_no_stop_sequences_passes_through_and_flushes_empty(self) -> None:
        tracker = StopSequenceTracker([])
        assert tracker.push("abc") == "abc"
        assert tracker.flush() == ""

    def test_stop_in_single_chunk_suppresses_stop_and_tail(self) -> None:
        tracker = StopSequenceTracker(["STOP"])
        assert tracker.push("abcSTOPdef") == "abc"
        assert tracker.stopped is True
        assert tracker.flush() == ""

    def test_stop_sequence_split_across_chunks_is_detected(self) -> None:
        tracker = StopSequenceTracker(["</s>"])
        assert tracker.push("</") == ""
        assert tracker.push("s>") == ""
        assert tracker.stopped is True
        assert tracker.flush() == ""

    def test_partial_suffix_is_held_then_released_when_safe(self) -> None:
        tracker = StopSequenceTracker(["XYZ"])
        assert tracker.push("abX") == "a"
        assert tracker.push("cd") == "bX"
        assert tracker.flush() == "cd"

    def test_after_stopped_push_returns_delta_without_buffering(self) -> None:
        tracker = StopSequenceTracker(["!"])
        assert tracker.push("hi!") == "hi"
        assert tracker.push("ignored") == "ignored"
