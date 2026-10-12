"""Protect the offline echo backend and the streaming helpers every backend shares."""

import asyncio
import contextlib
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from kumaru.backends.base import Backend, ReasoningSplitter, StopSequenceTracker, iterate_in_thread
from kumaru.backends.echo import EchoBackend
from kumaru.backends.llamacpp_backend import LlamaCppBackend
from kumaru.backends.openai_compat import OpenAICompatBackend, normalize_finish_reason
from kumaru.backends.transformers_backend import TransformersBackend
from kumaru.core.config import BackendConfig
from kumaru.core.errors import BackendError, ConfigError, DependencyMissingError, install_command
from kumaru.core.registry import BACKENDS
from kumaru.core.types import GenerationParams, Message

ECHO_THOUGHT = (
    "The user asked: hello. Echo cannot reason; this text only exercises the reasoning channel."
)


@pytest.mark.asyncio
async def test_echo_backend_load_info_stream_and_generate(loaded_echo_backend: EchoBackend) -> None:
    assert loaded_echo_backend.is_loaded
    info = loaded_echo_backend.info()
    assert info.name == "echo"
    assert info.device == "cpu"

    messages = [Message.user("hello")]
    params = GenerationParams(max_tokens=64, temperature=0.0)  # room for the whole reply
    chunks = [chunk async for chunk in loaded_echo_backend.stream(messages, params)]
    stream_text = "".join(chunk.delta for chunk in chunks)
    result = await loaded_echo_backend.generate(messages, params)
    assert result.message.content == stream_text
    assert chunks[-1].finish_reason == result.finish_reason == "stop"
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


# -- echo: thinking and loading -----------------------------------------------------


@pytest.mark.asyncio
async def test_echo_thinking_uses_only_the_reasoning_channel(loaded_echo_backend: EchoBackend) -> None:
    messages = [Message.user("hello")]
    plain = await loaded_echo_backend.generate(messages, GenerationParams(max_tokens=64))
    thought = await loaded_echo_backend.generate(
        messages, GenerationParams(max_tokens=64, thinking=True)
    )
    assert plain.message.reasoning is None
    assert thought.message.reasoning == ECHO_THOUGHT
    assert thought.message.content == plain.message.content  # the answer is untouched
    assert thought.usage.completion_tokens > plain.usage.completion_tokens  # thoughts cost tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(("budget", "expected"), [(0, None), (2, "The user asked: ")])
async def test_echo_thinking_budget_caps_the_thought(
    loaded_echo_backend: EchoBackend, budget: int, expected: str | None
) -> None:
    params = GenerationParams(max_tokens=8, thinking=True, thinking_budget=budget)
    result = await loaded_echo_backend.generate([Message.user("hello")], params)
    assert result.message.reasoning == expected  # echo thinks in 8-character pieces


# -- echo: the max_tokens cap ---------------------------------------------------------


@pytest.mark.asyncio
async def test_echo_max_tokens_cuts_the_answer_short_and_reports_length(
    loaded_echo_backend: EchoBackend,
) -> None:
    messages = [Message.user("hello")]
    full = (await loaded_echo_backend.generate(messages, GenerationParams(max_tokens=64))).message
    params = GenerationParams(max_tokens=3)
    chunks = [chunk async for chunk in loaded_echo_backend.stream(messages, params)]
    result = await loaded_echo_backend.generate(messages, params)
    # One 6-character piece per token, then a final chunk that says why it stopped.
    assert [chunk.delta for chunk in chunks[:-1]] == [full.content[i : i + 6] for i in (0, 6, 12)]
    assert (chunks[-1].finish_reason, chunks[-1].usage.completion_tokens) == ("length", 3)
    assert (result.message.content, result.finish_reason) == (full.content[:18], "length")
    assert result.usage.completion_tokens == 3


@pytest.mark.asyncio
async def test_echo_reports_stop_when_the_reply_fits_the_cap_exactly(
    loaded_echo_backend: EchoBackend,
) -> None:
    messages = [Message.user("hello")]
    # The reply quotes its cap, so two-digit caps all give a reply of the same length.
    roomy = await loaded_echo_backend.generate(messages, GenerationParams(max_tokens=99))
    pieces = roomy.usage.completion_tokens
    exact = await loaded_echo_backend.generate(messages, GenerationParams(max_tokens=pieces))
    short = await loaded_echo_backend.generate(messages, GenerationParams(max_tokens=pieces - 1))
    assert (roomy.finish_reason, exact.finish_reason) == ("stop", "stop")
    assert exact.usage.completion_tokens == pieces
    assert (short.finish_reason, short.usage.completion_tokens) == ("length", pieces - 1)
    assert len(short.message.content) == 6 * (pieces - 1)  # whole 6-character pieces


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_tokens", "reasoning", "answer_pieces"),
    [
        (5, ECHO_THOUGHT[:40], 0),  # the cap runs out mid-thought: no answer at all
        (14, ECHO_THOUGHT, 2),  # all 12 thought pieces, then 2 answer pieces
    ],
)
async def test_echo_max_tokens_counts_thinking_tokens_too(
    loaded_echo_backend: EchoBackend, max_tokens: int, reasoning: str, answer_pieces: int
) -> None:
    messages = [Message.user("hello")]
    full = (await loaded_echo_backend.generate(messages, GenerationParams(max_tokens=64))).message
    params = GenerationParams(max_tokens=max_tokens, thinking=True)
    chunks = [chunk async for chunk in loaded_echo_backend.stream(messages, params)]
    result = await loaded_echo_backend.generate(messages, params)
    assert len([chunk for chunk in chunks if chunk.delta]) == answer_pieces
    assert result.message.reasoning == reasoning
    assert result.message.content == full.content[: 6 * answer_pieces]
    assert (chunks[-1].finish_reason, result.finish_reason) == ("length", "length")
    assert result.usage.completion_tokens == max_tokens


@pytest.mark.asyncio
async def test_concurrent_loads_load_the_model_once() -> None:
    class SlowEcho(EchoBackend):
        def __init__(self, config: BackendConfig) -> None:
            super().__init__(config)
            self.loads = 0

        async def _load(self) -> None:
            self.loads += 1
            await asyncio.sleep(0.02)  # slow, like starting a real model server

    backend = SlowEcho(BackendConfig(name="echo", model="echo", extra={"delay_s": 0}))
    await asyncio.gather(*(backend.load() for _ in range(4)))
    assert (backend.loads, backend.is_loaded) == (1, True)
    await backend.aclose()
    assert not backend.is_loaded
    await backend.load()
    assert backend.loads == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("name", BACKENDS.names())
async def test_only_backends_that_attach_adapters_accept_one(name: str) -> None:
    backend = BACKENDS.create(name, BackendConfig(name=name, model="m", adapter_path="runs/lora/adapter"))
    loads: list[str] = []

    async def fake_load() -> None:  # stands in for the real, heavy load
        loads.append(name)

    backend._load = fake_load  # type: ignore[method-assign]
    if backend.supports_adapters:
        await backend.load()
        assert loads == [name]
    else:
        # A backend that ignored the adapter would serve the base model under the adapter's name.
        with pytest.raises(ConfigError, match="cannot attach a LoRA adapter"):
            await backend.load()
        assert (loads, backend.is_loaded) == ([], False)


def test_transformers_is_the_backend_that_attaches_adapters() -> None:
    assert [name for name in BACKENDS.names() if BACKENDS.get(name).supports_adapters] == ["transformers"]


@pytest.mark.parametrize(
    ("extra", "windows", "expected"),
    [
        ("torch", True, r".\scripts\setup.ps1 -Gpu"),
        ("quant", True, r".\scripts\setup.ps1 -Quant"),
        ("gguf", True, r".\scripts\setup.ps1 -Gguf"),
        ("bench", True, 'pip install -e ".[bench]" in the Kumaru folder'),
        ("torch", False, 'pip install -e ".[torch]" in the Kumaru folder'),
    ],
)
def test_install_hints_name_the_source_folder_not_pypi(extra: str, windows: bool, expected: str) -> None:
    command = install_command(extra, windows=windows)
    assert command.startswith(expected)
    assert "kumaru[" not in command  # PyPI has no Kumaru project; that name could be anyone's


def test_a_missing_engine_says_how_to_install_it() -> None:
    error = DependencyMissingError("llama-cpp-python", extra="gguf")
    assert error.hint == f"Install it with: {install_command('gguf')}"
    assert DependencyMissingError("rich").hint == "Install it with: pip install rich"


# -- Worker-thread backends (transformers, llamacpp) ------------------------------------


@pytest.mark.asyncio
async def test_iterate_in_thread_delivers_every_item_then_raises_the_producers_error() -> None:
    def produce():
        yield from range(600)  # more than the queue holds
        raise ValueError("boom")

    received = []
    with pytest.raises(ValueError, match="boom"):
        async for item in iterate_in_thread(produce, queue_size=8):
            received.append(item)
    assert received == list(range(600))


@pytest.mark.asyncio
async def test_an_abandoned_stream_closes_its_producer_on_the_worker_thread() -> None:
    closed = threading.Event()
    closed_on: list[str] = []
    produced = 0

    def produce():
        nonlocal produced
        try:
            for item in range(10_000):
                produced += 1
                yield item
                time.sleep(0.001)
        finally:  # where a backend stops model.generate or releases its lock
            closed_on.append(threading.current_thread().name)
            closed.set()

    stream = iterate_in_thread(produce, queue_size=4)
    assert [await anext(stream) for _ in range(3)] == [0, 1, 2]
    await stream.aclose()  # what the server does when the client presses Stop
    assert await asyncio.to_thread(closed.wait, 5)
    assert closed_on == ["kumaru-generate"]
    assert produced < 100


def test_a_cancelled_consumer_does_not_strand_an_executor_thread() -> None:
    async def scenario() -> int:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))

        def produce():
            yield 0
            time.sleep(0.3)  # the consumer gives up while waiting for this one
            yield 1

        stream = iterate_in_thread(produce)
        assert await anext(stream) == 0
        waiting = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0.05)  # now the only executor thread is blocked in queue.get
        waiting.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiting
        return await asyncio.wait_for(loop.run_in_executor(None, lambda: 42), 5)

    # Its own loop on its own thread: a stranded executor thread would hang
    # asyncio.run's shutdown, which must fail this test, not freeze the suite.
    result: list[int] = []
    runner = threading.Thread(target=lambda: result.append(asyncio.run(scenario())), daemon=True)
    runner.start()
    runner.join(15)
    assert result == [42]


class FakeLlama:
    """Stands in for llama_cpp.Llama, which is not thread-safe: counts overlapping generations."""

    def __init__(self) -> None:
        self.active = self.most_active = self.closed = 0
        self.guard = threading.Lock()

    def create_chat_completion(self, **kwargs: object):
        def events():
            with self.guard:
                self.active += 1
                self.most_active = max(self.most_active, self.active)
            try:
                for word in ("one ", "two ", "three"):
                    time.sleep(0.02)
                    yield {"choices": [{"delta": {"content": word}, "finish_reason": None}]}
                yield {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5},
                }
            finally:
                with self.guard:
                    self.active -= 1
                    self.closed += 1

        return events()


@pytest.mark.asyncio
async def test_llamacpp_generations_take_turns_and_an_abandoned_one_frees_the_model() -> None:
    backend = LlamaCppBackend(BackendConfig(name="llamacpp", model="fake.gguf"))
    backend.llm = fake = FakeLlama()
    backend._loaded = True
    params = GenerationParams(max_tokens=16, temperature=0.0)

    async def ask() -> str:
        return "".join(
            [chunk.delta async for chunk in backend.stream([Message.user("hi")], params)]
        )

    # The server admits several requests at once (server.max_concurrent_requests).
    assert await asyncio.gather(ask(), ask(), ask()) == ["one two three"] * 3
    assert fake.most_active == 1

    stream = backend.stream([Message.user("hi")], params)
    await anext(stream)
    await stream.aclose()  # abandoned mid-answer
    assert await asyncio.to_thread(backend._generate_lock.acquire, True, 5)
    backend._generate_lock.release()
    assert fake.closed == 4
    assert await ask() == "one two three"


@pytest.mark.asyncio
async def test_llamacpp_loads_off_the_event_loop_and_expands_the_home_folder(
    scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    class RecordingLlama:
        def __init__(self, model_path: str, **kwargs: object) -> None:
            seen["path"], seen["thread"] = model_path, threading.current_thread()

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = RecordingLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    monkeypatch.setenv("USERPROFILE", str(scratch_dir))  # what ~ means on Windows
    monkeypatch.setenv("HOME", str(scratch_dir))  # ...and everywhere else
    (scratch_dir / "m.gguf").write_bytes(b"GGUF")

    backend = LlamaCppBackend(BackendConfig(name="llamacpp", model="m", model_path="~/m.gguf"))
    await backend.load()

    assert Path(str(seen["path"])) == scratch_dir / "m.gguf"
    # A multi-gigabyte load on the event loop would freeze every other request.
    assert seen["thread"] is not threading.current_thread()


class FakeTokenizer:
    """Just enough tokenizer for TransformersBackend: every token decodes to "w "."""

    pad_token_id, eos_token_id, chat_template = 0, None, None

    def __call__(self, text: str, **kwargs: object) -> dict:
        import torch

        return {
            "input_ids": torch.tensor([[5, 6, 7]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }

    def decode(self, ids: list[int], **kwargs: object) -> str:
        return "w " * len(ids)


def tiny_gpt2(steps: list[int], finished: threading.Event) -> object:
    """A random two-layer GPT-2 (no download) that counts its decoding steps.

    It has no end-of-sequence token, so only max_new_tokens or Kumaru's halt
    can end a generation: exactly what the test below needs to tell apart.
    """
    import transformers

    config = transformers.GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=100,
        n_positions=4096,
        bos_token_id=None,
        eos_token_id=None,
    )
    model = transformers.GPT2LMHeadModel(config).eval()
    forward, generate = model.forward, model.generate

    def counting_forward(*args: object, **kwargs: object) -> object:
        steps.append(1)
        time.sleep(0.002)
        return forward(*args, **kwargs)

    def watched_generate(*args: object, **kwargs: object) -> object:
        try:
            return generate(*args, **kwargs)
        finally:
            finished.set()

    model.forward, model.generate = counting_forward, watched_generate
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["abandoned", "stop sequence"])
async def test_transformers_stops_generating_once_nobody_wants_the_rest(how: str) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    steps: list[int] = []
    finished = threading.Event()
    backend = TransformersBackend(BackendConfig(name="transformers", model="tiny"))
    backend.tokenizer, backend.model = FakeTokenizer(), tiny_gpt2(steps, finished)
    backend._loaded = True
    params = GenerationParams(
        max_tokens=2000, temperature=0.0, stop=["w w w"] if how == "stop sequence" else []
    )

    stream = backend.stream([Message.user("hi")], params)
    if how == "abandoned":
        await anext(stream)
        await stream.aclose()
    else:
        assert [chunk async for chunk in stream][-1].finish_reason == "stop"
    assert await asyncio.to_thread(finished.wait, 5)
    assert len(steps) < 200  # not the 2000 it would run to unattended


class WordTokenizer(FakeTokenizer):
    """Decodes without spaces, so the streamer holds the whole reply as one unfinished word."""

    def decode(self, ids: list[int], **kwargs: object) -> str:
        return "w" * len(ids)


@pytest.mark.asyncio
async def test_transformers_counts_tokens_not_pieces_of_text() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    backend = TransformersBackend(BackendConfig(name="transformers", model="tiny"))
    backend.tokenizer, backend.model = WordTokenizer(), tiny_gpt2([], threading.Event())
    backend._loaded = True

    params = GenerationParams(max_tokens=20, temperature=0.0)
    chunks = [chunk async for chunk in backend.stream([Message.user("hi")], params)]

    # One piece of text for 20 tokens: counting pieces (the old way) would report 1 and "stop".
    assert [chunk.delta for chunk in chunks if chunk.delta] == ["w" * 20]
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 20
    assert chunks[-1].finish_reason == "length"  # the budget ran out mid-answer


# -- ReasoningSplitter --------------------------------------------------------------


def split_all(pieces: list[str], **kwargs: bool) -> tuple[str, str]:
    """Feed every piece, then flush; return the joined (reasoning, answer)."""
    splitter = ReasoningSplitter(**kwargs)
    outputs = [splitter.push(piece) for piece in pieces] + [splitter.flush()]
    return "".join(r for r, _ in outputs), "".join(a for _, a in outputs)


class TestReasoningSplitter:
    RAW = "<think>\nweigh A against B\n</think>\n\nPick B."
    EXPECTED = ("weigh A against B\n", "Pick B.")

    def test_think_block_goes_to_reasoning_and_the_rest_to_answer(self) -> None:
        splitter = ReasoningSplitter()
        assert splitter.push(self.RAW) == self.EXPECTED
        assert splitter.flush() == ("", "")

    def test_any_chunking_gives_the_same_split(self) -> None:
        # Engines cut tokens anywhere, including inside "<think>" and "</think>".
        for cut in range(len(self.RAW) + 1):
            assert split_all([self.RAW[:cut], self.RAW[cut:]]) == self.EXPECTED, cut
        assert split_all(list(self.RAW)) == self.EXPECTED

    def test_tags_split_across_chunks_hold_back_only_the_ambiguous_tail(self) -> None:
        splitter = ReasoningSplitter()
        assert splitter.push("<thi") == ("", "")  # may still become "<think>"
        assert splitter.push("nk>abc</th") == ("abc", "")  # "</th" may become "</think>"
        assert splitter.in_reasoning
        assert splitter.push("ink>Answer") == ("", "Answer")
        assert not splitter.in_reasoning

    def test_held_text_is_released_once_it_cannot_be_a_tag(self) -> None:
        splitter = ReasoningSplitter()
        assert splitter.push("<think>a<") == ("a", "")
        assert splitter.push("/") == ("", "")
        assert splitter.push("x") == ("</x", "")
        other = ReasoningSplitter()
        assert other.push("<") == ("", "")
        assert other.push("b>bold</b>") == ("", "<b>bold</b>")

    def test_a_think_tag_after_the_start_is_literal_answer_text(self) -> None:
        text = "Wrap it as <think>x</think> in the template."
        assert split_all([text]) == ("", text)

    def test_whitespace_before_the_open_tag_is_tolerated(self) -> None:
        assert split_all(["\n", "<think>x</think>y"]) == ("x", "y")

    def test_pre_opened_block_starts_mid_thought(self) -> None:
        # Templates whose prompt ends in "<think>": only "</think>" ever appears...
        assert split_all(["\nplan</think>\n\nanswer"], start_in_reasoning=True) == ("plan", "answer")
        # ...though some models open it again anyway.
        assert split_all(["<think>\nplan</think>ans"], start_in_reasoning=True) == ("plan", "ans")

    def test_flush_keeps_an_unfinished_thought_as_reasoning(self) -> None:
        splitter = ReasoningSplitter()
        assert splitter.push("<think>out of tok</thi") == ("out of tok", "")
        assert splitter.flush() == ("</thi", "")  # never shown as an answer

    def test_flush_releases_an_incomplete_open_tag_as_answer(self) -> None:
        splitter = ReasoningSplitter()
        assert splitter.push("<thi") == ("", "")
        assert splitter.flush() == ("", "<thi")


# -- OpenAI-compatible protocol ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", "stop"),
        ("eos", "stop"),
        ("length", "length"),
        ("error", "error"),
        ("cancelled", "cancelled"),
        ("tool_calls", "stop"),
        ("content_filter", "stop"),
        (None, "stop"),
    ],
)
def test_normalize_finish_reason_maps_unknown_values_to_stop(raw: str | None, expected: str) -> None:
    assert normalize_finish_reason(raw) == expected


def test_strict_openai_payload_sends_only_standard_fields() -> None:
    params = GenerationParams(
        max_tokens=16, top_k=20, min_p=0.05, repetition_penalty=1.1, thinking=True,
        stop=["a", "b", "c", "d", "e"], seed=7,
    )  # fmt: skip
    messages = [Message.user("hi")]

    loose = OpenAICompatBackend(BackendConfig(name="openai_compat", model="m"))
    payload = loose._payload(messages, params, stream=True)
    assert payload["stop"] == ["a", "b", "c", "d"]  # servers cap stop sequences at 4
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert {"top_k", "min_p", "repetition_penalty"} <= payload.keys()

    strict = OpenAICompatBackend(
        BackendConfig(name="openai_compat", model="m", extra={"strict_openai": True})
    )
    payload = strict._payload(messages, params, stream=False)
    assert payload["seed"] == 7
    assert not {"top_k", "min_p", "repetition_penalty", "chat_template_kwargs"} & payload.keys()


def failing_client(base_url: str, failure: type[httpx.TransportError]) -> httpx.AsyncClient:
    """An HTTP client whose every request fails the way a dead or stuck server does."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise failure("simulated", request=request)

    return httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (httpx.ConnectError, r"Lost the connection to http://127\.0\.0\.1:9/v1 \(ConnectError: simulated\)"),
        (httpx.ReadError, r"Lost the connection to http://127\.0\.0\.1:9/v1 \(ReadError: simulated\)"),
        (httpx.ReadTimeout, r"^http://127\.0\.0\.1:9/v1 sent nothing for 300s.*backend\.timeout_s"),
    ],
)
async def test_network_failures_become_backend_errors_with_a_hint(
    failure: type[httpx.TransportError], message: str
) -> None:
    backend = OpenAICompatBackend(
        BackendConfig(name="openai_compat", model="m", base_url="http://127.0.0.1:9/v1")
    )
    backend._client = failing_client(backend._base_url, failure)
    backend._loaded = True
    try:
        with pytest.raises(BackendError, match=message) as error:
            await backend.generate([Message.user("hi")], GenerationParams(max_tokens=4))
    finally:
        await backend._client.aclose()
    assert error.value.hint
    assert isinstance(error.value.__cause__, failure)  # the original stays attached for debugging
