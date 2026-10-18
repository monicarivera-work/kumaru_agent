"""Protect the offline echo backend and the streaming helpers every backend shares."""

import asyncio
import contextlib
import functools
import sys
import threading
import time
import types
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest

from kumaru.backends.base import Backend, ReasoningSplitter, StopSequenceTracker, iterate_in_thread
from kumaru.backends.echo import EchoBackend
from kumaru.backends.llamacpp_backend import LlamaCppBackend
from kumaru.backends.openai_compat import OpenAICompatBackend, normalize_finish_reason
from kumaru.backends.transformers_backend import TransformersBackend, _row_repetition_penalty
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


def tiny_gpt2(steps: list[int], finished: threading.Event, positions: int = 4096) -> object:
    """A random two-layer GPT-2 (no download) that counts its decoding steps.

    It has no end-of-sequence token, so only max_new_tokens or Kumaru's halt
    can end a generation: exactly what the test below needs to tell apart.
    Like any GPT-2, it fails on a sequence longer than its `positions`.
    """
    import transformers

    config = transformers.GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=100,
        n_positions=positions,
        bos_token_id=None,
        eos_token_id=None,
    )
    model = transformers.GPT2LMHeadModel(config).eval()
    forward, generate = model.forward, model.generate

    # `wraps` keeps forward's signature visible: transformers reads it to decide
    # whether to derive position ids from the attention mask, which padding needs.
    @functools.wraps(forward)
    def counting_forward(*args: object, **kwargs: object) -> object:
        steps.append(1)
        time.sleep(0.002)
        return forward(*args, **kwargs)

    @functools.wraps(generate)
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


# -- transformers: greedy micro-batching ----------------------------------------------

THINK, END_THINK, EOS, EOS_2 = 90, 91, 50, 51


def needs_transformers() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")


class ScriptTokenizer:
    """Reads each number in a text as a token id (any other word is id 1).

    It decodes id n as "n ", and 90/91 as <think>/</think>. So a user message
    such as "5 7 50" is its own reply script: see ScriptedModel.
    """

    pad_token_id, chat_template = 0, None

    def __init__(self, eos: int | None = EOS) -> None:
        self.eos_token_id = eos

    def __call__(self, text: str, return_tensors: str | None = None, **kwargs: object) -> dict:
        ids = [int(word) if word.isdigit() else 1 for word in text.split()]
        if return_tensors is None:
            return {"input_ids": ids}
        import torch

        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def decode(self, ids: list[int], skip_special_tokens: bool = False, **kwargs: object) -> str:
        skipped = {EOS, EOS_2, self.pad_token_id} if skip_special_tokens else set()
        names = {THINK: "<think>", END_THINK: "</think>"}
        return "".join(names.get(i, f"{i} ") for i in ids if i not in skipped)


class ScriptedModel:
    """Replies with the numbers in each prompt, finishing rows as `generate` does.

    A row's prompt is what its attention mask keeps, so left padding is
    honoured. A row ends at an end-of-sequence id or when a stopping criterion
    says so, and is padded after that; past its script it repeats RAMBLE, like
    a model that will not stop. With a streamer, tokens reach it one by one.
    """

    RAMBLE = 33
    device = "cpu"

    def __init__(self, configured_eos: object = (EOS, EOS_2), fits: int = 64) -> None:
        self.generation_config = types.SimpleNamespace(eos_token_id=configured_eos)
        self.fits = fits  # rows per call before it "runs out of GPU memory"
        self.calls: list[dict] = []

    def generate(
        self,
        input_ids: object,
        attention_mask: object,
        max_new_tokens: int,
        pad_token_id: int = 0,
        eos_token_id: object = None,
        stopping_criteria: object = (),
        streamer: object = None,
        **kwargs: object,
    ) -> object:
        import torch

        self.calls.append(
            {"rows": len(input_ids), "streamer": streamer is not None, "eos": eos_token_id, **kwargs}
        )
        if len(input_ids) > self.fits:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
        ends = set(eos_token_id) if isinstance(eos_token_id, list) else {eos_token_id}
        scripts = [
            row[mask.bool()].tolist()[1:-1]
            for row, mask in zip(input_ids, attention_mask, strict=True)
        ]
        if streamer is not None:
            streamer.put(input_ids)
        done = [False] * len(scripts)
        for step in range(max_new_tokens):
            column = [
                pad_token_id if finished else script[step] if step < len(script) else self.RAMBLE
                for finished, script in zip(done, scripts, strict=True)
            ]
            input_ids = torch.cat([input_ids, torch.tensor([column]).T], dim=1)
            if streamer is not None:
                streamer.put(torch.tensor(column))
            done = [
                finished or token in ends for finished, token in zip(done, column, strict=True)
            ]
            for criterion in stopping_criteria:
                stops = criterion(input_ids, None).tolist()
                done = [finished or stop for finished, stop in zip(done, stops, strict=True)]
            if all(done):
                break
        if streamer is not None:
            streamer.end()
        return input_ids


def batching_backend(
    max_batch_size: int,
    model: object = None,
    tokenizer: object = None,
    max_context_length: int = 8192,
) -> TransformersBackend:
    backend = TransformersBackend(
        BackendConfig(
            name="transformers",
            model="scripted",
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
        )
    )
    backend.model = ScriptedModel() if model is None else model
    backend.tokenizer = ScriptTokenizer() if tokenizer is None else tokenizer
    backend._loaded = True
    return backend


def greedy(max_tokens: int, **kwargs: object) -> GenerationParams:
    return GenerationParams(max_tokens=max_tokens, temperature=0.0, **kwargs)


async def wait_until(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        await asyncio.sleep(0.005)


# name: (reply script, max_tokens, stop) -> (content, reasoning, finish_reason, completion_tokens)
REPLIES = {
    "runs to its budget": (("5 7 9", 6, []), ("5 7 9 33 33 33 ", None, "length", 6)),
    "ends at end-of-sequence": (("5 50 7", 6, []), ("5 ", None, "stop", 2)),
    "ends at the checkpoint's other end token": (("5 51 7", 6, []), ("5 ", None, "stop", 2)),
    "ends on its last allowed token": (("5 6 50", 3, []), ("5 6 ", None, "stop", 3)),
    "is cut by a stop sequence": (("5 7 9", 3, ["7 "]), ("5 ", None, "stop", 3)),
    "thinks first": (("90 5 91 6 50", 9, []), ("6 ", "5 ", "stop", 5)),
}


@pytest.mark.asyncio
async def test_greedy_requests_made_together_share_one_batch_and_end_like_streamed_ones() -> None:
    needs_transformers()
    batched, streamed = batching_backend(8), batching_backend(1)

    async def ask(backend: TransformersBackend, script: str, max_tokens: int, stop: list[str]):
        return await backend.generate([Message.user(script)], greedy(max_tokens, stop=stop))

    together = await asyncio.gather(*(ask(batched, *request) for request, _ in REPLIES.values()))

    assert [call["rows"] for call in batched.model.calls] == [len(REPLIES)]
    for (name, (request, expected)), result in zip(REPLIES.items(), together, strict=True):
        got = result.message.content, result.message.reasoning, result.finish_reason
        assert (*got, result.usage.completion_tokens) == expected, name
        assert result.usage.prompt_tokens == len(request[0].split()) + 2  # + "USER:", "ASSISTANT:"
        alone = await ask(streamed, *request)
        assert (alone.message.content, alone.message.reasoning, alone.finish_reason) == got, name
        if request[2]:  # a stop sequence ends a stream early, so it may count fewer tokens
            assert alone.usage.completion_tokens <= result.usage.completion_tokens
        else:
            assert alone.usage == result.usage, name
    # Both paths end a reply at every end token the checkpoint declares.
    assert all(
        call["eos"] == [EOS, EOS_2] for call in batched.model.calls + streamed.model.calls
    )


@pytest.mark.asyncio
async def test_a_batch_holds_at_most_max_batch_size_requests() -> None:
    needs_transformers()
    backend = batching_backend(2)

    replies = await asyncio.gather(
        *(backend.generate([Message.user(f"{n} 50")], greedy(4)) for n in range(2, 7))
    )

    assert [reply.message.content for reply in replies] == ["2 ", "3 ", "4 ", "5 ", "6 "]
    assert [call["rows"] for call in backend.model.calls] == [2, 2, 1]
    assert backend.info().extra["max_batch_size"] == 2


@pytest.mark.asyncio
async def test_a_caller_that_gives_up_while_queued_is_never_decoded() -> None:
    needs_transformers()
    backend = batching_backend(2)
    callers = [
        asyncio.create_task(backend.generate([Message.user(f"{n} 50")], greedy(4)))
        for n in (2, 3, 4)
    ]
    await asyncio.sleep(0)  # all three are queued; the first two make a batch
    callers[2].cancel()

    outcomes = await asyncio.gather(*callers, return_exceptions=True)
    await wait_until(lambda: backend._drainer is not None and backend._drainer.done())

    assert [reply.message.content for reply in outcomes[:2]] == ["2 ", "3 "]
    assert isinstance(outcomes[2], asyncio.CancelledError)
    assert [call["rows"] for call in backend.model.calls] == [2]


@pytest.mark.asyncio
async def test_sampled_requests_are_never_batched() -> None:
    needs_transformers()
    backend = batching_backend(8)
    sampled = GenerationParams(max_tokens=4, temperature=0.7)

    await asyncio.gather(*(backend.generate([Message.user("5 50")], sampled) for _ in range(3)))

    assert [(call["rows"], call["streamer"]) for call in backend.model.calls] == [(1, True)] * 3


@pytest.mark.asyncio
@pytest.mark.parametrize("max_batch_size", [1, 8])
async def test_greedy_decoding_applies_the_requested_repetition_penalty(max_batch_size: int) -> None:
    needs_transformers()
    backend = batching_backend(max_batch_size)

    await backend.generate([Message.user("5 50")], greedy(4, repetition_penalty=1.3))

    [call] = backend.model.calls
    if max_batch_size == 1:
        # Passed even though greedy: otherwise the checkpoint's own default applies.
        assert call["repetition_penalty"] == 1.3
    else:
        # Applied row by row instead, so padding is never penalised.
        assert call["repetition_penalty"] == 1.0
        assert [type(p).__name__ for p in call["logits_processor"]] == ["RowRepetitionPenalty"]


@pytest.mark.parametrize(
    ("penalties", "dtype"),
    [([1.3] * 3, "float32"), ([1.3, 0.8, 1.5], "float32"), ([1.3] * 3, "bfloat16")],
)
def test_row_repetition_penalty_is_the_built_in_penalty_on_each_unpadded_row(
    penalties: list[float], dtype: str
) -> None:
    needs_transformers()
    import torch
    from transformers import RepetitionPenaltyLogitsProcessor

    # Left-padded prompts (pad id 0), then two generated tokens each. Row 2
    # catches padding being penalised; row 3 generates 0 itself, a real token
    # that must still be penalised.
    input_ids = torch.tensor([[7, 3, 9, 4, 3], [0, 0, 5, 6, 5], [0, 0, 8, 0, 2]])
    padding = torch.tensor([[False, False, False], [True, True, False], [True, True, False]])
    unpadded = [[7, 3, 9, 4, 3], [5, 6, 5], [8, 0, 2]]
    scores = torch.randn(3, 12, generator=torch.Generator().manual_seed(0)).to(getattr(torch, dtype))

    processed = _row_repetition_penalty(penalties, padding, [7, 5, 8])(input_ids, scores)

    for row, (ids, penalty) in enumerate(zip(unpadded, penalties, strict=True)):
        alone = RepetitionPenaltyLogitsProcessor(penalty)(torch.tensor([ids]), scores[row : row + 1])
        assert torch.equal(processed[row : row + 1], alone), row


@pytest.mark.asyncio
@pytest.mark.parametrize("repetition_penalty", [1.0, 1.3])
async def test_a_batched_reply_is_the_reply_the_model_gives_alone(repetition_penalty: float) -> None:
    needs_transformers()
    steps: list[int] = []
    model = tiny_gpt2(steps, threading.Event())
    batched = batching_backend(8, model, ScriptTokenizer(eos=None))
    alone = batching_backend(1, model, ScriptTokenizer(eos=None))
    # Different lengths, so the shorter prompts are padded on the left.
    prompts = ["3", "14 15 92", "65 35 89 79 32 38 46"]
    params = greedy(12, repetition_penalty=repetition_penalty)

    together = await asyncio.gather(*(batched.generate([Message.user(p)], params) for p in prompts))

    assert len(steps) == 12  # one forward pass per token, for all three prompts at once
    for prompt, result in zip(prompts, together, strict=True):
        single = await alone.generate([Message.user(prompt)], params)
        assert (result.message.content, result.usage) == (single.message.content, single.usage)


@pytest.mark.asyncio
async def test_a_batch_never_spans_more_than_the_context_window() -> None:
    needs_transformers()
    steps: list[int] = []
    model = tiny_gpt2(steps, threading.Event(), positions=32)
    batched = batching_backend(8, model, ScriptTokenizer(eos=None), max_context_length=32)
    alone = batching_backend(1, model, ScriptTokenizer(eos=None), max_context_length=32)
    batches: list[list[tuple[int, int]]] = []
    run_batch = batched._run_batch

    def recording(batch: Any) -> Any:
        batches.append([(len(request.ids), request.max_new) for request in batch.requests])
        return run_batch(batch)

    batched._run_batch = recording  # type: ignore[method-assign]
    # (prompt, max_tokens). A prompt of k numbers is k + 2 tokens, and each
    # budget is cut to what the 32-token window leaves it.
    requests = [(" ".join(map(str, range(10, 33))), 100), ("3", 100), ("3", 4)]

    together = await asyncio.gather(
        *(batched.generate([Message.user(p)], greedy(n)) for p, n in requests)
    )

    # Together, the first two would span 25 + 29 = 54 tokens and run GPT-2 past
    # its 32 positions. The third fits beside the first exactly, so it joins it.
    assert batches == [[(25, 7), (3, 4)], [(3, 29)]]
    assert len(steps) == 7 + 29
    assert [result.usage.completion_tokens for result in together] == [7, 29, 4]
    for (prompt, max_tokens), result in zip(requests, together, strict=True):
        single = await alone.generate([Message.user(prompt)], greedy(max_tokens))
        assert (result.message.content, result.usage) == (single.message.content, single.usage)


@pytest.mark.asyncio
async def test_a_batch_stops_once_every_caller_has_gone() -> None:
    needs_transformers()
    steps: list[int] = []
    finished = threading.Event()
    backend = batching_backend(4, tiny_gpt2(steps, finished), ScriptTokenizer(eos=None))
    callers = [
        asyncio.create_task(backend.generate([Message.user(p)], greedy(2000))) for p in ("1 2", "3")
    ]
    await wait_until(lambda: len(steps) >= 3)
    assert [len(batch.requests) for batch in backend._running] == [2]

    callers[0].cancel()
    await asyncio.sleep(0.05)
    assert not finished.is_set()  # the other caller still wants its reply
    callers[1].cancel()

    assert await asyncio.to_thread(finished.wait, 5)
    assert len(steps) < 200  # not the 2000 it would run to unattended
    await asyncio.gather(*callers, return_exceptions=True)
    await backend.aclose()


@pytest.mark.asyncio
async def test_closing_the_backend_fails_waiting_callers_and_stops_their_batch() -> None:
    needs_transformers()
    steps: list[int] = []
    finished = threading.Event()
    backend = batching_backend(2, tiny_gpt2(steps, finished), ScriptTokenizer(eos=None))
    callers = [
        asyncio.create_task(backend.generate([Message.user(p)], greedy(2000))) for p in "123"
    ]
    await wait_until(lambda: len(steps) >= 3)
    assert len(backend._queue) == 1  # two being decoded, one waiting for room

    await backend.aclose()

    assert finished.is_set()  # decoding stopped before the model was released
    assert len(steps) < 200 and backend.model is None
    outcomes = await asyncio.gather(*callers, return_exceptions=True)
    assert all(isinstance(o, BackendError) and "closed" in str(o) for o in outcomes), outcomes


@pytest.mark.asyncio
async def test_requests_made_while_the_backend_closes_are_refused() -> None:
    needs_transformers()
    steps: list[int] = []
    finished = threading.Event()
    backend = batching_backend(2, tiny_gpt2(steps, finished), ScriptTokenizer(eos=None))
    caller = asyncio.create_task(backend.generate([Message.user("1")], greedy(2000)))
    await wait_until(lambda: len(steps) >= 3)

    closing = asyncio.create_task(backend.aclose())
    await asyncio.sleep(0)  # the close has begun and waits for the decoding thread
    assert not closing.done()
    # A server swapping models still holds the old agent, so requests can arrive now.
    # Accepted, a batched one would start a batch that the close never stops.
    for params in (greedy(2000), GenerationParams(max_tokens=2000, temperature=0.7)):
        with pytest.raises(BackendError, match="closing"):
            await backend.generate([Message.user("2")], params)
    await closing

    assert finished.is_set() and backend.model is None
    decoded = len(steps)
    await asyncio.sleep(0.05)
    assert len(steps) == decoded < 200  # nothing is decoding any more
    with pytest.raises(BackendError, match="closed"):
        await caller


@pytest.mark.asyncio
async def test_a_close_during_a_load_waits_for_it_and_releases_what_it_loaded() -> None:
    needs_transformers()
    backend = batching_backend(2)
    model, tokenizer = backend.model, backend.tokenizer
    backend.model = backend.tokenizer = None
    backend._loaded = False
    may_finish = asyncio.Event()

    async def slow_load() -> None:
        await may_finish.wait()
        backend.model, backend.tokenizer = model, tokenizer

    backend._load = slow_load  # type: ignore[method-assign]
    caller = asyncio.create_task(backend.generate([Message.user("5 50")], greedy(4)))
    await wait_until(lambda: backend._load_lock is not None and backend._load_lock.locked())
    closing = asyncio.create_task(backend.aclose())
    await asyncio.sleep(0.05)
    assert not closing.done()  # it waits for the load instead of racing it
    may_finish.set()

    with pytest.raises(BackendError, match="closing"):
        await caller
    await closing
    assert backend.model is None and not backend.is_loaded
    assert model.calls == []  # loaded, then released without decoding anything


@pytest.mark.asyncio
async def test_a_batch_too_big_for_the_gpu_is_retried_in_halves() -> None:
    needs_transformers()
    backend = batching_backend(4, ScriptedModel(fits=1))

    replies = await asyncio.gather(
        *(backend.generate([Message.user(f"{n} 50")], greedy(4)) for n in (2, 3))
    )

    assert [reply.message.content for reply in replies] == ["2 ", "3 "]
    assert [call["rows"] for call in backend.model.calls] == [2, 1, 1]
    assert backend.info().extra["max_batch_size"] == 1  # remembered until the next load


@pytest.mark.asyncio
async def test_a_single_request_too_big_for_the_gpu_fails_with_advice() -> None:
    needs_transformers()
    backend = batching_backend(4, ScriptedModel(fits=0))

    with pytest.raises(BackendError, match="Out of GPU memory") as caught:
        await backend.generate([Message.user("5 50")], greedy(4))
    assert "--max-tokens" in (caught.value.hint or "")


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
