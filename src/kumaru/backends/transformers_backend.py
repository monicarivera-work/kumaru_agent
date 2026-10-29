"""Local GPU/CPU inference with Hugging Face `transformers`.

This is the default backend and the one that runs on your own machine. It is
also the backend that consumes your LoRA adapters, so `kumaru train` and
`kumaru serve` close the loop here.

Performance model (worth internalising before you tune anything)
----------------------------------------------------------------
Single-stream decoding is **memory-bandwidth bound**, not compute bound. Each
generated token requires reading essentially every model weight from VRAM once.
So::

    tokens/sec  ~=  memory_bandwidth / bytes_of_weights_read_per_token

On an RTX 3090 (~936 GB/s) with a 7B model in bfloat16 (~15 GB), the ceiling is
roughly 936 / 15 ~= 60 tok/s, and you should expect 30-45 tok/s in practice.
Three consequences follow directly, and they explain nearly every tuning
decision in this file:

1. **Quantization buys speed, not just memory.** 4-bit halves the bytes read per
   token, so it roughly doubles the ceiling — at some quality cost.
2. **Batching is nearly free.** Weights are read once per *step*, not per
   sequence, so serving 8 concurrent users costs far less than 8x. This is why
   vLLM's continuous batching is the single biggest throughput upgrade available
   (see `openai_compat.py` and docs/SCALABILITY.md). This backend batches only
   greedy requests made at the same time (`backend.max_batch_size`), which is
   exactly what a benchmark sends: see `generate()`.
3. **Prompt processing is different.** Prefill is compute bound and highly
   parallel, which is why a 4,000-token prompt is processed far faster than
   4,000 tokens are generated.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import functools
import importlib.util
import os
import threading
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from kumaru.backends.base import (
    Backend,
    ReasoningSplitter,
    StopSequenceTracker,
    iterate_in_thread,
)
from kumaru.core.config import BackendConfig
from kumaru.core.errors import BackendError, DependencyMissingError
from kumaru.core.logging import get_logger
from kumaru.core.registry import BACKENDS
from kumaru.core.types import (
    BackendInfo,
    ChatResult,
    GenerationParams,
    Message,
    StreamChunk,
    Usage,
)

logger = get_logger(__name__)


@dataclass(eq=False)
class _Request:
    """One greedy request, waiting for a batch or riding in one."""

    ids: list[int]  # the rendered prompt, tokenized
    max_new: int  # this request's own output budget
    repetition_penalty: float
    stop: list[str]
    starts_in_reasoning: bool
    future: asyncio.Future[ChatResult]
    started: float  # time.perf_counter() when it was submitted


@dataclass(eq=False)
class _Batch:
    """Requests decoded by one `model.generate` call, and its stop signals.

    `dropped` and the two events are shared with the worker thread: the event
    loop writes them, the decoding loop reads them after every token.
    """

    requests: list[_Request]
    halt: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    dropped: list[bool] = field(init=False)

    def __post_init__(self) -> None:
        self.dropped = [False] * len(self.requests)

    def settled(self, index: int, future: asyncio.Future[ChatResult]) -> None:
        """A caller that gave up frees its row; once all have, the batch stops."""
        if future.cancelled():
            self.dropped[index] = True
            if all(self.dropped):
                self.halt.set()


class _OutOfMemory(Exception):
    """A batch did not fit in GPU memory; a smaller one may."""


def _row_repetition_penalty(penalties: list[float], padding: Any, fillers: list[int]) -> Any:
    """transformers' repetition penalty, applied to a left-padded batch row by row.

    The built-in processor penalises every id in `input_ids`, padding included,
    so batching would penalise a short prompt's pad token (often end-of-text)
    and change its reply. Here each pad column is swapped for a token its row
    already holds, which leaves the set of penalised tokens exactly what it is
    when the prompt is decoded alone. Rows may carry different penalties.
    """
    import torch
    from transformers import LogitsProcessor

    device = padding.device
    width = padding.shape[1]
    filler = torch.tensor(fillers, dtype=torch.long, device=device).unsqueeze(1)
    uniform = len(set(penalties)) == 1
    rows = torch.tensor(penalties, dtype=torch.float32, device=device).unsqueeze(1)

    class RowRepetitionPenalty(LogitsProcessor):
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            ids = input_ids.clone()
            ids[:, :width] = torch.where(padding, filler, ids[:, :width])
            score = torch.gather(scores, 1, ids)
            if uniform:
                # A Python float, as the built-in processor uses: the same
                # arithmetic to the last bit.
                penalty = penalties[0]
                score = torch.where(score < 0, score * penalty, score / penalty)
            else:
                # In float32, as a Python float is applied: bfloat16 would
                # round a penalty of 1.05 to 1.046875.
                wide = score.float()
                score = torch.where(wide < 0, wide * rows, wide / rows).to(scores.dtype)
            return scores.scatter(1, ids, score)

    return RowRepetitionPenalty()


@BACKENDS.register("transformers", aliases=("hf",))
class TransformersBackend(Backend):
    """Run a Hugging Face causal LM locally, with optional 4/8-bit and LoRA."""

    name = "transformers"
    supports_adapters = True

    #: A batch starts once it is full, once no request has arrived for
    #: BATCH_IDLE_S, or BATCH_MAX_WAIT_S after the first one queued. A
    #: benchmark refills a finished batch within milliseconds per question, so
    #: neither wait is noticeable next to a reply that takes seconds.
    BATCH_IDLE_S = 0.02
    BATCH_MAX_WAIT_S = 0.25

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self.model: Any = None
        self.tokenizer: Any = None
        self._device: str = "cpu"
        self._dtype_name: str = "unknown"
        # Greedy micro-batching; see generate().
        self._queue: collections.deque[_Request] = collections.deque()
        self._arrival: asyncio.Event | None = None
        self._drainer: asyncio.Task[None] | None = None
        self._running: list[_Batch] = []
        # Lowered after an out-of-memory error, for the rest of this load.
        self._batch_cap: int | None = None
        # aclose() calls under way. New requests are refused meanwhile, so the
        # batches a close stops are all the batches there are.
        self._closing = 0

    # -- hardware resolution ------------------------------------------------

    @staticmethod
    def _resolve_device(requested: str) -> str:
        import torch

        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_dtype(requested: str, device: str) -> Any:
        """Pick a numeric format.

        `bfloat16` is preferred over `float16` on Ampere and newer: it has the
        same exponent range as float32, so it does not overflow to inf during
        long generations. `float16` has more mantissa bits but a narrow range,
        which is the usual cause of NaN outputs on large models.
        """
        import torch

        named = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if requested != "auto":
            if requested not in named:
                raise BackendError(f"Unknown dtype '{requested}'", hint=f"Use one of: {list(named)}")
            return named[requested]

        if device.startswith("cuda"):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        # CPU bfloat16 is slow on most consumer chips; float32 is the safe default.
        return torch.float32

    # -- loading ------------------------------------------------------------

    async def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise DependencyMissingError("transformers", extra="torch") from exc

        cfg = self.config
        self._device = self._resolve_device(cfg.device)
        dtype = self._resolve_dtype(cfg.dtype, self._device)
        self._dtype_name = str(dtype).replace("torch.", "")
        self._batch_cap = None

        logger.info("loading %s on %s (%s)", cfg.model, self._device, self._dtype_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model, trust_remote_code=cfg.trust_remote_code
        )
        # Many base checkpoints omit a pad token. Generation needs one for
        # attention masking; reusing EOS is the standard, harmless fix.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": cfg.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if cfg.attn_implementation:
            model_kwargs["attn_implementation"] = cfg.attn_implementation

        if cfg.load_in_4bit or cfg.load_in_8bit:
            model_kwargs["quantization_config"] = self._quantization_config(dtype)

        # `device_map="auto"` lets accelerate shard across multiple GPUs and
        # spill to CPU when VRAM runs short. This one line is what allows a
        # config written for one 3090 to run unchanged on an 8-GPU node.
        if self._device.startswith("cuda"):
            model_kwargs["device_map"] = cfg.extra.get("device_map", "auto")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model, **model_kwargs)
        except TypeError:
            # transformers < 4.56 spells the dtype argument `torch_dtype`.
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model, **model_kwargs)

        if not self._device.startswith("cuda") and "device_map" not in model_kwargs:
            self.model = self.model.to(self._device)

        if cfg.adapter_path:
            self._attach_adapter(cfg.adapter_path)

        self.model.eval()
        # Inference never needs gradients; disabling globally for this module
        # avoids accidentally building a graph and leaking VRAM.
        torch.set_grad_enabled(False)

        detected = self._detect_context_length()
        if detected:
            self.config.max_context_length = min(cfg.max_context_length, detected)

    def _quantization_config(self, dtype: Any) -> Any:
        # transformers ships BitsAndBytesConfig even when the bitsandbytes library
        # it drives is missing, and would then fail later with a generic pip hint.
        if importlib.util.find_spec("bitsandbytes") is None:
            raise DependencyMissingError("bitsandbytes", extra="quant")
        from transformers import BitsAndBytesConfig

        if self.config.load_in_4bit:
            # NF4 + double quantization is the QLoRA recipe: it is
            # information-theoretically matched to normally-distributed weights
            # and loses markedly less quality than naive int4 rounding.
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    def _attach_adapter(self, adapter_path: str) -> None:
        """Layer a trained LoRA adapter on top of the base weights."""
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise DependencyMissingError("peft", extra="train") from exc

        if not os.path.exists(adapter_path):
            raise BackendError(
                f"Adapter not found: {adapter_path}",
                hint="Train one with `kumaru train configs/train/lora-qwen.yaml`",
            )
        logger.info("attaching LoRA adapter: %s", adapter_path)
        self.model = PeftModel.from_pretrained(self.model, adapter_path)

        if self.config.extra.get("merge_adapter", False):
            # Merging folds the adapter into the base weights: no per-layer
            # overhead at inference, at the cost of no longer being able to
            # swap adapters at runtime.
            self.model = self.model.merge_and_unload()
            logger.info("merged adapter into base weights")

    def _detect_context_length(self) -> int | None:
        for attr in ("max_position_embeddings", "n_positions", "seq_length"):
            value = getattr(self.model.config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    async def aclose(self) -> None:
        # Refuse new requests before the first await below: one accepted
        # meanwhile could start a batch that nothing here stops or waits for.
        self._closing += 1
        try:
            # Callers still waiting on a batch get an answer rather than a hang,
            # and the decoding thread stops before the model is released.
            running, self._running = self._running, []
            for batch in running:
                batch.halt.set()
            queued = list(self._queue)
            self._queue.clear()
            self._fail(
                queued + [request for batch in running for request in batch.requests],
                "The backend was closed before this request finished.",
            )
            drainer, self._drainer = self._drainer, None
            if (
                drainer is not None
                and not drainer.done()
                and drainer.get_loop() is asyncio.get_running_loop()
            ):
                drainer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await drainer
            for batch in running:
                await asyncio.to_thread(batch.finished.wait, 10.0)
            # A load under way finishes first, so the model it loads is
            # released here too rather than outliving the close.
            if self._load_lock is None:
                self._release()
            else:
                async with self._load_lock:
                    self._release()
        finally:
            self._closing -= 1

    def _refuse_if_closing(self) -> None:
        if self._closing:
            raise BackendError(
                "The backend is closing, so it cannot take new requests.",
                hint="Retry once the model switch or shutdown has finished.",
            )

    def _release(self) -> None:
        """Drop the model and tokenizer, and return their memory to the GPU."""
        self.model = None
        self.tokenizer = None
        self._loaded = False
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    # -- introspection ------------------------------------------------------

    def info(self) -> BackendInfo:
        quant = "4bit-nf4" if self.config.load_in_4bit else "8bit" if self.config.load_in_8bit else None
        params = None
        if self.model is not None:
            try:
                params = sum(p.numel() for p in self.model.parameters())
            except Exception:  # pragma: no cover
                params = None
        return BackendInfo(
            name=self.name,
            model_id=self.config.model,
            device=self._device,
            dtype=self._dtype_name,
            context_length=self.config.max_context_length,
            supports_streaming=True,
            quantization=quant,
            extra={
                "parameters": params,
                "adapter": self.config.adapter_path,
                "max_batch_size": self._cap(),
            },
        )

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Exact count using the model's own chat template."""
        if self.tokenizer is None:
            return super().count_tokens(messages)
        try:
            text = self._render_prompt(messages)
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:  # pragma: no cover - never fail a request over accounting
            return super().count_tokens(messages)

    def _render_prompt(self, messages: Sequence[Message], thinking: bool | None = None) -> str:
        """Turn structured messages into the exact string the model expects.

        Chat models are trained with specific control tokens (Qwen uses
        `<|im_start|>`, Llama uses `<|start_header_id|>`, ...). Getting these
        wrong is the most common cause of "the model works but its answers are
        subtly bad": it is being asked a question in a format it never saw
        during training. `apply_chat_template` reads the format out of the
        checkpoint, so it is always right for whatever model you loaded.

        `thinking` becomes the template variable `enable_thinking` (the Qwen3 /
        DeepSeek / GLM convention). Templates that do not use it ignore it.
        """
        payload = [m.to_openai() for m in messages]
        if getattr(self.tokenizer, "chat_template", None):
            extra: dict[str, Any] = {} if thinking is None else {"enable_thinking": thinking}
            return self.tokenizer.apply_chat_template(
                payload, tokenize=False, add_generation_prompt=True, **extra
            )
        # Fallback for base (non-instruct) checkpoints with no template.
        lines = [f"{m['role'].upper()}: {m['content']}" for m in payload]
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    # -- generation ---------------------------------------------------------

    async def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        *,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.check_tools(tools)
        self._refuse_if_closing()
        if not self._loaded:
            await self.load()
            self._refuse_if_closing()  # a close may have begun during the load

        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        tokenizer, model = self.tokenizer, self.model
        prompt = self._render_prompt(messages, thinking=params.thinking)
        # Some templates pre-open the thinking block in the prompt, so the
        # output starts mid-thought and only "</think>" ever appears.
        splitter = ReasoningSplitter(start_in_reasoning=prompt.rstrip().endswith("<think>"))
        eos = self._eos_ids()

        def produce():
            import threading

            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            max_new = min(params.max_tokens, self._budget(prompt_tokens))

            device = getattr(model, "device", self._device)
            encoded = {k: v.to(device) for k, v in encoded.items()}

            if params.seed is not None:
                torch.manual_seed(params.seed)

            class CountingStreamer(TextIteratorStreamer):
                """Also counts tokens: the text it yields comes in whole words."""

                def __init__(self) -> None:
                    super().__init__(
                        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600
                    )
                    self.prompt_seen = False
                    self.generated = 0
                    self.last_token: int | None = None

                def put(self, value: Any) -> None:
                    # `generate` passes the prompt first, then each new token.
                    if self.prompt_seen:
                        ids = value.reshape(-1).tolist()
                        self.generated += len(ids)
                        self.last_token = ids[-1] if ids else self.last_token
                    self.prompt_seen = True
                    super().put(value)

            streamer = CountingStreamer()
            # `generate` runs to max_new_tokens unless told otherwise; this flag,
            # checked after every token, tells it nobody wants the rest.
            halt = threading.Event()

            class HaltWhenAsked(StoppingCriteria):
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
                    return torch.full(
                        (input_ids.shape[0],),
                        halt.is_set(),
                        dtype=torch.bool,
                        device=input_ids.device,
                    )

            kwargs: dict[str, Any] = {
                **encoded,
                "streamer": streamer,
                "max_new_tokens": max_new,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": eos,
                # Passed for greedy decoding too: otherwise the checkpoint's own
                # default applies (Qwen2.5-Instruct ships 1.1), not the one asked for.
                "repetition_penalty": params.repetition_penalty,
                "stopping_criteria": StoppingCriteriaList([HaltWhenAsked()]),
            }
            if params.temperature > 0:
                kwargs.update(
                    do_sample=True,
                    temperature=params.temperature,
                    top_p=params.top_p,
                )
                if params.top_k > 0:
                    kwargs["top_k"] = params.top_k
                if params.min_p:
                    kwargs["min_p"] = params.min_p
                # presence_penalty and thinking_budget have no transformers
                # equivalent; they are honoured by the llama-server backend.
            else:
                # temperature == 0 means greedy. Passing sampling knobs alongside
                # do_sample=False makes transformers emit warnings and ignore them.
                kwargs["do_sample"] = False

            # `generate` blocks until done, so it runs on its own thread while we
            # drain the streamer queue here.
            worker = threading.Thread(target=model.generate, kwargs=kwargs, daemon=True)
            worker.start()

            tracker = StopSequenceTracker(params.stop)
            try:
                for piece in streamer:
                    if not piece:
                        continue
                    thought, text = splitter.push(piece)
                    safe = tracker.push(text) if text else ""
                    if thought or safe:
                        yield StreamChunk(delta=safe, reasoning=thought)
                    if tracker.stopped:
                        break
            finally:
                # Reached when generate finished, a stop sequence matched, or
                # the reader went away (the yield above raised GeneratorExit).
                halt.set()
                worker.join(timeout=1.0)

            thought, text = splitter.flush()
            tail = (tracker.push(text) if text else "") + tracker.flush()
            if thought or tail:
                yield StreamChunk(delta=tail, reasoning=thought)

            # Cut off only if the budget ran out on a token other than
            # end-of-sequence: a model can finish exactly on its last allowed one.
            truncated = streamer.generated >= max_new and streamer.last_token not in (eos or ())
            reason = "length" if truncated and not tracker.stopped else "stop"
            yield StreamChunk(
                finish_reason=reason,
                usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=streamer.generated),
            )

        async with contextlib.aclosing(iterate_in_thread(produce)) as chunks:
            async for chunk in chunks:
                yield chunk

    def _budget(self, prompt_tokens: int) -> int:
        """Tokens left for the reply once the prompt is in the context window."""
        budget = self.config.max_context_length - prompt_tokens
        if budget <= 0:
            raise BackendError(
                f"Prompt is {prompt_tokens} tokens but the context window is "
                f"{self.config.max_context_length}.",
                hint="Lower agent.max_history_tokens or raise backend.max_context_length.",
            )
        return budget

    def _eos_ids(self) -> list[int] | None:
        """Every token that ends a reply: the tokenizer's, plus the checkpoint's.

        Instruct checkpoints often list several in generation_config.json
        (Qwen2.5: `<|im_end|>` and `<|endoftext|>`). Passing only the
        tokenizer's would replace that list, so a reply could run on past an
        end-of-text the model emitted.
        """
        configured = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        found: list[int] = []
        for value in (self.tokenizer.eos_token_id, configured):
            for token in value if isinstance(value, (list, tuple)) else [value]:
                if isinstance(token, int) and token not in found:
                    found.append(token)
        return found or None

    # -- greedy micro-batching ------------------------------------------------

    async def generate(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> ChatResult:
        """Like `stream()`, but greedy requests made together share one batch.

        With `backend.max_batch_size` above 1, greedy requests (temperature 0)
        that arrive within milliseconds of each other are decoded by a single
        `model.generate` call, up to that many at a time. A benchmark asks
        exactly that way, and the batch multiplies its throughput because each
        decoding step reads the weights once for every row at once. Sampled
        requests, and all requests when batching is off, stream one at a time.
        """
        if self.config.max_batch_size <= 1 or params.temperature > 0:
            return await super().generate(messages, params)
        self._refuse_if_closing()
        if not self._loaded:
            await self.load()
            self._refuse_if_closing()  # a close may have begun during the load
        # No await from here until the request is queued, where a close finds it.
        request = self._prepare(messages, params)
        self._queue.append(request)
        loop = asyncio.get_running_loop()
        drainer = self._drainer
        if drainer is None or drainer.done() or drainer.get_loop() is not loop:
            self._arrival = asyncio.Event()
            self._drainer = loop.create_task(self._drain())
        assert self._arrival is not None
        self._arrival.set()
        return await request.future

    def _prepare(self, messages: Sequence[Message], params: GenerationParams) -> _Request:
        prompt = self._render_prompt(messages, thinking=params.thinking)
        ids = list(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        return _Request(
            ids=ids,
            max_new=min(params.max_tokens, self._budget(len(ids))),
            repetition_penalty=params.repetition_penalty,
            stop=list(params.stop),
            starts_in_reasoning=prompt.rstrip().endswith("<think>"),
            future=asyncio.get_running_loop().create_future(),
            started=time.perf_counter(),
        )

    def _cap(self) -> int:
        """Rows per batch: the configured maximum, or less after running out of memory."""
        return min(self.config.max_batch_size, self._batch_cap or self.config.max_batch_size)

    async def _drain(self) -> None:
        """Decode queued requests a batch at a time until none are left."""
        try:
            while self._queue:
                await self._gather()
                batch = self._take_batch()
                if batch:
                    await self._decode(batch)
        except Exception as exc:  # a bug here must not leave callers waiting forever
            logger.exception("the batch scheduler failed")
            queued = list(self._queue)
            self._queue.clear()
            self._fail(queued, "The batch scheduler failed", cause=exc)

    async def _gather(self) -> None:
        """Wait for a full batch, a lull between arrivals, or BATCH_MAX_WAIT_S."""
        assert self._arrival is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.BATCH_MAX_WAIT_S
        while len(self._queue) < self._cap():
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            self._arrival.clear()
            try:
                await asyncio.wait_for(self._arrival.wait(), min(self.BATCH_IDLE_S, remaining))
            except asyncio.TimeoutError:
                return

    def _take_batch(self) -> list[_Request]:
        """Take the next batch off the queue, oldest request first.

        Prompts are padded on the left to the longest one, and every row keeps
        stepping until the largest budget is spent, finished rows included. So
        a batch spans its longest prompt plus its largest budget, and a request
        joins only if that still fits the context window: past it, a model with
        fixed positions (GPT-2) fails the whole batch, though each request fits
        alone. A request left out keeps its place at the front of the queue.
        """
        loop = asyncio.get_running_loop()
        batch: list[_Request] = []
        left_out: list[_Request] = []
        width = budget = 0
        while self._queue and len(batch) < self._cap():
            request = self._queue.popleft()
            # Skip callers that gave up while queued, and any a closed event loop left behind.
            if request.future.done() or request.future.get_loop() is not loop:
                continue
            wider, longer = max(width, len(request.ids)), max(budget, request.max_new)
            if batch and wider + longer > self.config.max_context_length:
                left_out.append(request)
                continue
            batch.append(request)
            width, budget = wider, longer
        self._queue.extendleft(reversed(left_out))
        return batch

    async def _decode(self, requests: list[_Request]) -> None:
        """Run one batch on a worker thread and hand each caller its own result."""
        batch = _Batch(requests)
        for index, request in enumerate(requests):
            request.future.add_done_callback(functools.partial(batch.settled, index))
        self._running.append(batch)
        try:
            results = await asyncio.to_thread(self._run_batch, batch)
        except asyncio.CancelledError:
            batch.halt.set()
            for request in requests:
                request.future.cancel()
            raise
        except _OutOfMemory:
            if len(requests) > 1:
                self._batch_cap = max(1, len(requests) // 2)
                logger.warning(
                    "out of GPU memory decoding %d requests together; retrying %d at a time",
                    len(requests),
                    self._batch_cap,
                )
                self._queue.extendleft(reversed(requests))
                return
            self._fail(
                requests,
                "Out of GPU memory while answering a single request.",
                hint="Lower --max-tokens or backend.max_context_length, close other GPU "
                "programs, or use a smaller model.",
            )
            return
        except Exception as exc:
            self._fail(requests, "Batched generation failed", cause=exc)
            return
        finally:
            if batch in self._running:
                self._running.remove(batch)
        for request, result in zip(requests, results, strict=True):
            if not request.future.done():
                request.future.set_result(result)

    @staticmethod
    def _fail(
        requests: list[_Request],
        message: str,
        *,
        cause: BaseException | None = None,
        hint: str | None = None,
    ) -> None:
        """Give each caller still waiting its own error (one raised object per caller)."""
        if cause is not None:
            message = f"{message}: {type(cause).__name__}: {cause}"
        for request in requests:
            if request.future.done():
                continue
            error = BackendError(message, hint=hint)
            error.__cause__ = cause
            with contextlib.suppress(RuntimeError):  # its event loop may already be closed
                request.future.set_exception(error)

    def _run_batch(self, batch: _Batch) -> list[ChatResult]:
        """Decode a batch with one `model.generate` call (on a worker thread).

        Prompts are padded on the left, so every row's next token lands in the
        same, last column. The attention mask hides the padding, and
        transformers derives each row's positions from that mask, so each row
        is decoded as it would be alone, up to floating-point rounding.
        """
        import torch
        from transformers import LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

        requests = batch.requests
        try:
            model, tokenizer = self.model, self.tokenizer
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            width = max(len(r.ids) for r in requests)
            ids = torch.full((len(requests), width), pad, dtype=torch.long)
            mask = torch.zeros((len(requests), width), dtype=torch.long)
            for row, request in enumerate(requests):
                ids[row, width - len(request.ids) :] = torch.tensor(request.ids, dtype=torch.long)
                mask[row, width - len(request.ids) :] = 1
            device = getattr(model, "device", self._device)
            ids, mask = ids.to(device), mask.to(device)
            budgets = torch.tensor([r.max_new for r in requests], device=device)

            class StopRows(StoppingCriteria):
                """Finish rows that used their own budget or lost their caller."""

                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
                    done = budgets <= input_ids.shape[1] - width
                    if batch.halt.is_set():
                        return torch.ones_like(done)
                    if any(batch.dropped):
                        done |= torch.tensor(batch.dropped, device=done.device)
                    return done

            processors = LogitsProcessorList()
            penalties = [r.repetition_penalty for r in requests]
            if any(p != 1.0 for p in penalties):
                fillers = [r.ids[0] for r in requests]
                processors.append(_row_repetition_penalty(penalties, mask == 0, fillers))

            eos = self._eos_ids()
            out_of_memory = False
            try:
                output = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_new_tokens=max(r.max_new for r in requests),
                    do_sample=False,
                    repetition_penalty=1.0,  # applied per row by `processors` instead
                    pad_token_id=pad,
                    eos_token_id=eos,
                    logits_processor=processors,
                    stopping_criteria=StoppingCriteriaList([StopRows()]),
                )
            except torch.cuda.OutOfMemoryError:
                out_of_memory = True
            if out_of_memory:
                # Outside the except block, whose traceback would keep the
                # failed attempt's tensors, and so their memory, alive.
                torch.cuda.empty_cache()
                raise _OutOfMemory
            rows = output[:, width:].tolist()
            return [
                self._result(r, row, eos, tokenizer)
                for r, row in zip(requests, rows, strict=True)
            ]
        finally:
            # Only now: aclose() waits for this before it releases the model
            # and the tokenizer that _result() decodes with.
            batch.finished.set()

    def _result(
        self, request: _Request, row: list[int], eos: list[int] | None, tokenizer: Any
    ) -> ChatResult:
        """Finish one row the way the streaming path finishes a reply."""
        tokens = row[: request.max_new]
        ends = set(eos or ())
        end = next((i for i, token in enumerate(tokens) if token in ends), None)
        if end is not None:
            # A finished row is padded until the batch ends; the end token
            # itself counts as generated, as it does when streaming.
            tokens = tokens[: end + 1]
        splitter = ReasoningSplitter(start_in_reasoning=request.starts_in_reasoning)
        thought, answer = splitter.push(tokenizer.decode(tokens, skip_special_tokens=True))
        thought_tail, answer_tail = splitter.flush()
        answer, stopped = self.apply_stop_sequences(answer + answer_tail, request.stop)
        truncated = len(tokens) >= request.max_new and end is None
        return ChatResult(
            model=self.config.model,
            message=Message.assistant(answer, reasoning=(thought + thought_tail) or None),
            finish_reason="length" if truncated and not stopped else "stop",
            usage=Usage(prompt_tokens=len(request.ids), completion_tokens=len(tokens)),
            latency_ms=(time.perf_counter() - request.started) * 1000,
        )
