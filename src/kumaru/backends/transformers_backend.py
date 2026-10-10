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
   (see `openai_compat.py` and docs/SCALABILITY.md).
3. **Prompt processing is different.** Prefill is compute bound and highly
   parallel, which is why a 4,000-token prompt is processed far faster than
   4,000 tokens are generated.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
from collections.abc import AsyncIterator, Sequence
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
from kumaru.core.types import BackendInfo, GenerationParams, Message, StreamChunk, Usage

logger = get_logger(__name__)


@BACKENDS.register("transformers", aliases=("hf",))
class TransformersBackend(Backend):
    """Run a Hugging Face causal LM locally, with optional 4/8-bit and LoRA."""

    name = "transformers"
    supports_adapters = True

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self.model: Any = None
        self.tokenizer: Any = None
        self._device: str = "cpu"
        self._dtype_name: str = "unknown"

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
        self, messages: Sequence[Message], params: GenerationParams
    ) -> AsyncIterator[StreamChunk]:
        if not self._loaded:
            await self.load()

        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        tokenizer, model = self.tokenizer, self.model
        prompt = self._render_prompt(messages, thinking=params.thinking)
        # Some templates pre-open the thinking block in the prompt, so the
        # output starts mid-thought and only "</think>" ever appears.
        splitter = ReasoningSplitter(start_in_reasoning=prompt.rstrip().endswith("<think>"))

        def produce():
            import threading

            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            prompt_tokens = int(encoded["input_ids"].shape[-1])

            budget = self.config.max_context_length - prompt_tokens
            if budget <= 0:
                raise BackendError(
                    f"Prompt is {prompt_tokens} tokens but the context window is "
                    f"{self.config.max_context_length}.",
                    hint="Lower agent.max_history_tokens or raise backend.max_context_length.",
                )
            max_new = min(params.max_tokens, budget)

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
                "eos_token_id": tokenizer.eos_token_id,
                "stopping_criteria": StoppingCriteriaList([HaltWhenAsked()]),
            }
            if params.temperature > 0:
                kwargs.update(
                    do_sample=True,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    repetition_penalty=params.repetition_penalty,
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
            eos = tokenizer.eos_token_id
            eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
            truncated = streamer.generated >= max_new and streamer.last_token not in eos_ids
            reason = "length" if truncated and not tracker.stopped else "stop"
            yield StreamChunk(
                finish_reason=reason,
                usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=streamer.generated),
            )

        async with contextlib.aclosing(iterate_in_thread(produce)) as chunks:
            async for chunk in chunks:
                yield chunk
