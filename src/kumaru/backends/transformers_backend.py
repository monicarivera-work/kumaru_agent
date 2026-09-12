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

import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

from kumaru.backends.base import Backend, StopSequenceTracker, iterate_in_thread
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
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise DependencyMissingError("bitsandbytes", extra="quant") from exc

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
                hint="Train one with `kumaru train --config configs/train/lora-qwen.yaml`",
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

    def _render_prompt(self, messages: Sequence[Message]) -> str:
        """Turn structured messages into the exact string the model expects.

        Chat models are trained with specific control tokens (Qwen uses
        `<|im_start|>`, Llama uses `<|start_header_id|>`, ...). Getting these
        wrong is the most common cause of "the model works but its answers are
        subtly bad": it is being asked a question in a format it never saw
        during training. `apply_chat_template` reads the format out of the
        checkpoint, so it is always right for whatever model you loaded.
        """
        payload = [m.to_openai() for m in messages]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                payload, tokenize=False, add_generation_prompt=True
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
        from transformers import TextIteratorStreamer

        tokenizer, model = self.tokenizer, self.model
        prompt = self._render_prompt(messages)

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

            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600
            )
            kwargs: dict[str, Any] = {
                **encoded,
                "streamer": streamer,
                "max_new_tokens": max_new,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
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
            else:
                # temperature == 0 means greedy. Passing sampling knobs alongside
                # do_sample=False makes transformers emit warnings and ignore them.
                kwargs["do_sample"] = False

            # `generate` blocks until done, so it runs on its own thread while we
            # drain the streamer queue here.
            worker = threading.Thread(target=model.generate, kwargs=kwargs, daemon=True)
            worker.start()

            tracker = StopSequenceTracker(params.stop)
            produced = 0
            try:
                for piece in streamer:
                    if not piece:
                        continue
                    produced += 1
                    safe = tracker.push(piece)
                    if safe:
                        yield StreamChunk(delta=safe)
                    if tracker.stopped:
                        break
                tail = tracker.flush()
                if tail:
                    yield StreamChunk(delta=tail)
            finally:
                worker.join(timeout=1.0)

            reason = "stop" if (tracker.stopped or produced < max_new) else "length"
            yield StreamChunk(
                finish_reason=reason,
                usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=produced),
            )

        async for chunk in iterate_in_thread(produce):
            yield chunk
