"""GGUF inference via `llama.cpp` — the low-VRAM and CPU-only path.

When to reach for this
----------------------
`transformers` wants the whole model in VRAM in 16-bit. llama.cpp instead uses
**GGUF**, a format with aggressive k-quant schemes (Q4_K_M, Q5_K_M, Q8_0) and
transparent CPU/GPU layer splitting. Concretely:

===========================  =========  =====================================
Situation                    Backend    Reason
===========================  =========  =====================================
24 GB VRAM, 7B model         transformers  Fits in bf16; best quality
8 GB VRAM, 7B model          llamacpp      Q4_K_M is ~4.4 GB and fits
No GPU at all                llamacpp      CPU inference is actually usable
24 GB VRAM, 70B model        llamacpp      Q4 + partial offload; slow but works
Many concurrent users        openai_compat Point at vLLM instead
===========================  =========  =====================================

`n_gpu_layers` is the key knob: layers above the cut run on the GPU, the rest on
the CPU. Setting it to -1 offloads as many as will fit. Because decoding is
memory-bandwidth bound, every layer you move off the CPU is a direct speedup, so
tuning this number to just below an out-of-memory error is exactly the right
strategy.
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


@BACKENDS.register("llamacpp", aliases=("gguf",))
class LlamaCppBackend(Backend):
    """Run a quantized GGUF checkpoint with llama.cpp."""

    name = "llamacpp"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self.llm: Any = None

    async def _load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise DependencyMissingError("llama-cpp-python", extra="gguf") from exc

        cfg = self.config
        path = cfg.model_path or cfg.model

        kwargs: dict[str, Any] = {
            "n_ctx": cfg.max_context_length,
            "n_gpu_layers": cfg.n_gpu_layers,
            "verbose": False,
        }
        if cfg.n_threads:
            kwargs["n_threads"] = cfg.n_threads

        if os.path.exists(path):
            self.llm = Llama(model_path=path, **kwargs)
        elif "/" in path:
            # Treat it as a Hugging Face repo id and pull the GGUF from the hub.
            filename = cfg.extra.get("gguf_filename", "*Q4_K_M.gguf")
            logger.info("downloading GGUF %s from %s", filename, path)
            self.llm = Llama.from_pretrained(repo_id=path, filename=filename, **kwargs)
        else:
            raise BackendError(
                f"GGUF model not found: {path}",
                hint=(
                    "Set backend.model_path to a local .gguf file, or set backend.model "
                    "to a Hugging Face repo id such as 'Qwen/Qwen2.5-7B-Instruct-GGUF'."
                ),
            )

    async def aclose(self) -> None:
        self.llm = None
        self._loaded = False

    def info(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            model_id=self.config.model_path or self.config.model,
            device="cuda+cpu" if self.config.n_gpu_layers != 0 else "cpu",
            dtype="gguf",
            context_length=self.config.max_context_length,
            supports_streaming=True,
            quantization="gguf",
            extra={"n_gpu_layers": self.config.n_gpu_layers},
        )

    def count_tokens(self, messages: Sequence[Message]) -> int:
        if self.llm is None:
            return super().count_tokens(messages)
        try:
            blob = "\n".join(f"{m.role.value}: {m.content}" for m in messages)
            return len(self.llm.tokenize(blob.encode("utf-8"))) + 8
        except Exception:  # pragma: no cover
            return super().count_tokens(messages)

    async def stream(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> AsyncIterator[StreamChunk]:
        if not self._loaded:
            await self.load()

        llm = self.llm
        payload = [m.to_openai() for m in messages]

        def produce():
            tracker = StopSequenceTracker(params.stop)
            produced = 0
            prompt_tokens = 0
            finish_reason = "stop"

            # llama.cpp reads the chat template embedded in the GGUF, so the
            # prompt format matches what the model was trained on.
            stream = llm.create_chat_completion(
                messages=payload,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k or 0,
                repeat_penalty=params.repetition_penalty,
                stop=list(params.stop) or None,
                seed=params.seed if params.seed is not None else -1,
                stream=True,
            )

            for event in stream:
                if usage := event.get("usage"):
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                for choice in event.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content") or ""
                    if delta:
                        produced += 1
                        safe = tracker.push(delta)
                        if safe:
                            yield StreamChunk(delta=safe)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                if tracker.stopped:
                    break

            tail = tracker.flush()
            if tail:
                yield StreamChunk(delta=tail)
            yield StreamChunk(
                finish_reason=finish_reason,
                usage=Usage(
                    prompt_tokens=prompt_tokens or self.count_tokens(messages),
                    completion_tokens=produced,
                ),
            )

        async for chunk in iterate_in_thread(produce):
            yield chunk
