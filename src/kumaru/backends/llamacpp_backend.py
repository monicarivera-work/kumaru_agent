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
8 GB VRAM, 7B model          llamacpp      Q4_K_M is ~4.7 GB and fits
No GPU at all                llamacpp      CPU inference is actually usable
24 GB VRAM, 70B model        llamacpp      Q4 + partial offload; slow but works
Many concurrent users        openai_compat Point at vLLM instead
===========================  =========  =====================================

`n_gpu_layers` is the key knob: that many layers run on the GPU, the rest on the
CPU. -1 offloads every layer, and loading fails with an out-of-memory error if
they do not all fit; then lower it until the model loads. Because decoding is
memory-bandwidth bound, every layer you move off the CPU is a direct speedup, so
tuning this number to just below an out-of-memory error is exactly the right
strategy.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from collections.abc import AsyncIterator, Sequence
from typing import Any

from kumaru.backends.base import (
    Backend,
    ReasoningSplitter,
    StopSequenceTracker,
    iterate_in_thread,
)
from kumaru.backends.openai_compat import normalize_finish_reason
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
        # One Llama is one llama.cpp context, which is not thread-safe, and each
        # request generates on its own worker thread (see iterate_in_thread).
        # The server admits several requests at once, so they take turns here.
        self._generate_lock = threading.Lock()

    async def _load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise DependencyMissingError("llama-cpp-python", extra="gguf") from exc

        cfg = self.config
        path = os.path.expanduser(cfg.model_path or cfg.model)

        kwargs: dict[str, Any] = {
            "n_ctx": cfg.max_context_length,
            "n_gpu_layers": cfg.n_gpu_layers,
            "verbose": False,
        }
        if cfg.n_threads:
            kwargs["n_threads"] = cfg.n_threads

        # Loading reads gigabytes from disk (or the network), so it runs on a
        # worker thread: the server keeps answering other requests meanwhile.
        if os.path.exists(path):
            self.llm = await asyncio.to_thread(Llama, model_path=path, **kwargs)
        elif "/" in path:
            # Treat it as a Hugging Face repo id and pull the GGUF from the hub.
            filename = cfg.extra.get("gguf_filename", "*Q4_K_M.gguf")
            logger.info("downloading GGUF %s from %s", filename, path)
            self.llm = await asyncio.to_thread(
                Llama.from_pretrained, repo_id=path, filename=filename, **kwargs
            )
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
            # Everything that touches the model happens under the lock,
            # including closing the stream, so the next request starts clean.
            with self._generate_lock:
                tracker = StopSequenceTracker(params.stop)
                # llama-cpp-python returns thinking inline as <think>...</think>.
                splitter = ReasoningSplitter()
                produced = 0
                prompt_tokens = 0
                finish_reason = "stop"

                # llama.cpp reads the chat template embedded in the GGUF, so the
                # prompt format matches what the model was trained on.
                extra: dict[str, Any] = {}
                if params.min_p is not None:
                    extra["min_p"] = params.min_p
                if params.presence_penalty is not None:
                    extra["presence_penalty"] = params.presence_penalty
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
                    **extra,
                )

                try:
                    for event in stream:
                        if usage := event.get("usage"):
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        for choice in event.get("choices", []):
                            delta = (choice.get("delta") or {}).get("content") or ""
                            if delta:
                                produced += 1
                                thought, text = splitter.push(delta)
                                safe = tracker.push(text) if text else ""
                                if thought or safe:
                                    yield StreamChunk(delta=safe, reasoning=thought)
                            if choice.get("finish_reason"):
                                finish_reason = normalize_finish_reason(choice["finish_reason"])
                        if tracker.stopped:
                            break
                finally:
                    # llama.cpp computes a token only when asked for the next
                    # event, so closing the stream (stop sequence hit, or the
                    # reader went away) ends the generation.
                    getattr(stream, "close", lambda: None)()

                thought, text = splitter.flush()
                tail = (tracker.push(text) if text else "") + tracker.flush()
                if thought or tail:
                    yield StreamChunk(delta=tail, reasoning=thought)
                yield StreamChunk(
                    finish_reason=finish_reason,
                    usage=Usage(
                        prompt_tokens=prompt_tokens or self.count_tokens(messages),
                        completion_tokens=produced,
                    ),
                )

        async with contextlib.aclosing(iterate_in_thread(produce)) as chunks:
            async for chunk in chunks:
                yield chunk
