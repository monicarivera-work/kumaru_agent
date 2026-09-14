"""Run a quantised GGUF model on the CPU (or partly on the GPU).

When to choose this backend
---------------------------
You want a real model on a laptop with no CUDA. A 4-bit GGUF of a 7-8B model
needs roughly 5 GB of RAM and answers at a usable speed on modern CPUs, which
makes this the most reliable "works on my machine" option.

``gpu_layers`` offloads the first N transformer layers to the GPU; raise it
until you run out of VRAM, then back off by a few.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kumaru.backends.base import Backend, GenerationResult
from kumaru.core.config import BackendConfig
from kumaru.core.errors import BackendError, ModelNotAvailableError
from kumaru.core.logging import get_logger
from kumaru.core.types import ChatRequest, StreamChunk, Usage

log = get_logger("backends.llamacpp")


class LlamaCppBackend(Backend):
    """In-process generation via ``llama-cpp-python``."""

    name = "llamacpp"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        if not config.model:
            raise ModelNotAvailableError(
                "backend.model must be the path to a .gguf file",
                hint="download one from Hugging Face into ./models/",
            )
        model_path = Path(config.model).expanduser()
        if not model_path.is_file():
            raise ModelNotAvailableError(
                f"GGUF file not found: {model_path}",
                hint="check backend.model in your config",
            )
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ModelNotAvailableError(
                f"llamacpp backend requires llama-cpp-python: {exc}",
                hint="pip install -e '.[llamacpp]'",
            ) from exc

        self._lock = threading.Lock()  # llama.cpp contexts are not thread-safe
        log.info("loading GGUF", extra={"model": str(model_path)})
        try:
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=config.context_length,
                n_threads=config.n_threads or (os.cpu_count() or 4),
                n_gpu_layers=config.gpu_layers,
                verbose=False,
                **config.extra,
            )
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ModelNotAvailableError(
                f"could not load {model_path}: {exc}",
                hint="not enough RAM, or the file is not a valid GGUF",
            ) from exc

    def _kwargs(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        return {
            "messages": request.as_dicts(),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop": request.stop or None,
            "stream": stream,
        }

    @staticmethod
    def _usage(data: dict[str, Any]) -> Usage:
        usage = data.get("usage") or {}
        return Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )

    def generate(self, request: ChatRequest) -> GenerationResult:
        try:
            with self._lock:
                data = self.llm.create_chat_completion(
                    **self._kwargs(request, stream=False)
                )
        except Exception as exc:  # pragma: no cover - depends on environment
            raise BackendError(f"generation failed: {exc}") from exc
        text = (data["choices"][0].get("message") or {}).get("content") or ""
        return GenerationResult(text=text, usage=self._usage(data))

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        try:
            with self._lock:
                completion = 0
                for event in self.llm.create_chat_completion(
                    **self._kwargs(request, stream=True)
                ):
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    if delta:
                        completion += 1
                        yield StreamChunk(delta=delta)
        except Exception as exc:  # pragma: no cover - depends on environment
            raise BackendError(f"generation failed: {exc}") from exc
        yield StreamChunk(done=True, usage=Usage(completion_tokens=completion))

    def health(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "model": Path(self.config.model).name,
            "context_length": self.config.context_length,
            "gpu_layers": self.config.gpu_layers,
            "ready": True,
        }

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        if llm is not None and hasattr(llm, "close"):
            llm.close()
