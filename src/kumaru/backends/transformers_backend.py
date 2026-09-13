"""Run a Hugging Face model inside this process.

When to choose this backend
---------------------------
You have a GPU (or patience) and you want no extra server process, or you want
to load your own fine-tuned weights / LoRA adapter straight from disk. It is
the backend that turns "chat" into "chat with *my* model".

Everything here is imported lazily inside ``__init__`` so that merely having
this module on disk never forces a torch install.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

from kumaru.backends.base import Backend, GenerationResult
from kumaru.core.config import BackendConfig
from kumaru.core.errors import BackendError, ModelNotAvailableError
from kumaru.core.logging import get_logger
from kumaru.core.types import ChatRequest, StreamChunk, Usage

log = get_logger("backends.transformers")


class TransformersBackend(Backend):
    """In-process generation via ``transformers``."""

    name = "transformers"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        if not config.model:
            raise ModelNotAvailableError(
                "backend.model must be set for the transformers backend",
                hint="e.g. model: Qwen/Qwen2.5-1.5B-Instruct, or a local path",
            )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ModelNotAvailableError(
                f"transformers backend requires torch + transformers: {exc}",
                hint="pip install -e '.[transformers]'",
            ) from exc

        self._torch = torch
        self._lock = threading.Lock()  # one model, many HTTP requests
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = self._resolve_dtype(torch, config.dtype, device)

        log.info(
            "loading weights",
            extra={"model": config.model, "device": device, "dtype": str(dtype)},
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model,
                torch_dtype=dtype,
                device_map=device if device != "cpu" else None,
                **config.extra,
            )
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ModelNotAvailableError(
                f"could not load '{config.model}': {exc}",
                hint="check the model id/path, your disk space and VRAM",
            ) from exc
        if device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _resolve_dtype(torch: Any, name: str, device: str) -> Any:
        if name and name != "auto":
            resolved = getattr(torch, name, None)
            if resolved is None:
                raise BackendError(f"unknown dtype '{name}'")
            return resolved
        # bfloat16 on modern GPUs, float32 on CPU (fp16 on CPU is slow/unstable).
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def _encode(self, request: ChatRequest) -> Any:
        messages = request.as_dicts()
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Plain fallback for base models without a chat template.
            text = (
                "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                + "\nassistant:"
            )
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def _gen_kwargs(self, request: ChatRequest, inputs: Any) -> dict[str, Any]:
        return {
            **inputs,
            "max_new_tokens": request.max_tokens,
            "do_sample": request.temperature > 0,
            "temperature": max(request.temperature, 1e-5),
            "top_p": request.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def generate(self, request: ChatRequest) -> GenerationResult:
        inputs = self._encode(request)
        prompt_len = int(inputs["input_ids"].shape[-1])
        try:
            with self._lock, self._torch.inference_mode():
                output = self.model.generate(**self._gen_kwargs(request, inputs))
        except Exception as exc:  # pragma: no cover - depends on environment
            raise BackendError(f"generation failed: {exc}") from exc
        new_tokens = output[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            usage=Usage(
                prompt_tokens=prompt_len, completion_tokens=int(new_tokens.shape[-1])
            ),
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        try:
            from transformers import TextIteratorStreamer
        except ImportError:  # pragma: no cover - depends on environment
            yield from super().stream(request)
            return

        inputs = self._encode(request)
        prompt_len = int(inputs["input_ids"].shape[-1])
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = self._gen_kwargs(request, inputs) | {"streamer": streamer}
        error: list[BaseException] = []

        def _run() -> None:
            try:
                with self._lock, self._torch.inference_mode():
                    self.model.generate(**kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised on main thread
                error.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        completion = 0
        for piece in streamer:
            if piece:
                completion += 1
                yield StreamChunk(delta=piece)
        thread.join()
        if error:  # pragma: no cover - depends on environment
            raise BackendError(f"generation failed: {error[0]}") from error[0]
        yield StreamChunk(
            done=True,
            usage=Usage(prompt_tokens=prompt_len, completion_tokens=completion),
        )

    def health(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "model": self.config.model,
            "device": self.device,
            "ready": True,
        }

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            del self.model
        if getattr(self, "_torch", None) is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
