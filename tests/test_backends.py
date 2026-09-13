"""Backend layer: the echo backend, the loader, and the OpenAI-compatible client."""

from __future__ import annotations

import json

import httpx
import pytest

from kumaru.backends import available_backends, load_backend
from kumaru.backends.echo import EchoBackend
from kumaru.backends.openai_compat import OpenAICompatBackend
from kumaru.core.config import BackendConfig
from kumaru.core.errors import BackendError, ModelNotAvailableError, RegistryError
from kumaru.core.types import ChatRequest, Message


def test_echo_generate_mentions_the_question(backend: EchoBackend) -> None:
    result = backend.generate(ChatRequest(messages=[Message.user("ping")]))
    assert "ping" in result.text
    assert result.usage.completion_tokens > 0


def test_echo_stream_reassembles_to_generate(backend: EchoBackend) -> None:
    request = ChatRequest(messages=[Message.user("hello there")])
    chunks = list(backend.stream(request))
    assert chunks[-1].done is True
    streamed = "".join(c.delta for c in chunks)
    assert streamed == backend.generate(request).text


def test_loader_returns_echo() -> None:
    assert isinstance(load_backend(BackendConfig(name="echo")), EchoBackend)


def test_loader_rejects_unknown_backend() -> None:
    with pytest.raises(RegistryError):
        load_backend(BackendConfig(name="nope"))


def test_available_backends_lists_the_built_ins() -> None:
    assert {"echo", "openai_compat", "transformers", "llamacpp"} <= set(
        available_backends()
    )


def _client(handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://test/v1"
    )


def test_openai_compat_requires_a_model() -> None:
    with pytest.raises(ModelNotAvailableError):
        OpenAICompatBackend(BackendConfig(name="openai_compat", model=""))


def test_openai_compat_generate_parses_the_reply() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["model"] == "m"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    backend = OpenAICompatBackend(
        BackendConfig(name="openai_compat", model="m"), client=_client(handler)
    )
    result = backend.generate(ChatRequest(messages=[Message.user("hi")]))
    assert result.text == "hello"
    assert result.usage.total_tokens == 7


def test_openai_compat_stream_parses_sse() -> None:
    body = (
        'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    backend = OpenAICompatBackend(
        BackendConfig(name="openai_compat", model="m"), client=_client(handler)
    )
    chunks = list(backend.stream(ChatRequest(messages=[Message.user("hi")])))
    assert "".join(c.delta for c in chunks) == "hello"
    assert chunks[-1].done is True


def test_openai_compat_maps_404_to_model_not_available() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no such model"})

    backend = OpenAICompatBackend(
        BackendConfig(name="openai_compat", model="m"), client=_client(handler)
    )
    with pytest.raises(ModelNotAvailableError):
        backend.generate(ChatRequest(messages=[Message.user("hi")]))


def test_openai_compat_maps_connection_failure_to_backend_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    backend = OpenAICompatBackend(
        BackendConfig(name="openai_compat", model="m"), client=_client(handler)
    )
    with pytest.raises(BackendError):
        backend.generate(ChatRequest(messages=[Message.user("hi")]))
