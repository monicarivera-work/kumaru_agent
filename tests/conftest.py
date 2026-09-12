"""Shared offline fixtures for exercising Kumaru without model downloads."""

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kumaru.agent.chat import ChatAgent
from kumaru.backends.echo import EchoBackend
from kumaru.core.config import BackendConfig, KumaruConfig
from kumaru.core.types import GenerationParams
from kumaru.server.app import create_app


@pytest.fixture
def echo_config() -> KumaruConfig:
    """Return a small deterministic config that never touches GPU-backed code."""
    return KumaruConfig(
        backend=BackendConfig(
            name="echo",
            model="echo",
            max_context_length=2048,
            extra={"delay_s": 0},
        ),
        generation=GenerationParams(max_tokens=32, temperature=0.0),
    )


@pytest.fixture
async def loaded_echo_backend(echo_config: KumaruConfig) -> EchoBackend:
    backend = EchoBackend(echo_config.backend)
    await backend.load()
    try:
        yield backend
    finally:
        await backend.aclose()


@pytest.fixture
def chat_agent(loaded_echo_backend: EchoBackend, echo_config: KumaruConfig) -> ChatAgent:
    return ChatAgent(loaded_echo_backend, echo_config)


@pytest.fixture
def test_client(echo_config: KumaruConfig) -> Generator[TestClient, None, None]:
    app = create_app(echo_config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def scratch_dir() -> Generator[Path, None, None]:
    path = Path(__file__).resolve().parent / ".scratch"
    path.mkdir(exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
    try:
        yield path
    finally:
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
        path.rmdir()
