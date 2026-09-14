"""Shared fixtures.

Everything here runs against the ``echo`` backend so the suite needs no
weights, no GPU and no network - a test run must stay under a few seconds or
people stop running it.
"""

from __future__ import annotations

import pytest

from kumaru.agent.chat import ChatAgent
from kumaru.backends.echo import EchoBackend
from kumaru.core.config import KumaruConfig


@pytest.fixture
def config() -> KumaruConfig:
    cfg = KumaruConfig()
    cfg.backend.name = "echo"
    cfg.agent.max_history_turns = 4
    cfg.agent.max_history_chars = 2000
    cfg.server.serve_ui = False
    return cfg


@pytest.fixture
def backend(config: KumaruConfig) -> EchoBackend:
    # delay_s=0 keeps streaming tests instant.
    return EchoBackend(config.backend, delay_s=0.0)


@pytest.fixture
def agent(config: KumaruConfig, backend: EchoBackend) -> ChatAgent:
    return ChatAgent(config, backend=backend)
