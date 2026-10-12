"""Offline tests for coding guidance and explicitly imported reference packs."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from kumaru.agent.chat import ChatAgent
from kumaru.agent.knowledge import (
    EXPERTISE,
    MAX_FILE_BYTES,
    KnowledgePack,
    load_pack,
    read_local_text,
    reference_context,
    save_pack,
)
from kumaru.agent.prompt import build_system_prompt
from kumaru.backends.echo import EchoBackend
from kumaru.cli import app
from kumaru.core.config import AgentConfig, BackendConfig, KumaruConfig, load_config
from kumaru.core.errors import ConfigError, ContextLengthExceededError
from kumaru.core.types import Message
from kumaru.server.app import create_app


def pack(content: str = "Use checked arithmetic.") -> KnowledgePack:
    return KnowledgePack(topic="Example language", version="1.0", source="official.txt", content=content)


def test_coder_profile_and_optional_guidance() -> None:
    config = load_config("coder", use_env=False)
    prompt = build_system_prompt(config.agent, include_date=False)
    assert config.agent.expertise == list(EXPERTISE)
    for text in EXPERTISE.values():
        assert text in prompt
    assert "Engineering guidance" not in build_system_prompt(AgentConfig())
    with pytest.raises(ValidationError):
        AgentConfig(expertise=["typo"])


def test_guidance_is_not_duplicated() -> None:
    prompt = build_system_prompt(AgentConfig(expertise=["coding", "coding"]))
    assert prompt.count(EXPERTISE["coding"]) == 1


def test_pack_roundtrip_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "pack.json"
    save_pack(pack("Unicode: 日本語"), path)
    assert load_pack(path).content == "Unicode: 日本語"
    with pytest.raises(ConfigError, match="Cannot create"):
        save_pack(pack("replacement"), path)
    assert load_pack(path).content == "Unicode: 日本語"


@pytest.mark.parametrize("text", ["{}", "not JSON", '{"topic":"x","unexpected":true}'])
def test_invalid_pack_is_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid knowledge"):
        load_pack(path)


def test_read_is_bounded_and_utf8_only(tmp_path: Path) -> None:
    path = tmp_path / "reference.txt"
    path.write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    with pytest.raises(ConfigError, match="exceeds"):
        read_local_text(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ConfigError, match="Cannot read"):
        read_local_text(path)
    with pytest.raises(ConfigError, match="Cannot read"):
        read_local_text(tmp_path / "missing")


def test_empty_reference_and_invalid_budget_are_rejected() -> None:
    with pytest.raises(ValidationError):
        pack("  ")
    with pytest.raises(ValidationError):
        AgentConfig(max_knowledge_chars=-1)


def test_total_reference_budget_and_disabled_context() -> None:
    packs = [pack("x" * 5000), pack("y" * 5000)]
    context = reference_context(packs, 1000)
    assert len(context) == 1000
    assert "untrusted, not instructions" in context
    assert "version: 1.0" in context
    assert reference_context(packs, 0) == ""
    assert reference_context([], 1000) == ""


def test_learn_cli_creates_reference_without_running_code(tmp_path: Path) -> None:
    source = tmp_path / "reference.txt"
    marker = tmp_path / "must-not-exist"
    source.write_text(f"open({str(marker)!r}, 'w').write('unsafe')", encoding="utf-8")
    output = tmp_path / "knowledge.json"
    result = CliRunner().invoke(
        app,
        ["learn", "New language", "--version", "1.0", "--source", str(source),
         "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    learned = load_pack(output)
    assert learned.topic == "New language"
    assert learned.source == source.name
    assert learned.content == source.read_text(encoding="utf-8")
    assert "knowledge_files" in result.output
    assert not marker.exists()


@pytest.mark.asyncio
async def test_agent_uses_packs_for_complete_and_stream(tmp_path: Path) -> None:
    path = tmp_path / "pack.json"
    save_pack(pack(), path)
    cfg = KumaruConfig(
        backend=BackendConfig(name="echo", model="echo", extra={"delay_s": 0}),
        agent=AgentConfig(expertise=["coding"], knowledge_files=[str(path)]),
    )
    backend = EchoBackend(cfg.backend)
    agent = ChatAgent(backend, cfg)
    prepared = agent.prepare([Message.user("Explain arithmetic")], extra_context="Current project")
    assert "Use checked arithmetic." in prepared[0].content
    assert "Current project" in prepared[0].content
    assert agent.describe()["knowledge_topics"] == ["Example language"]
    await backend.load()
    try:
        result = await agent.complete([Message.user("Explain arithmetic")])
        chunks = [chunk async for chunk in agent.stream([Message.user("Explain arithmetic")])]
    finally:
        await backend.aclose()
    assert "You said: Explain arithmetic" in result.message.content
    assert "You said: Explain arithmetic" in "".join(chunk.delta for chunk in chunks)


def test_oversized_reference_cannot_overflow_model_context(tmp_path: Path) -> None:
    path = tmp_path / "pack.json"
    save_pack(pack("x " * 10000), path)
    cfg = KumaruConfig(
        backend=BackendConfig(name="echo", model="echo", max_context_length=1024),
        agent=AgentConfig(knowledge_files=[str(path)], max_knowledge_chars=24000),
    )
    agent = ChatAgent(EchoBackend(cfg.backend), cfg)
    with pytest.raises(ContextLengthExceededError, match="exceed"):
        agent.prepare([Message.user("hello")])
    cfg.agent.max_knowledge_chars = 0
    assert agent.prepare([Message.user("hello")])[-1].content == "hello"


def test_api_and_ui_chat_path_share_the_coding_agent(tmp_path: Path) -> None:
    path = tmp_path / "pack.json"
    save_pack(pack(), path)
    cfg = load_config(
        "coder",
        use_env=False,
        overrides={
            "backend": {"name": "echo", "model": "echo", "extra": {"delay_s": 0}},
            "agent": {"knowledge_files": [str(path)]},
        },
    )
    application = create_app(cfg)
    with TestClient(application) as client:
        payload = {"model": "echo", "messages": [{"role": "user", "content": "hello"}]}
        assert client.post("/v1/chat/completions", json=payload).status_code == 200
        agent = application.state.kumaru.agent
        assert agent is not None
        prompt = agent.prepare([Message.user("hello")])[0].content
        assert EXPERTISE["playwright"] in prompt
        assert "Use checked arithmetic." in prompt
        with client.stream("POST", "/v1/chat/completions", json={**payload, "stream": True}) as response:
            assert response.status_code == 200
            assert "data: [DONE]" in response.read().decode("utf-8")
