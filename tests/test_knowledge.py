"""Offline tests for coding guidance and explicitly imported reference packs."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from rich.console import Console
from typer.testing import CliRunner

from kumaru.agent.chat import ChatAgent
from kumaru.agent.knowledge import (
    EXPERTISE,
    MAX_FILE_BYTES,
    REFERENCE_HEADER,
    KnowledgePack,
    load_pack,
    read_local_text,
    reference_context,
    save_pack,
)
from kumaru.agent.prompt import build_system_prompt
from kumaru.backends.echo import EchoBackend
from kumaru.cli import _chat_loop, app
from kumaru.core.config import AgentConfig, BackendConfig, KumaruConfig, load_config
from kumaru.core.errors import ConfigError, ContextLengthExceededError
from kumaru.core.types import Message
from kumaru.eval.benchmarks.runner import BenchSettings, run_benchmarks
from kumaru.eval.harness import run_eval
from kumaru.server.app import AppState, create_app


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


@pytest.mark.parametrize(
    ("text", "detail"),
    [
        ("{}", "topic: Field required"),
        ("not JSON", "Invalid JSON: expected"),
        ('{"topic":"x","unexpected":"PRIVATE-VALUE"}', "unexpected: Extra inputs are not permitted"),
        (
            '{"topic":"x","version":"1","source":"s","content":" \\n "}',
            "content: String should contain non-whitespace text",
        ),
        ('{"a":1,"b":2,"c":3,"d":4,"e":5,"f":6}', "; and 5 more"),
    ],
    ids=["missing", "syntax", "extra", "blank", "many"],
)
def test_invalid_pack_is_rejected_with_the_reason(tmp_path: Path, text: str, detail: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid knowledge") as caught:
        load_pack(path)
    # The CLI and the server show only the message and hint, and the server's
    # /healthz is unauthenticated: name the problem, never echo a value.
    assert detail in caught.value.message
    assert "PRIVATE-VALUE" not in str(caught.value)
    assert "kumaru learn" in (caught.value.hint or "")


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


def test_reference_budget_is_an_exact_prefix_at_every_limit() -> None:
    packs = [pack("x" * 40), pack("  indented\n"), pack("z")]
    label = "Topic: Example language; version: 1.0; source: official.txt\n"
    whole = REFERENCE_HEADER + "\n\n".join(label + p.content for p in packs)
    assert reference_context(packs, len(whole) + 10) == whole
    for limit in range(1, len(whole) + 1):
        assert reference_context(packs, limit) == whole[:limit]


def test_reference_budget_stops_reading_packs_once_full() -> None:
    # Runs on every request: the work must not grow with the size or number of packs.
    class Unread:
        def __getattr__(self, name: str) -> str:
            raise AssertionError(f"read .{name} of a pack past the budget")

    packs = [pack("x" * 5000), *[Unread()] * 1000]
    assert len(reference_context(packs, 1000)) == 1000  # type: ignore[arg-type]


def test_learned_content_keeps_its_whitespace_and_labels_are_trimmed(tmp_path: Path) -> None:
    text = "\n    def area(r):\r\n        return 3.14159 * r * r\n\n"
    source = tmp_path / "reference.py"
    source.write_bytes(text.encode("utf-8"))
    output = tmp_path / "knowledge.json"
    result = CliRunner().invoke(
        app,
        ["learn", "  Python ", "--version", " 3.12\n", "--source", str(source),
         "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    learned = load_pack(output)
    assert (learned.topic, learned.version, learned.source) == ("Python", "3.12", "reference.py")
    assert learned.content == text
    assert text in reference_context([learned], 24_000)
    with pytest.raises(ValidationError, match="topic"):
        KnowledgePack(topic=" \t", version="1.0", source="s", content="x")


def test_learn_cli_says_why_a_reference_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "reference.txt"
    source.write_text(" \n\t\n", encoding="utf-8")
    output = tmp_path / "knowledge.json"
    result = CliRunner().invoke(
        app,
        ["learn", "Topic", "--version", "1.0", "--source", str(source), "--output", str(output)],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ConfigError)
    assert "content: String should contain non-whitespace text" in result.exception.message
    assert not output.exists()


@pytest.mark.asyncio
async def test_a_bad_pack_fails_every_entry_point_before_the_model_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[str] = []

    async def record_load(self: EchoBackend) -> None:
        loads.append(self.name)

    monkeypatch.setattr(EchoBackend, "_load", record_load)
    bad = tmp_path / "pack.json"
    bad.write_text('{"topic": "x"}', encoding="utf-8")
    cfg = KumaruConfig(
        backend=BackendConfig(name="echo", model="echo", extra={"delay_s": 0}),
        agent=AgentConfig(knowledge_files=[str(bad)]),
    )
    golden = tmp_path / "golden.jsonl"
    golden.write_text('{"id": "hello", "prompt": "hello", "answer": "hello"}\n', encoding="utf-8")
    reason = "Invalid knowledge pack .*version: Field required"

    with pytest.raises(ConfigError, match=reason):
        await _chat_loop(cfg, "hello", {})
    assert loads == [], "kumaru chat loaded the model first"
    with pytest.raises(ConfigError, match=reason):
        await run_eval(cfg)
    assert loads == [], "kumaru eval loaded the model first"
    with pytest.raises(ConfigError, match=reason):
        await run_benchmarks(
            cfg,
            [str(golden)],
            BenchSettings(out_dir=tmp_path / "bench"),
            console=Console(file=io.StringIO()),
            show_progress=False,
        )
    assert loads == [], "kumaru bench run loaded the model first"
    assert not (tmp_path / "bench").exists()
    state = AppState(cfg)
    await state._load()
    assert loads == [], "the server loaded the model first"
    assert state.backend is None
    assert not state.ready
    assert "version: Field required" in (state.load_error or "")


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
