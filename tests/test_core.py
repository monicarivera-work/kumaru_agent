"""Core layer: types, errors, registry, config."""

from __future__ import annotations

import pytest

from kumaru.core.config import KumaruConfig, apply_env_overrides, load_config
from kumaru.core.errors import ConfigError, RegistryError
from kumaru.core.registry import Registry
from kumaru.core.types import ChatRequest, Message, Role, Usage


def test_message_normalises_string_role() -> None:
    assert Message("user", "hi").role is Role.USER
    assert Message.system("s").to_dict() == {"role": "system", "content": "s"}


def test_message_round_trips_through_dict() -> None:
    original = Message.assistant("hello")
    assert Message.from_dict(original.to_dict()) == original


def test_usage_totals() -> None:
    assert Usage(3, 4).total_tokens == 7
    assert Usage(3, 4).to_dict()["total_tokens"] == 7


def test_chat_request_wire_format() -> None:
    request = ChatRequest(messages=[Message.user("a"), Message.assistant("b")])
    assert request.as_dicts() == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_registry_registers_and_creates() -> None:
    registry: Registry[str] = Registry("thing")
    registry.register("upper", str.upper)
    assert "upper" in registry
    assert registry.create("upper", "abc") == "ABC"
    assert registry.names() == ["upper"]


def test_registry_rejects_duplicates_and_unknowns() -> None:
    registry: Registry[str] = Registry("thing")
    registry.register("a", str)
    with pytest.raises(RegistryError):
        registry.register("a", str)
    with pytest.raises(RegistryError):
        registry.get("missing")


def test_defaults_are_valid() -> None:
    KumaruConfig().validate()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: setattr(c.agent, "max_tokens", 0),
        lambda c: setattr(c.agent, "temperature", 9.0),
        lambda c: setattr(c.agent, "top_p", 0.0),
        lambda c: setattr(c.server, "port", 0),
        lambda c: setattr(c.backend, "name", ""),
    ],
)
def test_validation_rejects_bad_values(mutate) -> None:
    config = KumaruConfig()
    mutate(config)
    with pytest.raises(ConfigError):
        config.validate()


def test_env_overrides_are_type_coerced() -> None:
    config = apply_env_overrides(
        KumaruConfig(),
        {"KUMARU_PORT": "9001", "KUMARU_TEMPERATURE": "0.1", "KUMARU_BACKEND": "echo"},
    )
    assert config.server.port == 9001
    assert config.agent.temperature == pytest.approx(0.1)
    assert config.backend.name == "echo"


def test_env_override_with_bad_type_names_the_variable() -> None:
    with pytest.raises(ConfigError, match="KUMARU_PORT"):
        apply_env_overrides(KumaruConfig(), {"KUMARU_PORT": "not-a-number"})


def test_load_config_reads_yaml(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("backend:\n  name: echo\nagent:\n  max_tokens: 32\n")
    config = load_config(path, env={})
    assert config.agent.max_tokens == 32


def test_load_config_rejects_unknown_keys(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("agent:\n  nonsense: 1\n")
    with pytest.raises(ConfigError, match="nonsense"):
        load_config(path, env={})


def test_load_config_missing_file() -> None:
    with pytest.raises(ConfigError):
        load_config("/nope/does-not-exist.yaml", env={})


def test_shipped_configs_load() -> None:
    from pathlib import Path

    for path in sorted(Path("configs").glob("*.yaml")):
        load_config(path, env={}).validate()
