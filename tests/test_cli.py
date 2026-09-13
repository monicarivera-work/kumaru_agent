"""CLI surface: exit codes and the plumbing from flags to config."""

from __future__ import annotations

import json

from kumaru.cli import main


def test_backends_lists_echo(capsys) -> None:
    assert main(["backends"]) == 0
    assert "echo" in capsys.readouterr().out


def test_config_prints_json_with_overrides(capsys) -> None:
    assert main(["--backend", "echo", "--model", "demo", "config"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["backend"]["name"] == "echo"
    assert data["backend"]["model"] == "demo"


def test_ask_prints_an_answer(capsys) -> None:
    assert main(["--backend", "echo", "ask", "--no-stream", "hello", "world"]) == 0
    assert "hello world" in capsys.readouterr().out


def test_missing_config_file_exits_nonzero(capsys) -> None:
    assert main(["-c", "/nope/missing.yaml", "config"]) == 1
    assert "ConfigError" in capsys.readouterr().err


def test_shipped_test_config_loads(capsys) -> None:
    assert main(["-c", "configs/test.yaml", "config"]) == 0
    assert json.loads(capsys.readouterr().out)["agent"]["max_tokens"] == 64
