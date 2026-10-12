"""Protect the training preflight: every data file a run will read is checked at minute 0.

No GPU, no model, no tokenizer, no download: `torch` is replaced in `sys.modules` by a
stub (or blocked outright), and every path points at the repo's own sample files.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kumaru.core.config import REPO_ROOT
from kumaru.core.errors import ConfigError
from kumaru.train import runner
from kumaru.train.config import DataConfig, TrainConfig
from kumaru.train.runner import preflight

SAMPLES = REPO_ROOT / "data" / "samples"
CHAT = str(SAMPLES / "chat_train.jsonl")
CORPUS = str(SAMPLES / "corpus.txt")
MISSING = str(SAMPLES / "no-such-file.jsonl")
MISSING_TOO = str(SAMPLES / "no-such-file-either.txt")


@pytest.fixture
def torch_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `import torch` fail, so a data error proves preflight stopped before torch."""
    monkeypatch.setitem(sys.modules, "torch", None)


@pytest.fixture
def cpu_only_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in torch that reports no CUDA device, so passing runs touch no GPU."""
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake)


def lora(train_file: str, eval_file: str | None = None) -> TrainConfig:
    return TrainConfig(task="lora", data=DataConfig(train_file=train_file, eval_file=eval_file))


def pretrain(text_file: str | None, train_file: str) -> TrainConfig:
    return TrainConfig(task="pretrain", data=DataConfig(train_file=train_file, text_file=text_file))


def test_lora_missing_eval_file_fails_before_torch(torch_blocked: None) -> None:
    with pytest.raises(ConfigError, match="Training data not found") as excinfo:
        preflight(lora(CHAT, eval_file=MISSING))
    message = excinfo.value.message
    assert f"{MISSING} (data.eval_file)" in message
    assert CHAT not in message
    assert excinfo.value.hint == "Paths are resolved relative to the directory you run from."


@pytest.mark.parametrize("eval_file", [None, ""])
def test_lora_without_eval_file_checks_only_train_file(cpu_only_torch: None, eval_file: str | None) -> None:
    preflight(lora(CHAT, eval_file=eval_file))


def test_lora_reports_every_missing_file_at_once(torch_blocked: None) -> None:
    with pytest.raises(ConfigError, match="Training data not found") as excinfo:
        preflight(lora(MISSING, eval_file=MISSING_TOO))
    message = excinfo.value.message
    assert f"{MISSING} (data.train_file)" in message
    assert f"{MISSING_TOO} (data.eval_file)" in message


@pytest.mark.parametrize("text_file", [None, ""])
def test_pretrain_without_text_file_checks_the_train_file_fallback(
    torch_blocked: None, text_file: str | None
) -> None:
    with pytest.raises(ConfigError, match="Training data not found") as excinfo:
        preflight(pretrain(text_file, train_file=MISSING))
    assert f"{MISSING} (data.train_file, used because data.text_file is not set)" in excinfo.value.message


def test_pretrain_missing_text_file_fails_before_torch(torch_blocked: None) -> None:
    with pytest.raises(ConfigError, match="Training data not found") as excinfo:
        preflight(pretrain(MISSING, train_file=CORPUS))
    assert f"{MISSING} (data.text_file)" in excinfo.value.message


def test_pretrain_with_text_file_ignores_a_missing_train_file(cpu_only_torch: None) -> None:
    preflight(pretrain(CORPUS, train_file=MISSING))


def test_run_training_stops_before_loading_any_task(monkeypatch: pytest.MonkeyPatch, torch_blocked: None) -> None:
    config = lora(CHAT, eval_file=MISSING)
    monkeypatch.setattr(runner, "load_train_config", lambda path: config)
    # If dispatch were reached, these imports would raise ImportError instead of ConfigError.
    monkeypatch.setitem(sys.modules, "kumaru.train.finetune_lora", None)
    monkeypatch.setitem(sys.modules, "kumaru.train.pretrain_nano", None)
    with pytest.raises(ConfigError, match="Training data not found"):
        runner.run_training(Path("unused.yaml"))
