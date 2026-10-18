"""Protect what reaches the Hugging Face Trainer: a setting it does not read is lost without a word.

`finetune_lora` passes the Trainer only the arguments the installed transformers
accepts, so one codebase runs across releases. The cost: a field renamed in a new
release is dropped with only a debug line, and the run trains without it. That is
how transformers 5.2, which removed `warmup_ratio`, removed every recipe's warmup.
These tests build the real `TrainingArguments` from every checked-in LoRA recipe,
on the CPU: no GPU, no model, no download. The last test covers the peak GPU
memory that `metrics.json` reports.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kumaru.core.config import REPO_ROOT
from kumaru.train.config import OptimConfig, load_train_config
from kumaru.train.finetune_lora import (
    _filter_kwargs,
    _peak_gpu_memory_gb,
    _training_arguments,
    _warmup_kwargs,
)

LORA_RECIPES = sorted(
    path
    for path in (REPO_ROOT / "configs" / "train").glob("*.yaml")
    if load_train_config(path).task == "lora"
)
# Spellings for other transformers releases. The installed release reads the other
# name or, for group_by_length (replaced by train_sampling_strategy), a default that
# does the same thing.
VERSION_SPELLINGS = {"eval_strategy", "evaluation_strategy", "group_by_length"}


def test_the_recipes_are_found() -> None:
    names = {path.stem for path in LORA_RECIPES}
    assert {"lora-small", "lora-qwen", "distill-gsm8k", "distill-gsm8k-gold"} <= names


@pytest.mark.parametrize("recipe", LORA_RECIPES, ids=lambda path: path.stem)
def test_every_setting_of_every_recipe_reaches_the_trainer(recipe: Path, tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    training_args = transformers.TrainingArguments
    config = load_train_config(recipe)

    wanted = _training_arguments(training_args, config, tmp_path, has_eval=True, cuda=False)
    kept = _filter_kwargs(training_args, wanted)
    assert set(wanted) - set(kept) <= VERSION_SPELLINGS

    args = training_args(**kept, use_cpu=True)
    assert config.optim.warmup_ratio > 0
    assert args.get_warmup_steps(1000) == math.ceil(1000 * config.optim.warmup_ratio)
    assert args.learning_rate == config.optim.learning_rate
    assert (args.eval_steps, args.save_steps) == (config.eval_steps, config.save_steps)
    assert args.load_best_model_at_end
    assert args.metric_for_best_model == "eval_loss"


class _Before5_2:
    def __init__(self, warmup_ratio: float = 0.0, warmup_steps: int = 0) -> None: ...


class _From5_2:
    def __init__(self, warmup_steps: float = 0) -> None: ...


@pytest.mark.parametrize(
    ("release", "expected"),
    [(_Before5_2, {"warmup_ratio": 0.05}), (_From5_2, {"warmup_steps": 0.05})],
    ids=["before-transformers-5.2", "transformers-5.2-on"],
)
def test_the_warmup_ratio_goes_by_the_name_each_release_reads(
    release: type, expected: dict[str, float]
) -> None:
    assert _warmup_kwargs(release, 0.05) == expected


@pytest.mark.parametrize("ratio", [-0.01, 1.0, 1.5])
def test_a_warmup_ratio_outside_0_to_1_is_refused(ratio: float) -> None:
    # transformers 5 reads a warmup_steps of 1 or more as whole steps, not a share.
    with pytest.raises(ValidationError, match="warmup_ratio"):
        OptimConfig(warmup_ratio=ratio)


@pytest.mark.parametrize(("cuda", "expected"), [(True, 9.88), (False, None)])
def test_peak_gpu_memory_is_reported_in_gb_and_is_none_without_cuda(
    cuda: bool, expected: float | None
) -> None:
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, max_memory_reserved=lambda: 9_876_543_210)
    )
    assert _peak_gpu_memory_gb(torch) == expected
