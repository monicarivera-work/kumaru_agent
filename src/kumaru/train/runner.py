"""Dispatch for `kumaru train` — picks a task and prints a useful preflight.

A deliberately thin file. Its only real job is the preflight report: before a
run that may occupy your GPU for hours, it prints what will happen and checks
the things that commonly fail at minute 55 instead of minute 0 (missing CUDA,
missing optional packages, any data file the task will read that does not exist).
"""

from __future__ import annotations

from pathlib import Path

from kumaru.core.errors import ConfigError, install_command
from kumaru.core.logging import get_logger
from kumaru.train.config import TrainConfig, load_train_config

logger = get_logger(__name__)


def _data_files(config: TrainConfig) -> list[tuple[str, str]]:
    """Every data file the selected task will open, as (config field, path) pairs.

    Mirrors the reads in `run_pretrain()` and `run_lora_finetune()` exactly, so a
    missing file fails here instead of after the tokenizer and model load.
    """
    data = config.data
    if config.task == "pretrain":
        if data.text_file:
            return [("data.text_file", data.text_file)]
        return [("data.train_file, used because data.text_file is not set", data.train_file)]
    files = [("data.train_file", data.train_file)]
    if data.eval_file:
        files.append(("data.eval_file", data.eval_file))
    return files


def preflight(config: TrainConfig) -> None:
    """Log the run plan and warn about foreseeable failures."""
    logger.info("=" * 68)
    logger.info("task            %s", config.task)
    logger.info("base model      %s", config.base_model if config.task == "lora" else "(from scratch)")
    logger.info("output          %s", config.output_dir)
    logger.info("epochs          %s", config.optim.epochs)
    logger.info(
        "batch           %d per device x %d accumulation = %d effective",
        config.optim.per_device_batch_size,
        config.optim.gradient_accumulation_steps,
        config.effective_batch_size,
    )
    logger.info("learning rate   %g (%s schedule)", config.optim.learning_rate, config.optim.lr_scheduler)
    if config.task == "lora":
        logger.info(
            "lora            r=%d alpha=%d on %d module type(s)%s",
            config.lora.r, config.lora.alpha, len(config.lora.target_modules),
            " [4-bit base]" if config.load_in_4bit else "",
        )
    logger.info("=" * 68)

    missing = [(field, path) for field, path in _data_files(config) if not Path(path).exists()]
    if missing:
        raise ConfigError(
            "Training data not found: " + "; ".join(f"{path} ({field})" for field, path in missing),
            hint="Paths are resolved relative to the directory you run from.",
        )

    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning(
                "no CUDA device visible — training will run on CPU and be roughly "
                "50-100x slower. Verify with `kumaru info`."
            )
        else:
            props = torch.cuda.get_device_properties(0)
            logger.info(
                "gpu             %s (%.1f GB, bf16=%s)",
                props.name, props.total_memory / 1024**3,
                "yes" if torch.cuda.is_bf16_supported() else "no",
            )
            if config.optim.bf16 and not torch.cuda.is_bf16_supported():
                logger.warning("bf16 requested but unsupported here; set optim.fp16 instead.")
    except ImportError:
        raise ConfigError(
            "PyTorch is not installed.",
            hint=f"Install it with: {install_command('train')}",
        ) from None

    if config.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            raise ConfigError(
                "load_in_4bit requires bitsandbytes.",
                hint=f"Install it with: {install_command('quant')}",
            ) from None


def run_training(config_path: Path, *, resume: str | None = None, dry_run: bool = False) -> Path:
    """Load a training config, preflight it, and run the matching task."""
    config = load_train_config(config_path)
    preflight(config)

    if config.task == "lora":
        from kumaru.train.finetune_lora import run_lora_finetune

        return run_lora_finetune(config, resume=resume, dry_run=dry_run)

    if config.task == "pretrain":
        from kumaru.train.pretrain_nano import run_pretrain

        return run_pretrain(config, resume=resume, dry_run=dry_run)

    raise ConfigError(f"Unknown training task: {config.task}")
