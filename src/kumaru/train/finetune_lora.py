"""LoRA / QLoRA supervised fine-tuning.

What this actually does
-----------------------
Freezes the base model and trains a small low-rank update alongside it. For
Qwen2.5-7B at r=16, that is roughly 40M trainable parameters instead of 7.6B —
about 0.5%. The memory arithmetic is what makes it possible on one consumer GPU:

=============================  ===================  ==================
Component                      Full fine-tune       QLoRA (r=16)
=============================  ===================  ==================
Base weights                   15.2 GB (bf16)       4.5 GB (4-bit NF4)
Gradients                      15.2 GB              0.08 GB
Adam state (2 moments, fp32)   60.8 GB              0.32 GB
Activations (2k seq, bs 1)     ~4 GB                ~2.5 GB (checkpointed)
-----------------------------  -------------------  ------------------
**Total**                      **~95 GB**           **~7.5 GB**
=============================  ===================  ==================

That is the whole story: optimizer state is the dominant cost of training, and
it scales with *trainable* parameters, not total ones. LoRA attacks exactly that
term. 95 GB needs an H100 cluster; 7.5 GB fits on your 3090 with room to spare.

What fine-tuning can and cannot do
----------------------------------
It reliably teaches **format, tone, and domain-specific phrasing** — how to
answer. It is a poor and expensive way to inject **facts**; retrieval (RAG) does
that better, cheaper, and updatably. If your goal is "the model should know our
internal docs", build retrieval first and fine-tune second, if at all.
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any

from kumaru.core.errors import DependencyMissingError, KumaruError
from kumaru.core.logging import get_logger
from kumaru.train.config import TrainConfig
from kumaru.train.data import (
    SFTCollator,
    build_sft_dataset,
    load_chat_jsonl,
    split_rows,
)

logger = get_logger(__name__)


class ListDataset:
    """Minimal `torch.utils.data.Dataset` over pre-encoded examples.

    The whole dataset is already tokenized and in memory, so the extra machinery
    of `datasets.Dataset` buys nothing here — and this way the data path stays
    readable end to end.
    """

    def __init__(self, items: list[dict[str, list[int]]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.items[idx]


def _filter_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only the kwargs a given class actually accepts.

    `TrainingArguments` renames fields between releases (`evaluation_strategy` →
    `eval_strategy`, for example). Filtering against the real signature makes
    this trainer work across several transformers versions instead of breaking
    on every upgrade.
    """
    try:
        accepted = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        return kwargs
    dropped = set(kwargs) - accepted
    if dropped:
        logger.debug("dropping unsupported TrainingArguments: %s", sorted(dropped))
    return {k: v for k, v in kwargs.items() if k in accepted}


def _warmup_kwargs(training_args_cls: Any, ratio: float) -> dict[str, float]:
    """The warmup share, under the name this transformers release reads.

    transformers 5.2 removed `warmup_ratio` (deprecated in 5.0, where
    `warmup_steps` began reading a float below 1 as a share of the optimizer
    steps). Passed under the old name, the ratio would be filtered out with only
    a debug line, and the run would start at the full learning rate.
    """
    if "warmup_ratio" in inspect.signature(training_args_cls.__init__).parameters:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": ratio}


def _training_arguments(
    training_args_cls: Any, config: TrainConfig, out_dir: Path, *, has_eval: bool, cuda: bool
) -> dict[str, Any]:
    """Everything the Trainer is told, before `_filter_kwargs` drops what this release lacks."""
    return {
        "output_dir": str(out_dir),
        "run_name": config.run_name or out_dir.name,
        "num_train_epochs": config.optim.epochs,
        "per_device_train_batch_size": config.optim.per_device_batch_size,
        "per_device_eval_batch_size": config.optim.per_device_batch_size,
        "gradient_accumulation_steps": config.optim.gradient_accumulation_steps,
        "learning_rate": config.optim.learning_rate,
        "lr_scheduler_type": config.optim.lr_scheduler,
        **_warmup_kwargs(training_args_cls, config.optim.warmup_ratio),
        "weight_decay": config.optim.weight_decay,
        "max_grad_norm": config.optim.max_grad_norm,
        "gradient_checkpointing": config.optim.gradient_checkpointing,
        "optim": config.optim.optimizer,
        "bf16": config.optim.bf16 and cuda,
        "fp16": config.optim.fp16,
        "seed": config.optim.seed,
        "logging_steps": config.logging_steps,
        "eval_strategy": "steps" if has_eval else "no",
        "evaluation_strategy": "steps" if has_eval else "no",
        "eval_steps": config.eval_steps,
        "save_strategy": "steps",
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": has_eval,
        "metric_for_best_model": "eval_loss" if has_eval else None,
        "greater_is_better": False,
        "report_to": config.report_to or "none",
        "remove_unused_columns": False,  # our collator needs the raw fields
        "dataloader_pin_memory": cuda,
        "group_by_length": False,
    }


def _peak_gpu_memory_gb(torch: Any) -> float | None:
    """The most GPU memory PyTorch reserved during the run, in GB; None without CUDA.

    Reserved, not allocated: PyTorch keeps freed blocks for reuse, so the
    reserved peak is what a game or OBS sharing the GPU could not have.
    nvidia-smi shows a few hundred MB more, the CUDA context.
    """
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_reserved() / 1e9, 2)


def run_lora_finetune(config: TrainConfig, *, resume: str | None = None, dry_run: bool = False) -> Path:
    """Execute a LoRA/QLoRA fine-tuning run. Returns the adapter directory."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise DependencyMissingError("transformers", extra="train") from exc

    try:
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise DependencyMissingError("peft", extra="train") from exc

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Archive the exact config beside the weights. A checkpoint without its
    # recipe is an artifact you cannot reproduce or trust.
    config.save(out_dir / "train_config.yaml")

    # -- tokenizer ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None) is None:
        raise KumaruError(
            f"{config.base_model} has no chat template.",
            hint="Fine-tune an -Instruct checkpoint, or set a chat_template on the tokenizer.",
        )

    # -- data ---------------------------------------------------------------
    rows = load_chat_jsonl(config.data.train_file)
    if config.data.eval_file:
        train_rows, eval_rows = rows, load_chat_jsonl(config.data.eval_file)
    else:
        train_rows, eval_rows = split_rows(rows, config.data.eval_split, config.optim.seed)

    train_items = build_sft_dataset(
        train_rows, tokenizer,
        max_seq_length=config.data.max_seq_length,
        completions_only=config.data.train_on_completions_only,
    )
    eval_items = (
        build_sft_dataset(
            eval_rows, tokenizer,
            max_seq_length=config.data.max_seq_length,
            completions_only=config.data.train_on_completions_only,
        )
        if eval_rows
        else []
    )

    steps_per_epoch = max(1, len(train_items) // max(1, config.effective_batch_size))
    total_steps = int(steps_per_epoch * config.optim.epochs)

    logger.info(
        "plan: %d train / %d eval example(s) | effective batch %d | ~%d optimizer steps",
        len(train_items), len(eval_items), config.effective_batch_size, total_steps,
    )
    if len(train_items) < 50:
        logger.warning(
            "only %d training example(s). LoRA overfits fast on tiny datasets — "
            "prefer 500+, and watch eval loss for divergence.", len(train_items)
        )

    if dry_run:
        _preview(train_items[0], tokenizer, out_dir)
        logger.info("dry run complete — config and data are valid. Nothing was trained.")
        return out_dir

    # -- model --------------------------------------------------------------
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "dtype": torch.bfloat16 if config.optim.bf16 else torch.float32,
    }

    if config.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise DependencyMissingError("bitsandbytes", extra="quant") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.optim.bf16 else torch.float32,
        )

    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}

    logger.info("loading base model %s%s", config.base_model, " (4-bit)" if config.load_in_4bit else "")
    try:
        model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    except TypeError:
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)

    if config.load_in_4bit:
        # Casts layer norms to fp32 and enables input-gradient flow — both are
        # required for gradients to propagate through frozen quantized layers.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.optim.gradient_checkpointing
        )

    model.config.use_cache = False  # incompatible with gradient checkpointing

    peft_config = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "trainable: %s / %s parameters (%.3f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    # -- trainer ------------------------------------------------------------
    settings = _training_arguments(
        TrainingArguments, config, out_dir,
        has_eval=bool(eval_items), cuda=torch.cuda.is_available(),
    )
    args = TrainingArguments(**_filter_kwargs(TrainingArguments, settings))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ListDataset(train_items),
        eval_dataset=ListDataset(eval_items) if eval_items else None,
        data_collator=SFTCollator(tokenizer.pad_token_id),
    )

    started = time.perf_counter()
    logger.info("training started")
    trainer.train(resume_from_checkpoint=resume)
    elapsed = time.perf_counter() - started
    peak_gb = _peak_gpu_memory_gb(torch)

    # -- save ---------------------------------------------------------------
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "train_seconds": round(elapsed, 1),
        "peak_gpu_memory_gb": peak_gb,
        "train_examples": len(train_items),
        "eval_examples": len(eval_items),
        "trainable_parameters": trainable,
        "effective_batch_size": config.effective_batch_size,
        "log_history": trainer.state.log_history[-30:],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("training finished in %.1f min", elapsed / 60)
    if peak_gb is not None:
        logger.info("peak GPU memory: %.1f GB reserved", peak_gb)
    logger.info("adapter saved to %s", adapter_dir)
    logger.info("try it:  kumaru chat --model %s --adapter %s", config.base_model, adapter_dir)
    return adapter_dir


def _preview(item: dict[str, list[int]], tokenizer: Any, out_dir: Path) -> None:
    """Write a human-readable view of one encoded example.

    Reading this file is the fastest way to catch a masking bug. If the
    `SUPERVISED` section is empty, or contains the user's question, your model
    is about to learn the wrong task — and you will only find out hours later
    from the loss curve.
    """
    from kumaru.train.data import IGNORE_INDEX

    supervised = [
        t
        for t, label in zip(item["input_ids"], item["labels"], strict=False)
        if label != IGNORE_INDEX
    ]
    text = (
        "=== FULL SEQUENCE (model input) ===\n"
        f"{tokenizer.decode(item['input_ids'])}\n\n"
        "=== SUPERVISED TOKENS (where loss is computed) ===\n"
        f"{tokenizer.decode(supervised)}\n\n"
        f"tokens: {len(item['input_ids'])} total, {len(supervised)} supervised "
        f"({100 * len(supervised) / max(1, len(item['input_ids'])):.1f}%)\n"
    )
    path = out_dir / "example_preview.txt"
    path.write_text(text, encoding="utf-8")
    logger.info("wrote %s — check that only the assistant's answer is supervised", path)
