"""Typed schemas for training runs.

Why training gets its own config tree
-------------------------------------
Inference config answers "how do I serve this?"; training config answers "what
experiment am I running?". They have different lifetimes: a serving config is
edited constantly, a training config is *frozen and archived* — it is the
primary record of how a checkpoint came to exist.

Every field is validated and every run dumps its resolved config into the output
directory. Six months later, "what produced this adapter?" has a precise answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from kumaru.core.errors import ConfigError


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Base):
    """Where the training data lives and how it is shaped."""

    # JSONL, one object per line: {"messages": [{"role": ..., "content": ...}]}
    train_file: str
    eval_file: str | None = None
    # Fraction of train_file held out when eval_file is absent. Never skip this:
    # training loss alone cannot tell you whether you are memorizing.
    eval_split: float = Field(default=0.05, ge=0.0, lt=0.5)

    # Sequences longer than this are dropped (not truncated). Truncating an SFT
    # example mid-answer teaches the model to stop mid-sentence. LoRA only:
    # pretraining packs windows of `model.max_seq_length` and never reads this.
    max_seq_length: int = 2048

    # Compute loss only on assistant turns. Almost always correct for SFT: you
    # want the model to learn to *produce* answers, not to reproduce questions.
    train_on_completions_only: bool = True

    # Raw-text corpus fields, used only by the pretrain task.
    text_file: str | None = None
    tokenizer: str = "Qwen/Qwen2.5-1.5B-Instruct"


class LoraConfig(_Base):
    """Low-Rank Adaptation hyperparameters.

    The idea: freeze W and learn a low-rank update, W + (alpha/r)·B·A, where A is
    r x d and B is d x r. A 7B model has ~7e9 trainable parameters; at r=16 a LoRA
    has ~4e7 — about 0.5%. Optimizer state (Adam keeps two moments per trainable
    parameter) shrinks by the same factor, which is precisely why fine-tuning
    fits on one consumer GPU at all.
    """

    # Rank. The capacity knob. 8-16 for style and format; 32-64 to teach genuinely
    # new domain behaviour. Higher r overfits faster on small datasets.
    r: int = 16
    # Scaling factor. The update is multiplied by alpha/r, so the convention
    # alpha = 2*r keeps the effective learning rate stable as you change r.
    alpha: int = 32
    dropout: float = 0.05
    # Which projections get adapters. Attention-only (q,k,v,o) is the original
    # paper's recipe; including the MLP projections (gate,up,down) is measurably
    # better and is now standard practice.
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: Literal["none", "all", "lora_only"] = "none"


class OptimConfig(_Base):
    """Optimization schedule."""

    epochs: float = 3.0
    # Per-step batch size on one device. On a 24 GB GPU with a 7B model in 4-bit
    # and 2k sequences, 1-2 is the realistic ceiling.
    per_device_batch_size: int = 1
    # Effective batch = per_device x accumulation x num_gpus. Accumulation buys
    # a large effective batch with a small memory footprint by summing gradients
    # over several forward passes before stepping. This is the single most
    # important lever for training big models on small GPUs.
    gradient_accumulation_steps: int = 16

    # 1e-4 to 3e-4 for LoRA (the adapter starts at zero, so it tolerates a much
    # higher LR than full fine-tuning, which wants 1e-5 to 5e-5).
    learning_rate: float = 2e-4
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Recompute activations instead of storing them: ~30% slower, ~40% less
    # memory. Turn it on the moment you hit OOM.
    gradient_checkpointing: bool = True
    # paged_adamw_8bit keeps optimizer state in 8-bit and pages it to CPU under
    # pressure; requires bitsandbytes.
    optimizer: str = "adamw_torch"
    bf16: bool = True
    fp16: bool = False
    seed: int = 42


class NanoModelConfig(_Base):
    """Architecture for the from-scratch model.

    The defaults describe a ~101M-parameter model that trains on one RTX 3090.
    Note how that total splits: with Qwen's ~152k-token vocabulary and
    `d_model=512`, the embedding table alone is ~78M parameters and the eight
    transformer blocks are only ~24M. **At small scale the vocabulary dominates
    the model.** That is why `tie_embeddings` defaults to true — untying it
    would add another 78M parameters that do almost no reasoning work.

    The ratio inverts as you scale: at 7B with the same vocabulary, embeddings
    are ~4% of parameters and the blocks carry the rest.
    """

    vocab_size: int = 0  # 0 = inherit from the tokenizer
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4  # < n_heads enables grouped-query attention
    d_model: int = 512
    d_ff: int = 1408  # ~2.75 x d_model, the usual SwiGLU ratio
    max_seq_length: int = 512  # packed window and max context; the length pretraining uses
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True  # share input/output embeddings; saves parameters


class TrainConfig(_Base):
    """One training run, fully described."""

    task: Literal["lora", "pretrain"] = "lora"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "runs/untitled"
    run_name: str | None = None

    data: DataConfig
    lora: LoraConfig = Field(default_factory=LoraConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    model: NanoModelConfig = Field(default_factory=NanoModelConfig)

    # Quantize the frozen base to 4-bit while training the adapter in bf16.
    # This is QLoRA, and it is what makes 7B fine-tuning fit comfortably in 24 GB.
    load_in_4bit: bool = False
    trust_remote_code: bool = False

    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    report_to: list[str] = Field(default_factory=list)  # e.g. ["tensorboard"]

    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def effective_batch_size(self) -> int:
        return self.optim.per_device_batch_size * self.optim.gradient_accumulation_steps

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )


def load_train_config(path: str | Path) -> TrainConfig:
    """Parse and validate a training YAML."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(
            f"Training config not found: {p}",
            hint="Start from one of the examples in configs/train/",
        )
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    try:
        return TrainConfig(**raw)
    except Exception as exc:
        raise ConfigError(f"Invalid training config {p}: {exc}") from exc
