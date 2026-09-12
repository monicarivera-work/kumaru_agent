"""From-scratch pretraining loop — every mechanism visible, and DDP-ready.

There is no Trainer class here. The optimizer step, the learning-rate schedule,
gradient accumulation, clipping, mixed precision, and checkpointing are all
written out, because these are the mechanisms you need to understand to debug a
run that is not converging.

The same loop scales
--------------------
=================  ==============================================================
1 GPU              ``kumaru train configs/train/pretrain-nano.yaml``
N GPUs, 1 node     ``torchrun --nproc_per_node=8 -m kumaru.train.pretrain_nano cfg.yaml``
N nodes            same command plus ``--nnodes`` / ``--rdzv_endpoint``
=================  ==============================================================

Data-parallel training works because gradients are *linear*: each rank computes
gradients on its own shard, an all-reduce averages them, and every rank applies
the identical update. The result is mathematically equivalent to one giant
batch. That is the entire idea behind scaling to hundreds of GPUs.

Beyond data parallelism — when the *model* stops fitting on one GPU rather than
the batch — the next steps are FSDP/ZeRO (shard optimizer state, gradients, and
parameters across ranks) and tensor parallelism (split individual matrices).
docs/SCALABILITY.md covers when each becomes necessary.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from kumaru.core.errors import DependencyMissingError
from kumaru.core.logging import get_logger
from kumaru.train.config import TrainConfig, load_train_config

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


class DistEnv:
    """Reads the process-group environment `torchrun` sets up.

    Single-GPU runs take the same code path with `world_size == 1`, so there is
    no separate "distributed mode" to maintain or get wrong.
    """

    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        # Only rank 0 logs and writes checkpoints; otherwise 8 processes race to
        # write the same file and interleave their log lines.
        return self.rank == 0

    def setup(self) -> Any:
        import torch

        if self.is_distributed:
            import torch.distributed as dist

            # NCCL on Linux/NVIDIA; gloo elsewhere (Windows has no NCCL).
            backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
            dist.init_process_group(backend=backend)
            torch.cuda.set_device(self.local_rank)
            logger.info("rank %d/%d initialized (%s)", self.rank, self.world_size, backend)
            return torch.device(f"cuda:{self.local_rank}")

        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def teardown(self) -> None:
        if self.is_distributed:
            import torch.distributed as dist

            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------


def lr_at_step(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_ratio: float = 0.1) -> float:
    """Linear warmup, then cosine decay.

    **Warmup** exists because Adam's second-moment estimate is meaningless for
    the first few dozen steps; taking full-size steps then can knock a randomly
    initialized model into a bad region it never recovers from.

    **Cosine decay** spends most of the run at a high, exploratory learning rate
    and anneals smoothly at the end, which reliably beats a step schedule.
    Decaying to ~10% rather than 0 keeps the model learning until the last step.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1 - min_ratio) * cosine)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_pretrain(config: TrainConfig, *, resume: str | None = None, dry_run: bool = False) -> Path:
    """Train a `NanoLM` from random initialization on a packed token corpus."""
    try:
        import torch
        from torch.nn.parallel import DistributedDataParallel
    except ImportError as exc:
        raise DependencyMissingError("torch", extra="torch") from exc

    from transformers import AutoTokenizer

    from kumaru.train.data import PackedTokenDataset, pack_corpus_to_memmap
    from kumaru.train.model_nano import NanoConfig, NanoLM, estimate_parameters

    env = DistEnv()
    device = env.setup()
    out_dir = Path(config.output_dir)
    if env.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        config.save(out_dir / "train_config.yaml")

    torch.manual_seed(config.optim.seed + env.rank)

    # -- data ---------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer)
    tokens_path = out_dir / "tokens.bin"

    if env.is_main and not tokens_path.exists():
        source = config.data.text_file or config.data.train_file
        logger.info("packing corpus %s -> %s", source, tokens_path)
        pack_corpus_to_memmap(source, tokens_path, tokenizer)

    if env.is_distributed:
        import torch.distributed as dist

        dist.barrier()  # other ranks must wait for rank 0 to finish packing

    seq_len = config.model.max_seq_length
    dataset = PackedTokenDataset(tokens_path, seq_len)

    # -- model --------------------------------------------------------------
    model_cfg = NanoConfig(
        vocab_size=config.model.vocab_size or len(tokenizer),
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        n_kv_heads=config.model.n_kv_heads,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        max_seq_length=seq_len,
        dropout=config.model.dropout,
        rope_theta=config.model.rope_theta,
        tie_embeddings=config.model.tie_embeddings,
    )

    if env.is_main:
        breakdown = estimate_parameters(model_cfg)
        logger.info(
            "model: %.1fM parameters (embedding %.1fM, blocks %.1fM) | %s windows of %d tokens",
            breakdown["total"] / 1e6,
            breakdown["embedding"] / 1e6,
            breakdown["blocks"] / 1e6,
            f"{len(dataset):,}",
            seq_len,
        )

    if dry_run:
        logger.info("dry run complete — config, corpus, and model all valid.")
        env.teardown()
        return out_dir

    model = NanoLM(model_cfg).to(device)

    start_step = 0
    if resume:
        ckpt = torch.load(Path(resume) / "model.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        logger.info("resumed from %s at step %d", resume, start_step)

    if env.is_distributed:
        model = DistributedDataParallel(model, device_ids=[env.local_rank])

    # -- optimizer ----------------------------------------------------------
    # Weight decay on matrices only. Decaying biases, norm weights, and
    # embeddings is a well-documented mistake: they are not the parameters
    # overfitting is coming from, and shrinking them hurts.
    decay, no_decay = [], []
    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.optim.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.optim.learning_rate,
        betas=(0.9, 0.95),  # beta2=0.95, not 0.999: standard for LM pretraining
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )

    use_bf16 = config.optim.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    micro_bs = config.optim.per_device_batch_size
    accum = config.optim.gradient_accumulation_steps
    tokens_per_step = micro_bs * accum * seq_len * env.world_size

    steps_per_epoch = max(1, len(dataset) // (micro_bs * accum * env.world_size))
    total_steps = max(1, int(steps_per_epoch * config.optim.epochs))
    warmup_steps = max(1, int(total_steps * config.optim.warmup_ratio))

    if env.is_main:
        logger.info(
            "schedule: %d steps (%d warmup) | %s tokens/step | %s tokens total | bf16=%s",
            total_steps, warmup_steps, f"{tokens_per_step:,}",
            f"{tokens_per_step * total_steps:,}", use_bf16,
        )
        if len(dataset) < micro_bs * accum:
            logger.warning(
                "corpus yields only %d window(s) of %d tokens — smaller than one "
                "effective batch (%d). The model will memorize it almost immediately. "
                "Point data.text_file at a real corpus to train properly.",
                len(dataset), seq_len, micro_bs * accum,
            )

    # -- loop ---------------------------------------------------------------
    history: list[dict[str, float]] = []
    model.train()
    batches = dataset.iter_batches(micro_bs, shuffle=True, seed=config.optim.seed)
    started = time.perf_counter()
    running_loss = 0.0

    for step in range(start_step, total_steps):
        lr = lr_at_step(step, total_steps, config.optim.learning_rate, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro in range(accum):
            try:
                batch = next(batches)
            except StopIteration:
                batches = dataset.iter_batches(micro_bs, shuffle=True, seed=config.optim.seed + step)
                batch = next(batches)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Skip the gradient all-reduce on every micro-batch except the last:
            # syncing 8 times per optimizer step wastes ~7/8 of the network
            # traffic for an identical result.
            sync = (micro == accum - 1) or not env.is_distributed
            ctx = model.no_sync() if (env.is_distributed and not sync) else _nullcontext()

            with ctx, torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_bf16):
                out = model(input_ids, labels=labels)
                # Divide by `accum` so the accumulated gradient equals the mean
                # over the full effective batch, not the sum.
                loss = out["loss"] / accum

            loss.backward()
            step_loss += loss.item()

        # Clip by global norm before stepping. Language-model training produces
        # occasional huge-gradient batches; without clipping, one of them can
        # destroy hours of progress in a single update.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_grad_norm)
        optimizer.step()

        running_loss += step_loss

        if env.is_main and (step + 1) % config.logging_steps == 0:
            elapsed = time.perf_counter() - started
            done = step - start_step + 1
            avg_loss = running_loss / config.logging_steps
            running_loss = 0.0
            tok_per_s = tokens_per_step * done / elapsed
            record = {
                "step": step + 1,
                "loss": round(avg_loss, 4),
                # Perplexity = exp(loss): the model's effective branching factor.
                # More interpretable than raw loss — ppl 20 means "about as
                # uncertain as choosing among 20 equally likely tokens".
                "perplexity": round(math.exp(min(20, avg_loss)), 2),
                "lr": round(lr, 8),
                "grad_norm": round(float(grad_norm), 3),
                "tokens_per_second": round(tok_per_s),
            }
            history.append(record)
            logger.info(
                "step %d/%d | loss %.4f | ppl %.1f | lr %.2e | gnorm %.2f | %s tok/s | eta %s",
                step + 1, total_steps, avg_loss, record["perplexity"], lr,
                record["grad_norm"], f"{int(tok_per_s):,}",
                _format_eta((total_steps - step - 1) * elapsed / done),
            )

        if env.is_main and config.save_steps and (step + 1) % config.save_steps == 0:
            _save_checkpoint(out_dir / f"checkpoint-{step + 1}", model, model_cfg, step + 1, history)

    # -- finish -------------------------------------------------------------
    if env.is_main:
        final = out_dir / "final"
        _save_checkpoint(final, model, model_cfg, total_steps, history)
        tokenizer.save_pretrained(str(final))
        (out_dir / "metrics.json").write_text(
            json.dumps({"history": history, "total_steps": total_steps}, indent=2), encoding="utf-8"
        )
        logger.info("done in %.1f min — checkpoint at %s", (time.perf_counter() - started) / 60, final)
        _sample(model, tokenizer, device)

    env.teardown()
    return out_dir


def _nullcontext():
    import contextlib

    return contextlib.nullcontext()


def _format_eta(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _save_checkpoint(path: Path, model: Any, cfg: Any, step: int, history: list) -> None:
    import dataclasses

    import torch

    path.mkdir(parents=True, exist_ok=True)
    # Unwrap DDP so the checkpoint loads without a process group.
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({"model": state, "step": step, "config": dataclasses.asdict(cfg)}, path / "model.pt")
    (path / "history.json").write_text(json.dumps(history[-100:], indent=2), encoding="utf-8")
    logger.info("checkpoint -> %s", path)


def _sample(model: Any, tokenizer: Any, device: Any, prompt: str = "The") -> None:
    """Generate a short sample so you can *see* whether the model learned anything.

    Loss curves tell you training is proceeding; only reading output tells you it
    is proceeding toward language. A run whose loss drops steadily while the
    samples stay incoherent usually means a data bug, not an optimization one.
    """
    import torch

    net = model.module if hasattr(model, "module") else model
    ids = torch.tensor([tokenizer(prompt, add_special_tokens=False)["input_ids"]], device=device)
    out = net.generate(ids, max_new_tokens=80, temperature=0.8, top_k=50)
    logger.info("sample: %s", tokenizer.decode(out[0].tolist())[:400].replace("\n", " "))


def main() -> None:
    """Entry point for `torchrun -m kumaru.train.pretrain_nano <config.yaml>`.

    `kumaru train` calls `run_pretrain` directly for single-GPU runs; this exists
    so multi-GPU launches go through torchrun, which sets up the process group.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain a NanoLM from scratch.")
    parser.add_argument("config", help="Training config YAML.")
    parser.add_argument("--resume", default=None, help="Checkpoint directory to resume from.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_pretrain(load_train_config(args.config), resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
