# `src\kumaru\train\pretrain_nano.py`

> Custom from-scratch pretraining loop for NanoLM, including DDP setup, LR scheduling, accumulation, clipping, checkpointing, and sampling.

**Read this when:** you are running `task: pretrain`, launching multi-GPU `torchrun`, debugging convergence, or changing optimizer/schedule/checkpoint behavior.

---

## What it does

This file trains `NanoLM` from random initialization on a packed token corpus. It sets up single-process or `torchrun` distributed execution, packs text to `tokens.bin`, builds the model, runs AdamW with linear warmup plus cosine decay, accumulates gradients, clips global norm, writes checkpoints, records metrics, and prints a generated sample.

Unlike the LoRA path, it does not use Hugging Face `Trainer`. The point is to make each training mechanism explicit.

## Why it exists

When a from-scratch run fails, you need to see optimizer steps, LR values, accumulation, all-reduce timing, and checkpoint writes. A high-level trainer can hide the exact behavior.

This loop also demonstrates why data parallelism works: each rank computes gradients on a shard, all-reduce averages them, and every rank applies the same update. That is mathematically equivalent to one larger batch.

## Mental model

1. **Packed windows feed next-token loss.** `PackedTokenDataset` returns inputs and already-shifted labels.
2. **Gradient accumulation defines the optimizer batch.** Loss is divided by accumulation so gradients average, not sum.
3. **DDP synchronizes only when needed.** `no_sync()` skips all-reduce for intermediate micro-batches.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `DistEnv` | `DistEnv()` | Reads `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` from `torchrun`. |
| `DistEnv.is_distributed` | `property -> bool` | True when `WORLD_SIZE > 1`. |
| `DistEnv.is_main` | `property -> bool` | True on rank 0; only this rank logs and writes files. |
| `DistEnv.setup` | `setup() -> Any` | Initializes process group and returns the device. |
| `DistEnv.teardown` | `teardown() -> None` | Destroys the process group. |
| `lr_at_step` | `lr_at_step(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_ratio: float = 0.1) -> float` | Linear warmup followed by cosine decay to 10% by default. |
| `run_pretrain` | `run_pretrain(config: TrainConfig, *, resume: str | None = None, dry_run: bool = False) -> Path` | Runs or validates NanoLM pretraining and returns the output directory. |
| `_nullcontext` | `_nullcontext()` | Internal context manager for non-DDP micro-batches. |
| `_format_eta` | `_format_eta(seconds: float) -> str` | Formats ETA for logs. |
| `_save_checkpoint` | `_save_checkpoint(path: Path, model: Any, cfg: Any, step: int, history: list) -> None` | Saves `model.pt` and recent `history.json`. |
| `_sample` | `_sample(model: Any, tokenizer: Any, device: Any, prompt: str = "The") -> None` | Logs a short generated sample at the end. |
| `main` | `main() -> None` | CLI entry point for `python -m`/`torchrun -m`. |

## How to use it

Single GPU through Kumaru CLI:

```powershell
kumaru train configs\train\pretrain-nano.yaml --dry-run
kumaru train configs\train\pretrain-nano.yaml
```

Multi-GPU on one node:

```powershell
torchrun --nproc_per_node=4 -m kumaru.train.pretrain_nano configs\train\pretrain-nano.yaml
```

Resume from a saved checkpoint directory:

```powershell
kumaru train configs\train\pretrain-nano.yaml --resume runs\nano\checkpoint-500
```

Call the schedule helper:

```python
from kumaru.train.pretrain_nano import lr_at_step
for step in [0, 10, 100, 900]:
    print(step, lr_at_step(step, total_steps=1000, base_lr=3e-4, warmup_steps=20))
```

The checked-in config uses:

```yaml
task: pretrain
output_dir: runs/nano
data:
  train_file: data/samples/corpus.txt
  text_file: data/samples/corpus.txt
  tokenizer: Qwen/Qwen2.5-0.5B-Instruct
model:
  n_layers: 8
  n_heads: 8
  n_kv_heads: 4
  d_model: 512
  d_ff: 1408
  max_seq_length: 512
optim:
  epochs: 3
  per_device_batch_size: 16
  gradient_accumulation_steps: 4
  learning_rate: 3.0e-4
  warmup_ratio: 0.02
  weight_decay: 0.1
  max_grad_norm: 1.0
  bf16: true
```

## Tuning / operational notes

Warmup exists because Adam's second-moment estimate is unreliable at the beginning. The code linearly ramps to the base LR, then uses cosine decay. By default `lr_at_step` decays to 10% of base LR, not zero.

The loop computes `tokens_per_step = per_device_batch_size * gradient_accumulation_steps * seq_len * world_size`. With `pretrain-nano.yaml`, that is `16 * 4 * 512 * 1 = 32,768` tokens per optimizer step on one GPU. Four GPUs make it 131,072 tokens per optimizer step with the same per-device settings.

The starter model is around 35M parameters. The config comments note that Chinchilla-optimal training wants about 20 tokens per parameter, so a serious 35M run wants about 700M tokens, roughly 3 GB of text. The sample file is intentionally tiny and will overfit.

On a 24 GB RTX 3090, the starter should run in under an hour on sample-scale data. This loop does not shard optimizer state or parameters; when the model no longer fits, move to FSDP/ZeRO or tensor parallelism.

If you hit OOM, change these in order:

1. Lower `optim.per_device_batch_size`.
2. Keep accumulation if you still want the same effective batch.
3. Lower `model.max_seq_length`.
4. Lower `model.n_layers` or `model.d_model`.
5. Keep `model.tie_embeddings: true`.
6. Use distributed/sharded training before attempting truly large models.

## How to extend it

To add an LR schedule option, extend `OptimConfig.lr_scheduler` in `config.py`, then branch near `lr_at_step()` usage:

```python
if config.optim.lr_scheduler == "constant":
    lr = config.optim.learning_rate
else:
    lr = lr_at_step(step, total_steps, config.optim.learning_rate, warmup_steps)
```

To add validation loss, create a second `PackedTokenDataset` for a held-out memmap and run `model.eval()` every `eval_steps` under `torch.no_grad()`. Only rank 0 should log the result; keep ranks synchronized around evaluation.

To save optimizer state for exact resume, include `optimizer.state_dict()` in `_save_checkpoint()` and load it after creating the optimizer.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Optional dependency 'torch' is not installed` | Training extras missing | `.\scripts\setup.ps1 -Gpu`; elsewhere `pip install -e ".[torch]"` in the Kumaru folder (`.[train]` also works). |
| `tokens.bin` missing on non-main rank | Rank 0 did not pack or barrier failed | Ensure shared output directory and launch with `torchrun`. |
| `too few for seq_length` | Corpus token count is less than one window | Use more text or reduce `model.max_seq_length`. |
| NaN loss or exploding grad norm | LR too high, bad data, or unstable precision | Lower LR, keep warmup, use grad clipping, verify bf16 support. |
| Loss drops but sample is incoherent | Data bug or corpus too small | Inspect corpus, use larger text, confirm tokenizer. |
| Multi-GPU launch hangs on Windows | Backend/process-group environment issue | Use `torchrun`; Windows uses `gloo`, not NCCL. Start with one node. |

## Related files

- `src\kumaru\train\model_nano.py` — defines `NanoLM`, `NanoConfig`, and parameter estimates.
- `src\kumaru\train\data.py` — packs corpus and yields shifted token batches.
- `src\kumaru\train\config.py` — defines optimizer and model config fields.
- `src\kumaru\train\runner.py` — calls `run_pretrain` for `task: pretrain`.
- `configs\train\pretrain-nano.yaml` — starter config and operator comments.
- `docs\SCALABILITY.md` — referenced for FSDP/ZeRO and tensor parallel next steps.
