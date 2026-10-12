# `src\kumaru\train\runner.py`

> Thin dispatcher behind `kumaru train` that loads a training config, runs preflight checks, and calls the selected training implementation.

**Read this when:** `kumaru train` fails before training starts, you are adding a new training task, or you want to understand the preflight output.

---

## What it does

`runner.py` loads a YAML file into `TrainConfig`, logs the planned task, model, output directory, epochs, effective batch, learning rate, LoRA settings, and GPU info, then checks common early failures. It verifies that every data file the task will read exists: for LoRA, `data.train_file` plus `data.eval_file` when it is set; for pretraining, `data.text_file`, or `data.train_file` when `text_file` is unset (the same fallback `run_pretrain()` uses). All missing files are reported in one error, before PyTorch is imported. It then imports PyTorch, warns if CUDA is unavailable, checks bf16 support, and validates bitsandbytes when 4-bit loading is requested.

After preflight, it dispatches `task: lora` to `run_lora_finetune()` or `task: pretrain` to `run_pretrain()`.

## Why it exists

Training failures are expensive when they happen after the model has loaded or after an hour of GPU time. This file moves predictable failures to minute 0: missing data, missing PyTorch, no CUDA, unsupported bf16, and missing bitsandbytes.

It also keeps the CLI command simple. `src\kumaru\cli.py` only parses arguments; this module owns training-specific decisions.

## Mental model

1. **Preflight is not training.** It prints and checks the plan before importing heavy task modules.
2. **YAML `task` is the switch.** Only `lora` and `pretrain` are accepted.
3. **Paths are process-relative.** Data paths are resolved relative to the directory where you run `kumaru train`.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `_data_files` | `_data_files(config: TrainConfig) -> list[tuple[str, str]]` | Internal. The `(config field, path)` pairs the selected task will open, mirroring the reads in `run_lora_finetune()` and `run_pretrain()`. |
| `preflight` | `preflight(config: TrainConfig) -> None` | Logs the run plan and raises/warns for foreseeable failures. |
| `run_training` | `run_training(config_path: Path, *, resume: str | None = None, dry_run: bool = False) -> Path` | Loads a config, runs preflight, dispatches to LoRA or pretraining, and returns the output path. |

## How to use it

Normal CLI usage:

```powershell
kumaru train configs\train\lora-small.yaml --dry-run
kumaru train configs\train\lora-qwen.yaml
kumaru train configs\train\pretrain-nano.yaml
```

Resume through the dispatcher:

```powershell
kumaru train configs\train\lora-qwen.yaml --resume runs\lora-qwen\checkpoint-100
kumaru train configs\train\pretrain-nano.yaml --resume runs\nano\checkpoint-500
```

Call from Python:

```python
from pathlib import Path
from kumaru.train.runner import run_training

out = run_training(Path("configs/train/lora-small.yaml"), dry_run=True)
print(out)
```

Expected preflight information includes task, base model, output directory, epochs, effective batch, learning rate, LoRA rank/alpha, whether the base is 4-bit, and visible GPU memory/bf16 support.

## Tuning / operational notes

`preflight()` does not estimate exact memory, but it surfaces the values that determine memory:

- task: LoRA vs pretrain
- base model for LoRA
- `load_in_4bit`
- per-device batch and accumulation
- learning rate and schedule
- LoRA rank and target-module count
- visible GPU memory and bf16 support

Use it with `--dry-run` before long runs. For `lora-qwen.yaml`, expect about 9-11 GB VRAM and about 20-40 minutes per epoch on a 24 GB RTX 3090/4090 with around 1,000 examples of about 1k tokens. For `lora-small.yaml`, expect a few minutes and about 2 GB VRAM. For `pretrain-nano.yaml`, expect an educational run that overfits the sample corpus quickly.

If preflight finds risk, change these in order:

1. For LoRA, enable `load_in_4bit`.
2. Reduce `optim.per_device_batch_size` to 1.
3. Enable `optim.gradient_checkpointing`.
4. Reduce `data.max_seq_length` (LoRA) or `model.max_seq_length` (pretraining, which ignores `data.max_seq_length`).
5. Use the small config to isolate data/config problems.

## How to extend it

To add a new training task, extend the literal in `TrainConfig.task`, create a training implementation, then add a dispatch branch:

```python
if config.task == "dpo":
    from kumaru.train.dpo import run_dpo
    return run_dpo(config, resume=resume, dry_run=dry_run)
```

Also update `preflight()` to print task-specific settings. Keep imports inside branches so unrelated tasks do not pay import costs for optional packages.

If the new task reads data files, add a branch for it to `_data_files()` so preflight still catches every missing file before training.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Training data not found: <path> (<field>); ...` | A file the task reads is missing: LoRA `data.train_file` or `data.eval_file`; pretraining `data.text_file`, or `data.train_file` when `text_file` is unset. Paths resolve from the current directory. | Run from the repo root or fix the YAML paths. The message lists every missing file with the field it came from. |
| `PyTorch is not installed` | Training extra missing | `.\scripts\setup.ps1 -Gpu`; elsewhere `pip install -e ".[train]"` in the Kumaru folder. The hint (`install_command('train')`) says the same. |
| Warning: no CUDA device visible | CPU-only PyTorch or driver/CUDA issue | Run `kumaru info`; install CUDA-enabled PyTorch. |
| Warning: bf16 requested but unsupported | GPU or PyTorch build lacks bf16 | Set `optim.bf16: false`; for LoRA consider `optim.fp16: true`. |
| `load_in_4bit requires bitsandbytes` | Quantization dependency missing | `.\scripts\setup.ps1 -Quant`, or `pip install -e ".[quant]"` in the Kumaru folder (`install_command('quant')`); or set `load_in_4bit: false`. |
| `Unknown training task` | YAML `task` not implemented | Use `lora` or `pretrain`, or add a dispatch branch. |

## Related files

- `src\kumaru\cli.py` — `train()` command calls `run_training(Path(config), resume=..., dry_run=...)`.
- `src\kumaru\train\config.py` — defines `TrainConfig` and `load_train_config`.
- `src\kumaru\train\finetune_lora.py` — LoRA dispatch target.
- `src\kumaru\train\pretrain_nano.py` — pretraining dispatch target.
- `src\kumaru\core\errors.py` — defines `ConfigError` used for preflight failures.
