# `src\kumaru\train\__init__.py`

> Package-level orientation for Kumaru training: LoRA fine-tuning for practical adaptation and NanoLM pretraining for learning the mechanics.

**Read this when:** you need to choose between `kumaru train configs/train/lora-qwen.yaml` and `kumaru train configs/train/pretrain-nano.yaml`, or you want the shortest map of the training package.

---

## What it does

This file makes `kumaru.train` a Python package and documents the two supported training paths. It points operators toward LoRA/QLoRA for real adaptation work and toward from-scratch pretraining when they want to study or scale the training loop itself.

The module has no runtime code. Its value is the docstring: it states the operating default for Kumaru training.

## Why it exists

Training a model from scratch and adapting a pretrained model solve different problems. Without this package-level note, a new operator can easily choose the expensive path first and expect a useful assistant from a tiny pretraining run.

LoRA fine-tuning changes style, format, and domain-specific behavior on existing knowledge. Pretraining from zero makes mechanisms visible and can scale to real pretraining, but competitive pretraining requires enormous compute.

## Mental model

1. **Use LoRA by default.** If you already have a capable instruct model, train an adapter.
2. **Use NanoLM pretraining to understand the loop.** The nano path is educational and scalable, not a shortcut to a production assistant.
3. **Both paths enter through `kumaru train`.** The YAML `task` field selects the implementation.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| Module docstring | n/a | Documents the two training paths and points to `docs/TRAINING.md`. |

This module intentionally exports no classes, functions, or constants.

## How to use it

Choose the 7B QLoRA path for practical supervised fine-tuning:

```powershell
kumaru train configs\train\lora-qwen.yaml --dry-run
kumaru train configs\train\lora-qwen.yaml
kumaru chat --model Qwen/Qwen2.5-7B-Instruct --adapter runs\lora-qwen\adapter
```

Use the small LoRA config to validate the end-to-end pipeline before spending an evening on a larger model:

```powershell
kumaru train configs\train\lora-small.yaml --dry-run
kumaru train configs\train\lora-small.yaml
kumaru chat --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs\lora-small\adapter
```

Use the nano pretraining config to train a small model from random initialization:

```powershell
kumaru train configs\train\pretrain-nano.yaml --dry-run
kumaru train configs\train\pretrain-nano.yaml
```

For multiple GPUs, bypass the Typer command and launch the pretraining module with `torchrun`:

```powershell
torchrun --nproc_per_node=4 -m kumaru.train.pretrain_nano configs\train\pretrain-nano.yaml
```

## Tuning / operational notes

| Path | Best use | 24 GB RTX 3090 expectation |
|---|---|---|
| `lora-small.yaml` | Pipeline smoke test on Qwen2.5-0.5B | A few minutes, about 2 GB VRAM. |
| `lora-qwen.yaml` | Practical QLoRA on Qwen2.5-7B-Instruct | About 9-11 GB VRAM; about 20-40 minutes per epoch on 1,000 examples of about 1k tokens. |
| `pretrain-nano.yaml` | Understanding and debugging from-scratch training | Under an hour for the sample-sized 35M-ish model; quality is not assistant-grade. |

The most important knobs live below this package:

- `data.max_seq_length`: longer SFT examples are dropped, not truncated.
- `data.train_on_completions_only`: keep `true` for SFT so only assistant tokens carry loss.
- `optim.gradient_accumulation_steps`: increases effective batch without increasing activation memory.
- `load_in_4bit`: enables QLoRA for 7B-class models.
- `model.*`: controls NanoLM size for from-scratch runs.

If you hit CUDA OOM, change these in order:

1. Use `configs\train\lora-small.yaml` to confirm the pipeline is healthy.
2. For QLoRA, ensure `load_in_4bit: true`.
3. Set `optim.per_device_batch_size: 1`.
4. Turn on `optim.gradient_checkpointing: true`.
5. Reduce `data.max_seq_length`.
6. Reduce LoRA `r`, or move from 7B to 0.5B.

## How to extend it

Add package-level exports only when there is a stable public training API worth importing from `kumaru.train`. For example, if operators commonly need the config loader directly:

```python
from kumaru.train.config import TrainConfig, load_train_config

__all__ = ["TrainConfig", "load_train_config"]
```

Keep the docstring's decision rule current whenever a new training task is added. If you add `task: dpo`, state when DPO is preferable to SFT and point to the exact starter config.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Operator starts `pretrain-nano` expecting a useful assistant | Confusing pretraining from zero with fine-tuning an instruct model | Use `configs\train\lora-qwen.yaml` for adaptation. |
| `kumaru train` reports an unknown task | YAML `task` is not `lora` or `pretrain` | Use one of the checked-in configs as the base. |
| 7B training is much slower than expected | Running on CPU or no CUDA device visible | Run `kumaru info`; install CUDA-enabled PyTorch. |
| CUDA OOM before training starts | Full precision 7B or too-large sequence/batch | Use QLoRA, batch size 1, checkpointing, then shorter sequences. |
| Loss decreases but generations are worse | Data formatting or loss masking problem | Run `--dry-run` and inspect `example_preview.txt`. |

## Related files

- `src\kumaru\train\runner.py` — dispatches `task: lora` and `task: pretrain`.
- `src\kumaru\train\config.py` — validates the YAML files used by both paths.
- `src\kumaru\train\finetune_lora.py` — implements LoRA/QLoRA supervised fine-tuning.
- `src\kumaru\train\pretrain_nano.py` — implements the visible from-scratch pretraining loop.
- `docs\TRAINING.md` — broader operator walkthrough referenced by the module docstring.
