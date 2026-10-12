# `src\kumaru\train\finetune_lora.py`

> LoRA and QLoRA supervised fine-tuning implementation built on Hugging Face Trainer and PEFT.

**Read this when:** you are running adapter training, debugging a failed QLoRA load, inspecting masking previews, or changing LoRA targets and Trainer arguments.

---

## What it does

This file freezes a pretrained causal LM and trains a small PEFT LoRA adapter on Kumaru chat data. It loads the tokenizer, verifies a chat template exists, builds completion-masked datasets, optionally quantizes the base model to 4-bit NF4, configures PEFT, runs `Trainer`, and saves the adapter, tokenizer, metrics, and resolved training config.

It also supports `--dry-run`, which validates config and data and writes `example_preview.txt` without loading or training the base model.

## Why it exists

Full fine-tuning a 7B model is dominated by optimizer state and gradients. The module docstring shows the arithmetic: full bf16 fine-tuning is roughly 95 GB, while QLoRA at `r=16` is roughly 7.5 GB for the same class of model. That difference is why a consumer GPU can train an adapter.

Without this file, Kumaru would have no practical local training path for style, format, and domain-specific behavior. Pretraining from scratch is educational, but LoRA is the operator path for useful adaptation.

## Mental model

1. **The base model is frozen.** Only low-rank adapter weights are trainable.
2. **QLoRA saves base-weight memory.** `load_in_4bit` stores the frozen base in NF4 while adapter math runs in bf16/float.
3. **Good SFT depends on data masking.** The Trainer sees labels where only assistant tokens are supervised.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `ListDataset` | `ListDataset(items: list[dict[str, list[int]]])` | Minimal in-memory dataset wrapper for pre-tokenized SFT examples. |
| `ListDataset.__len__` | `__len__() -> int` | Returns number of encoded examples. |
| `ListDataset.__getitem__` | `__getitem__(idx: int) -> dict[str, list[int]]` | Returns one encoded item for Trainer. |
| `_filter_kwargs` | `_filter_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]` | Internal compatibility helper for different `TrainingArguments` versions. |
| `run_lora_finetune` | `run_lora_finetune(config: TrainConfig, *, resume: str | None = None, dry_run: bool = False) -> Path` | Executes or validates a LoRA/QLoRA run and returns the adapter/output path. |
| `_preview` | `_preview(item: dict[str, list[int]], tokenizer: Any, out_dir: Path) -> None` | Internal dry-run helper that writes full/supervised text preview. |

## How to use it

Validate the 7B QLoRA run first:

```powershell
kumaru train configs\train\lora-qwen.yaml --dry-run
Get-Content runs\lora-qwen\example_preview.txt
```

Train and load the adapter:

```powershell
kumaru train configs\train\lora-qwen.yaml
kumaru chat --model Qwen/Qwen2.5-7B-Instruct --adapter runs\lora-qwen\adapter
```

Run the fast 0.5B smoke test:

```powershell
kumaru train configs\train\lora-small.yaml --dry-run
kumaru train configs\train\lora-small.yaml
kumaru chat --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs\lora-small\adapter
```

Resume a Trainer checkpoint:

```powershell
kumaru train configs\train\lora-qwen.yaml --resume runs\lora-qwen\checkpoint-100
```

Call from Python:

```python
from kumaru.train.config import load_train_config
from kumaru.train.finetune_lora import run_lora_finetune

cfg = load_train_config("configs/train/lora-small.yaml")
adapter_dir = run_lora_finetune(cfg, dry_run=True)
print(adapter_dir)
```

The 7B QLoRA config shape is:

```yaml
task: lora
base_model: Qwen/Qwen2.5-7B-Instruct
output_dir: runs/lora-qwen
load_in_4bit: true
data:
  train_file: data/samples/chat_train.jsonl
  eval_file: null
  eval_split: 0.1
  max_seq_length: 2048
  train_on_completions_only: true
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  bias: none
optim:
  per_device_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-4
  optimizer: paged_adamw_8bit
  gradient_checkpointing: true
  bf16: true
```

## Tuning / operational notes

The memory budget in the source is the key fact:

| Component | Full fine-tune | QLoRA r=16 |
|---|---:|---:|
| Base weights | 15.2 GB bf16 | 4.5 GB 4-bit NF4 |
| Gradients | 15.2 GB | 0.08 GB |
| Adam state | 60.8 GB | 0.32 GB |
| Activations, 2k seq, batch 1 | about 4 GB | about 2.5 GB checkpointed |
| Total | about 95 GB | about 7.5 GB |

The checked-in `lora-qwen.yaml` reports about 9-11 GB VRAM on a 3090 and about 20-40 minutes per epoch on roughly 1,000 examples of about 1k tokens. Use `lora-small.yaml` first; it finishes in a few minutes and about 2 GB VRAM.

Most important hyperparameters:

- `lora.r`: capacity. 8-16 for style/format, 32-64 for harder domain behavior. Higher ranks overfit faster on small datasets.
- `lora.alpha`: scaling. Keep `alpha = 2 * r` unless changing update magnitude intentionally.
- `target_modules`: attention+MLP is standard and stronger; attention-only is cheaper for smoke tests.
- `data.max_seq_length`: memory and retention tradeoff; long rows are dropped.
- `optim.gradient_accumulation_steps`: effective batch without storing more activations.
- `optim.learning_rate`: LoRA commonly uses `1e-4` to `3e-4`; the starter uses `2e-4`.

If you hit OOM, change these in order:

1. Ensure `load_in_4bit: true` for 7B.
2. Set `optim.per_device_batch_size: 1`.
3. Set `optim.gradient_checkpointing: true`.
4. Reduce `data.max_seq_length` from 2048 to 1536 or 1024.
5. Reduce `lora.r` or use attention-only targets.
6. Fall back to `Qwen/Qwen2.5-0.5B-Instruct` to verify the pipeline.

## How to extend it

To add a different adapter target set, edit YAML first; no code change is needed if module names match the base model:

```yaml
lora:
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

To support a new Trainer argument, add the field to `OptimConfig` or `TrainConfig`, then include it in the kwargs dictionary passed to `TrainingArguments`. Keep `_filter_kwargs()` in place so older Transformers versions ignore unsupported names.

To add another PEFT method, add a new config section in `config.py`, branch in `run_lora_finetune()`, and keep the adapter save contract (`out_dir / "adapter"`) so `kumaru chat --adapter ...` continues to work.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Optional dependency 'transformers' is not installed` | Training extras are missing | `.\scripts\setup.ps1 -Gpu`; elsewhere `pip install -e ".[train]"` in the Kumaru folder. |
| `Optional dependency 'peft' is not installed` | PEFT missing | `.\scripts\setup.ps1 -Gpu`; elsewhere `pip install -e ".[train]"` in the Kumaru folder. |
| `load_in_4bit requires bitsandbytes` or bitsandbytes import fails | Quant extras missing | `.\scripts\setup.ps1 -Quant`, or `pip install -e ".[quant]"` in the Kumaru folder; or disable 4-bit. |
| Base model has no chat template | Non-instruct tokenizer/model | Use an `-Instruct` checkpoint or set a tokenizer chat template. |
| Adapter not changing output | Wrong adapter path, too little data, or labels supervise wrong tokens | Load `runs\...\adapter`, inspect `example_preview.txt`, use 500+ examples. |
| Eval loss rises while train loss falls | Overfitting small data or rank too high | Reduce epochs/rank, add data, watch held-out eval loss. |

## Related files

- `src\kumaru\train\data.py` — supplies chat loading, masking, and collator.
- `src\kumaru\train\config.py` — defines `TrainConfig`, `LoraConfig`, and optimizer settings.
- `src\kumaru\train\runner.py` — calls `run_lora_finetune` when `task: lora`.
- `configs\train\lora-qwen.yaml` — production-like QLoRA starter.
- `configs\train\lora-small.yaml` — fast smoke-test starter.
- `src\kumaru\cli.py` — exposes `kumaru train <config> [--resume] [--dry-run]`.
