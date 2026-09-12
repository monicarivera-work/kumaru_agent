# `src\kumaru\train\config.py`

> Typed Pydantic schemas for every Kumaru training YAML file.

**Read this when:** you are editing `configs\train\*.yaml`, comparing two training runs, or debugging a `ConfigError` before a run starts.

---

## What it does

`config.py` defines the complete schema for LoRA fine-tuning and NanoLM pretraining. It forbids unknown fields, validates ranges such as `eval_split`, computes effective batch size, and saves the resolved config beside each run's outputs.

The loader reads YAML and converts it into a `TrainConfig`. Downstream training code receives typed objects instead of loose dictionaries.

## Why it exists

A training config is the experiment record. Six months after an adapter is saved, `runs\...\train_config.yaml` should answer what base model, data, rank, learning rate, precision, and schedule produced it. Without this schema, typos silently become ignored settings.

## Mental model

1. **The YAML is the contract.** If a key is not in this file, it is rejected.
2. **Sections map directly to classes.** `data`, `lora`, `optim`, and `model` become nested config objects.
3. **Resolved configs are artifacts.** `TrainConfig.save()` writes the exact recipe to the output directory.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `_Base` | `class _Base(BaseModel)` | Internal base model with `extra="forbid"`; prevents silent typos. |
| `DataConfig` | `class DataConfig(_Base)` | File paths, split behavior, sequence length, masking, and tokenizer selection. |
| `LoraConfig` | `class LoraConfig(_Base)` | LoRA rank, scaling, dropout, target modules, and bias policy. |
| `OptimConfig` | `class OptimConfig(_Base)` | Epochs, batch sizing, optimizer, precision, LR schedule, clipping, and seed. |
| `NanoModelConfig` | `class NanoModelConfig(_Base)` | From-scratch transformer architecture parameters. |
| `TrainConfig` | `class TrainConfig(_Base)` | Top-level training run configuration. |
| `TrainConfig.effective_batch_size` | `property -> int` | `per_device_batch_size * gradient_accumulation_steps`; used in logs and planning. |
| `TrainConfig.save` | `save(path: Path) -> None` | Writes resolved YAML to disk, creating parent directories. |
| `load_train_config` | `load_train_config(path: str | Path) -> TrainConfig` | Parses YAML, validates it, and raises `ConfigError` with a useful message. |

### YAML fields

| Field | Type | Default | Controls | How to choose |
|---|---:|---:|---|---|
| `task` | `lora` or `pretrain` | `lora` | Which training implementation runs. | Use `lora` for adapters; `pretrain` for NanoLM. |
| `base_model` | string | `Qwen/Qwen2.5-7B-Instruct` | Hugging Face model for LoRA. | Use an instruct checkpoint with a chat template. |
| `output_dir` | string | `runs/untitled` | Where run artifacts are written. | Use a unique directory under `runs\`. |
| `run_name` | string or null | null | Trainer display name. | Set for TensorBoard or external reporting. |
| `load_in_4bit` | bool | false | QLoRA 4-bit NF4 base loading. | True for 7B on 24 GB GPUs; false for 0.5B smoke. |
| `trust_remote_code` | bool | false | Hugging Face custom code loading. | Keep false unless the model requires trusted custom code. |
| `logging_steps` | int | 10 | Log interval. | Lower for short tests, higher for long runs. |
| `eval_steps` | int | 100 | LoRA evaluation interval. | Choose several evals per epoch. |
| `save_steps` | int | 200 | Checkpoint interval. | Balance recovery with disk use. |
| `save_total_limit` | int | 3 | Number of LoRA checkpoints retained by Trainer. | Use 1-3 locally. |
| `report_to` | list[string] | `[]` | Trainer integrations. | `[]` for none; `["tensorboard"]` for curves. |
| `extra` | object | `{}` | Free-form metadata. | Store notes; code does not interpret it. |
| `data.train_file` | string | required | SFT JSONL path; fallback corpus path for pretrain. | Use `data/samples/chat_train.jsonl` for examples. |
| `data.eval_file` | string or null | null | Explicit SFT eval JSONL. | Prefer for stable eval; else use `eval_split`. |
| `data.eval_split` | float `[0, 0.5)` | 0.05 | Deterministic held-out fraction. | Starter LoRA configs use 0.1. |
| `data.max_seq_length` | int | 2048 | SFT token limit; overlong examples are dropped. | 2048 for 7B, 1024 for small. |
| `data.train_on_completions_only` | bool | true | Masks non-assistant labels to `-100`. | Keep true for SFT. |
| `data.text_file` | string or null | null | Raw UTF-8 corpus for pretraining. | Set for `task: pretrain`. |
| `data.tokenizer` | string | `Qwen/Qwen2.5-1.5B-Instruct` | Tokenizer for pretraining and vocab size. | Starter uses `Qwen/Qwen2.5-0.5B-Instruct`. |
| `lora.r` | int | 16 | Adapter rank/capacity. | 8-16 for style/format; 32-64 for harder behavior. |
| `lora.alpha` | int | 32 | LoRA update scale. | Usually `2 * r`. |
| `lora.dropout` | float | 0.05 | Adapter dropout. | 0.05 is safe for small/medium data. |
| `lora.target_modules` | list[string] | q/k/v/o + gate/up/down | Modules receiving adapters. | All seven for 7B; attention-only for smoke. |
| `lora.bias` | `none`, `all`, `lora_only` | `none` | Bias training policy. | Keep `none` unless reproducing an experiment. |
| `optim.epochs` | float | 3.0 | Dataset passes. | 2-4 for LoRA; sample pretraining overfits quickly. |
| `optim.per_device_batch_size` | int | 1 | Micro-batch per GPU. | 1 for 7B/2k context; 2 for 0.5B; 16 for NanoLM. |
| `optim.gradient_accumulation_steps` | int | 16 | Micro-batches per optimizer step. | Increase effective batch without much memory cost. |
| `optim.learning_rate` | float | `2e-4` | Peak LR. | `2e-4` LoRA; `3e-4` NanoLM starter. |
| `optim.lr_scheduler` | `cosine`, `linear`, `constant` | `cosine` | LR schedule. | Use cosine unless comparing schedules. |
| `optim.warmup_ratio` | float | 0.03 | Warmup fraction. | 0.02-0.05 in checked-in configs. |
| `optim.weight_decay` | float | 0.0 | AdamW decay. | 0 for LoRA; 0.1 for NanoLM matrices. |
| `optim.max_grad_norm` | float | 1.0 | Global gradient clipping. | Keep 1.0 unless diagnosing instability. |
| `optim.gradient_checkpointing` | bool | true | Recompute activations to save memory. | Enable on OOM; about 30% slower, about 40% less activation memory. |
| `optim.optimizer` | string | `adamw_torch` | Trainer optimizer name or NanoLM AdamW reference. | `paged_adamw_8bit` needs bitsandbytes. |
| `optim.bf16` | bool | true | Use bfloat16 where supported. | Good on modern NVIDIA GPUs. |
| `optim.fp16` | bool | false | Use fp16 in Trainer. | Use when bf16 is unavailable. |
| `optim.seed` | int | 42 | Split, shuffle, and torch seed. | Fix for comparable runs. |
| `model.vocab_size` | int | 0 | NanoLM vocabulary size. | `0` inherits tokenizer length. |
| `model.n_layers` | int | 8 | Transformer block count. | Scale up for larger models. |
| `model.n_heads` | int | 8 | Query attention heads. | Must divide `d_model`. |
| `model.n_kv_heads` | int | 4 | Key/value heads for GQA. | Must divide `n_heads`; fewer saves KV memory. |
| `model.d_model` | int | 512 | Residual width. | Main size knob with layers. |
| `model.d_ff` | int | 1408 | SwiGLU hidden width. | About 2.67-2.75x `d_model`. |
| `model.max_seq_length` | int | 512 | NanoLM context/window. | Match corpus packing. |
| `model.dropout` | float | 0.0 | Transformer dropout. | 0 for large corpus; consider >0 for tiny runs. |
| `model.rope_theta` | float | 10000.0 | RoPE base. | Keep default unless context-scaling experiments. |
| `model.tie_embeddings` | bool | true | Share input/output embeddings. | Keep true with large vocabularies. |

## How to use it

Validate a config without training:

```powershell
kumaru train configs\train\lora-qwen.yaml --dry-run
```

Load and inspect in Python:

```python
from kumaru.train.config import load_train_config
cfg = load_train_config("configs/train/lora-qwen.yaml")
print(cfg.task, cfg.effective_batch_size)
```

Minimal LoRA YAML matching the small starter:

```yaml
task: lora
base_model: Qwen/Qwen2.5-0.5B-Instruct
output_dir: runs/lora-small
load_in_4bit: false
data:
  train_file: data/samples/chat_train.jsonl
  eval_split: 0.1
  max_seq_length: 1024
  train_on_completions_only: true
lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
optim:
  epochs: 3
  per_device_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.05
  gradient_checkpointing: false
  optimizer: adamw_torch
  bf16: true
```

Pretraining YAML matching the checked-in starter:

```yaml
task: pretrain
output_dir: runs/nano
data:
  train_file: data/samples/corpus.txt
  text_file: data/samples/corpus.txt
  tokenizer: Qwen/Qwen2.5-0.5B-Instruct
  max_seq_length: 512
model:
  vocab_size: 0
  n_layers: 8
  n_heads: 8
  n_kv_heads: 4
  d_model: 512
  d_ff: 1408
  max_seq_length: 512
  dropout: 0.0
  rope_theta: 10000.0
  tie_embeddings: true
optim:
  epochs: 3
  per_device_batch_size: 16
  gradient_accumulation_steps: 4
  learning_rate: 3.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.02
  weight_decay: 0.1
  max_grad_norm: 1.0
  gradient_checkpointing: false
  optimizer: adamw_torch
  bf16: true
```

## Tuning / operational notes

On a 24 GB RTX 3090, `lora-qwen.yaml` is the realistic full run: 4-bit Qwen2.5-7B-Instruct, batch 1, accumulation 16, 2048-token limit, and `paged_adamw_8bit`. Its comments report about 9-11 GB VRAM and about 20-40 minutes per epoch on 1,000 examples of about 1k tokens.

`lora-small.yaml` is the safety check: Qwen2.5-0.5B-Instruct, batch 2, accumulation 4, no quantization, no gradient checkpointing, about 2 GB VRAM, and a few minutes.

`pretrain-nano.yaml` defines a roughly 35M-parameter model. Chinchilla-style sizing wants about 20 tokens per parameter, so a properly sized 35M run wants about 700M tokens, roughly 3 GB of text.

If you hit OOM, change these in order:

1. `optim.per_device_batch_size: 1`.
2. `load_in_4bit: true` for 7B LoRA.
3. `optim.gradient_checkpointing: true`.
4. Reduce `data.max_seq_length` or `model.max_seq_length`.
5. Reduce `lora.r` or `model.d_model`/`model.n_layers`.

## How to extend it

Add fields only when training code consumes them. Example: configurable cosine LR floor for pretraining:

```python
class OptimConfig(_Base):
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
```

Then pass `config.optim.min_lr_ratio` into `lr_at_step()` in `pretrain_nano.py`. Because `_Base` forbids extra keys, YAML cannot claim support before code implements it.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Invalid training config ... extra fields not permitted` | YAML key does not exist in schema | Fix spelling or add the field and implementation. |
| `Training config not found` | Wrong config path | Run from repo root or pass an absolute path. |
| Eval loss missing during LoRA | No eval file, split disabled, or fewer than 20 rows | Provide `data.eval_file` or enough rows. |
| All examples dropped | `data.max_seq_length` too small | Increase it or shorten examples. |
| 7B run requires bitsandbytes | 4-bit or paged optimizer without quant deps | Install `kumaru[quant]` or use `adamw_torch` and no 4-bit. |
| bf16 warning in preflight | GPU/PyTorch lacks bf16 | Set `optim.bf16: false`; consider fp16 for LoRA. |

## Related files

- `configs\train\lora-qwen.yaml` — QLoRA 7B example using most fields.
- `configs\train\lora-small.yaml` — quick LoRA smoke-test config.
- `configs\train\pretrain-nano.yaml` — NanoLM from-scratch config.
- `src\kumaru\train\runner.py` — loads `TrainConfig` and dispatches the task.
- `src\kumaru\core\errors.py` — defines `ConfigError` used by the loader.
