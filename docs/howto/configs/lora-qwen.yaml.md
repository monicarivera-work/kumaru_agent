# `configs/train/lora-qwen.yaml`

> QLoRA fine-tuning profile for Qwen2.5 7B on a single 24 GB GPU.

**Read this when:** you are ready to train a production-scale LoRA adapter after validating the data and pipeline on a smaller run.

---

## What it does

This training config freezes `Qwen/Qwen2.5-7B-Instruct` in 4-bit NF4, trains LoRA adapters in bf16, evaluates a deterministic holdout split, and writes the run under `runs/lora-qwen`.

The source comments report about 9-11 GB VRAM on an RTX 3090 and roughly 20-40 minutes per epoch for about 1,000 examples of about 1k tokens each.

## When to use this profile

Use it on a single 24 GB CUDA GPU after:

- the dataset validates with `--dry-run`;
- `configs/train/lora-small.yaml` has proven the training path;
- you want an adapter for `Qwen/Qwen2.5-7B-Instruct`, not a from-scratch model.

## Every field explained

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `task` | string | `lora` | Selects supervised LoRA fine-tuning. | Keep `lora` for adapter training. |
| `base_model` | string | `Qwen/Qwen2.5-7B-Instruct` | Frozen base checkpoint. | Must match the model you will later serve with the adapter. |
| `output_dir` | string | `runs/lora-qwen` | Directory for checkpoints, adapter, and resolved config. | Use a unique directory per experiment. |
| `run_name` | string | `qwen7b-lora-v1` | Human-readable run label. | Include model, method, and version. |
| `load_in_4bit` | bool | `true` | Quantizes frozen base weights for QLoRA. | Keep true on 24 GB GPUs; requires quant dependencies. |
| `data.train_file` | string | `data/samples/chat_train.jsonl` | JSONL SFT training data. | Point to your dataset when leaving samples. |
| `data.eval_file` | string/null | `null` | Optional separate eval JSONL. | Use for a fixed validation set; otherwise use `eval_split`. |
| `data.eval_split` | float | `0.1` | Fraction held out from train file when no eval file is set. | 5-10% is typical; keep enough examples for stable eval. |
| `data.max_seq_length` | int | `2048` | Max tokenized conversation length; longer examples are dropped. | Raise only if VRAM allows; shorten data if many examples drop. |
| `data.train_on_completions_only` | bool | `true` | Masks loss to assistant turns only. | Keep true for SFT unless deliberately training all tokens. |
| `lora.r` | int | `16` | LoRA rank/capacity. | 8-16 for style/format; 32-64 for harder domain behavior with more data. |
| `lora.alpha` | int | `32` | LoRA scaling factor. | Convention is `2 * r`; keep paired with rank. |
| `lora.dropout` | float | `0.05` | Adapter dropout. | Keep small; increase only for overfitting. |
| `lora.target_modules` | list[string] | attention + MLP projections | Modules that receive adapters. | This profile adapts `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. |
| `lora.bias` | string | `none` | Whether LoRA trains biases. | Keep `none` for standard PEFT LoRA. |
| `optim.epochs` | float | `3` | Full passes over training data. | 2-4 is typical; more can overfit small sets. |
| `optim.per_device_batch_size` | int | `1` | Microbatch per GPU. | Keep one for 2048-token 7B QLoRA on 24 GB. |
| `optim.gradient_accumulation_steps` | int | `16` | Steps accumulated before optimizer update. | Effective batch is `1 * 16 = 16`; increase for smoother gradients if time allows. |
| `optim.learning_rate` | float | `2.0e-4` | Adapter learning rate. | Good LoRA default; lower if eval regresses. |
| `optim.lr_scheduler` | string | `cosine` | LR schedule. | Use `cosine`, `linear`, or `constant` per schema. |
| `optim.warmup_ratio` | float | `0.03` | Fraction of training spent warming up LR. | Keep 0.02-0.05 for stable starts. |
| `optim.weight_decay` | float | `0.0` | Weight decay on trainable params. | LoRA is already constrained; keep zero unless overfitting. |
| `optim.max_grad_norm` | float | `1.0` | Gradient clipping norm. | Keep one to prevent spikes. |
| `optim.gradient_checkpointing` | bool | `true` | Recomputes activations to save memory. | Keep true for large models unless you have ample VRAM and want speed. |
| `optim.optimizer` | string | `paged_adamw_8bit` | Optimizer implementation. | Requires bitsandbytes; use `adamw_torch` if unavailable. |
| `optim.bf16` | bool | `true` | Enables bf16 training. | Use on Ampere+ GPUs. |
| `optim.seed` | int | `42` | Reproducibility seed. | Keep fixed for comparable eval splits/runs. |
| `logging_steps` | int | `5` | Training log interval. | Low value gives frequent feedback. |
| `eval_steps` | int | `50` | Evaluation interval. | Keep frequent enough to catch overfitting. |
| `save_steps` | int | `100` | Checkpoint save interval. | Balance recoverability with disk use. |
| `save_total_limit` | int | `3` | Number of checkpoints retained. | Increase if you need more rollback points. |
| `report_to` | list[string] | `[]` | Trainer reporting integrations. | Use `["tensorboard"]` if you want TensorBoard curves. |

## How to use it

Prerequisites, in the Kumaru folder:

```powershell
.\scripts\setup.ps1 -Gpu -Quant    # elsewhere: pip install -e ".[train,quant]"
```

Validate config and data first:

```powershell
kumaru train configs/train/lora-qwen.yaml --dry-run
```

Train:

```powershell
kumaru train configs/train/lora-qwen.yaml
```

Use the adapter:

```powershell
kumaru chat --model Qwen/Qwen2.5-7B-Instruct --adapter runs/lora-qwen/adapter -q "What changed?"
```

Or serve it:

```powershell
kumaru serve -c local-gpu --model Qwen/Qwen2.5-7B-Instruct --adapter runs/lora-qwen/adapter
```

## Tuning it

1. Replace `data.train_file` with your curated JSONL.
2. Run `--dry-run`; if examples drop, raise `data.max_seq_length` or shorten examples.
3. If OOM occurs, keep batch size one, keep checkpointing on, and lower sequence length.
4. If train loss falls but eval loss rises, reduce `epochs`, `learning_rate`, or `lora.r`.
5. If the adapter underfits, add better data before raising rank; then try `r: 32`, `alpha: 64`.
6. If bitsandbytes fails, switch `optim.optimizer` to `adamw_torch` and reassess memory.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `--dry-run` rejects JSONL | Bad line shape or missing assistant turn. | Fix data to match `{"messages": [...]}` or Alpaca fields. |
| Every example is dropped | Tokenized examples exceed `max_seq_length`. | Raise length or shorten examples. |
| CUDA OOM | Sequence length, optimizer, or batch exceeds VRAM. | Keep QLoRA, reduce length, keep checkpointing, or lower batch. |
| Eval loss rises while train loss falls | Overfitting. | Stop earlier, lower rank/LR, improve validation split. |
| Adapter loads poorly | Served with a different base model. | Use the same `base_model` at inference. |

## Related files

- `configs/train/lora-small.yaml` — fast pipeline validation profile.
- `data/samples/chat_train.jsonl` — accepted SFT JSONL shape.
- `src/kumaru/train/config.py` — authoritative training schema.
- `src/kumaru/train/data.py` — data validation, chat templating, and loss masking.
- `src/kumaru/cli.py` — `kumaru train <config> --dry-run` and `--resume`.

