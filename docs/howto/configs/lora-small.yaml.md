# `configs/train/lora-small.yaml`

> Fast LoRA smoke-training profile for Qwen2.5 0.5B that validates the training pipeline before expensive 7B runs.

**Read this when:** you are changing training data, masking, PEFT setup, or runner code and need a cheap end-to-end check.

---

## What it does

This config trains a LoRA adapter on `Qwen/Qwen2.5-0.5B-Instruct`, writes to `runs/lora-small`, and uses a small attention-only adapter. It avoids 4-bit quantization and bitsandbytes because the 0.5B model fits easily in bf16.

## When to use this profile

Use it first, every time, before `lora-qwen`. It targets a GPU with roughly 2 GB free VRAM and should complete in minutes. It is for pipeline validation, not final model quality.

## Every field explained

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `task` | string | `lora` | Selects LoRA fine-tuning. | Keep `lora`. |
| `base_model` | string | `Qwen/Qwen2.5-0.5B-Instruct` | Small frozen base model. | Keep for smoke tests; do not compare quality with 7B. |
| `output_dir` | string | `runs/lora-small` | Output checkpoint/adapter directory. | Use a unique path if comparing changes. |
| `run_name` | string | `qwen0.5b-smoke` | Run label. | Keep descriptive and short. |
| `load_in_4bit` | bool | `false` | Whether to quantize base weights. | Keep false; quantization buys little at 0.5B. |
| `data.train_file` | string | `data/samples/chat_train.jsonl` | SFT JSONL file. | Point to your candidate dataset. |
| `data.eval_split` | float | `0.1` | Deterministic holdout fraction. | Keep 0.1 for small validation. |
| `data.max_seq_length` | int | `1024` | Max tokenized conversation length; longer examples are dropped. | Keep short for speed; raise only if validating longer data. |
| `data.train_on_completions_only` | bool | `true` | Loss on assistant tokens only. | Keep true for SFT. |
| `lora.r` | int | `8` | Adapter rank. | Low rank is enough for smoke testing. |
| `lora.alpha` | int | `16` | Adapter scaling. | Keep `2 * r`. |
| `lora.dropout` | float | `0.05` | Adapter dropout. | Keep default. |
| `lora.target_modules` | list[string] | `[q_proj, k_proj, v_proj, o_proj]` | Attention projections adapted. | Attention-only is faster for smoke tests. |
| `optim.epochs` | float | `3` | Number of passes. | Enough to see loss movement. |
| `optim.per_device_batch_size` | int | `2` | Microbatch per GPU. | Increase if VRAM is plentiful; keep stable for comparisons. |
| `optim.gradient_accumulation_steps` | int | `4` | Accumulation steps. | Effective batch is `8`. |
| `optim.learning_rate` | float | `2.0e-4` | LoRA learning rate. | Keep same family as 7B profile. |
| `optim.lr_scheduler` | string | `cosine` | Learning-rate schedule. | Valid values are `cosine`, `linear`, `constant`. |
| `optim.warmup_ratio` | float | `0.05` | Warmup fraction. | Slightly higher than 7B profile for small runs. |
| `optim.gradient_checkpointing` | bool | `false` | Activation recomputation. | Unnecessary at this size; leaving it off keeps smoke runs fast. |
| `optim.optimizer` | string | `adamw_torch` | Optimizer. | Avoids bitsandbytes dependency. |
| `optim.bf16` | bool | `true` | bf16 training. | Use on supported GPUs; change only if hardware requires. |
| `optim.seed` | int | `42` | Deterministic split/order seed. | Keep fixed. |
| `logging_steps` | int | `2` | Log interval. | Frequent logs for short runs. |
| `eval_steps` | int | `10` | Eval interval. | Frequent enough to catch masking/data issues. |
| `save_steps` | int | `50` | Save interval. | Low overhead for small model. |
| `save_total_limit` | int | `1` | Retained checkpoint count. | One is enough for smoke validation. |

## How to use it

Prerequisites, in the Kumaru folder:

```powershell
.\scripts\setup.ps1 -Gpu    # elsewhere: pip install -e ".[train]"
```

Validate:

```powershell
kumaru train configs/train/lora-small.yaml --dry-run
```

Train:

```powershell
kumaru train configs/train/lora-small.yaml
```

Load the adapter:

```powershell
kumaru chat --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/lora-small/adapter -q "Explain gradient accumulation."
```

## Tuning it

1. Change `data.train_file` to the dataset you want to validate.
2. Keep `data.max_seq_length` low unless the target data requires longer examples.
3. If dry-run drops many examples, inspect data lengths before using the 7B profile.
4. If loss is flat, verify labels are not all masked and examples contain assistant turns.
5. Only after this run succeeds, move to `lora-qwen`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Dry-run fails on line number | Malformed JSONL or missing fields. | Fix the reported line. |
| Loss does not move | Data is too tiny, labels are wrong, or all examples are near-identical. | Inspect encoded supervised token count and sample rows. |
| Adapter cannot be loaded | Wrong base model at inference. | Use `Qwen/Qwen2.5-0.5B-Instruct` with this adapter. |
| OOM on a small GPU | Batch/sequence too high for available VRAM. | Lower batch or sequence length. |
| Smoke run succeeds but 7B fails | Larger profile adds quantization, bitsandbytes, and more memory pressure. | Add the `quant` extra (`.\scripts\setup.ps1 -Quant`, or `pip install -e ".[quant]"` in the Kumaru folder) and tune `lora-qwen`. |

## Related files

- `configs/train/lora-qwen.yaml` — full 7B QLoRA profile.
- `data/samples/chat_train.jsonl` — sample SFT data.
- `src/kumaru/train/config.py` — training config schema.
- `src/kumaru/train/data.py` — JSONL validation and masking.

