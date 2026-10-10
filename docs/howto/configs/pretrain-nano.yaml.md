# `configs/train/pretrain-nano.yaml`

> Tiny from-scratch language-model pretraining profile for observing the full training loop on a readable corpus.

**Read this when:** you want to validate or study pretraining mechanics, token packing, small transformer architecture, or overfitting behavior.

---

## What it does

This config trains a small transformer from random initialization using raw text from `data/samples/corpus.txt`. It uses the Qwen2.5 0.5B tokenizer, builds a packed token stream, and trains a roughly 35M-parameter model under `runs/nano`.

It will not produce a useful assistant. The sample corpus is intentionally tiny, so loss will fall because the model memorizes.

## When to use this profile

Use it for mechanism tests and education on one GPU. The comments note that proper Chinchilla-style training for 35M parameters wants about 700M tokens, far beyond the sample file. Point `data.text_file` at a real UTF-8 corpus for serious experiments.

## Every field explained

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `task` | string | `pretrain` | Selects from-scratch pretraining. | Keep `pretrain`; use `lora` configs for adapters. |
| `output_dir` | string | `runs/nano` | Output directory. | Use a unique directory per run. |
| `run_name` | string | `nano-35m` | Run label. | Include model scale/version. |
| `data.train_file` | string | `data/samples/corpus.txt` | Required schema field but ignored for pretrain per source comment. | Leave or mirror `text_file`. |
| `data.text_file` | string | `data/samples/corpus.txt` | Raw UTF-8 corpus for pretraining. | Replace with your real corpus path. |
| `data.tokenizer` | string | `Qwen/Qwen2.5-0.5B-Instruct` | Tokenizer used to convert text to ids. | Reuse a strong BPE tokenizer unless training your own. |
| `data.max_seq_length` | int | `512` | Packed training sequence length. | Raise for longer contexts if memory permits. |
| `model.vocab_size` | int | `0` | Vocabulary size; `0` inherits from tokenizer. | Keep `0` unless using a custom tokenizer/model head. |
| `model.n_layers` | int | `8` | Transformer block count. | More layers increase capacity and cost. |
| `model.n_heads` | int | `8` | Attention query head count. | Must divide `d_model`. |
| `model.n_kv_heads` | int | `4` | Key/value head count for grouped-query attention. | Use less than `n_heads` to reduce KV cache. |
| `model.d_model` | int | `512` | Hidden width. | Main capacity and compute knob. |
| `model.d_ff` | int | `1408` | Feed-forward width. | About 2.75x `d_model` matches SwiGLU convention. |
| `model.max_seq_length` | int | `512` | Model positional/context length. | Keep aligned with `data.max_seq_length`. |
| `model.dropout` | float | `0.0` | Dropout probability. | Large pretraining corpora usually need none; small toy runs overfit anyway. |
| `model.rope_theta` | float | `10000.0` | RoPE frequency base. | Leave default unless experimenting with context scaling. |
| `model.tie_embeddings` | bool | `true` | Shares input and output embeddings. | Keep true with large vocabularies to save parameters. |
| `optim.epochs` | float | `3` | Passes over packed windows. | Toy corpus overfits quickly; serious corpora use schedule-driven steps. |
| `optim.per_device_batch_size` | int | `16` | Sequences per GPU step. | Reduce on OOM; raise only if memory allows. |
| `optim.gradient_accumulation_steps` | int | `4` | Steps accumulated before update. | Effective batch is 64 sequences. |
| `optim.learning_rate` | float | `3.0e-4` | Optimizer learning rate. | From-scratch training uses higher LR than fine-tuning. |
| `optim.lr_scheduler` | string | `cosine` | LR schedule. | Valid values are `cosine`, `linear`, `constant`. |
| `optim.warmup_ratio` | float | `0.02` | Warmup fraction. | Keep small but nonzero. |
| `optim.weight_decay` | float | `0.1` | Matrix weight decay. | Standard for pretraining; norms/biases are excluded by runner. |
| `optim.max_grad_norm` | float | `1.0` | Gradient clipping norm. | Keep to prevent spikes. |
| `optim.gradient_checkpointing` | bool | `false` | Activation recomputation. | Off because this tiny model should fit. |
| `optim.optimizer` | string | `adamw_torch` | Optimizer implementation. | Good default for from-scratch training. |
| `optim.bf16` | bool | `true` | bf16 training. | Use on supported GPUs. |
| `optim.seed` | int | `42` | Reproducibility seed. | Keep fixed for comparisons. |
| `logging_steps` | int | `10` | Log interval. | Adjust based on dataset size. |
| `save_steps` | int | `500` | Checkpoint interval. | Lower for short tests, raise for large runs. |
| `save_total_limit` | int | `2` | Retained checkpoint count. | Increase if you need more rollback points. |

## How to use it

Prerequisites, in the Kumaru folder:

```powershell
.\scripts\setup.ps1 -Gpu    # elsewhere: pip install -e ".[train]"
```

Validate:

```powershell
kumaru train configs/train/pretrain-nano.yaml --dry-run
```

Train:

```powershell
kumaru train configs/train/pretrain-nano.yaml
```

Multi-GPU entry point from the source comments:

```powershell
torchrun --nproc_per_node=4 -m kumaru.train.pretrain_nano configs/train/pretrain-nano.yaml
```

## Tuning it

1. Replace `data.text_file` with a real UTF-8 corpus.
2. Keep `data.max_seq_length` and `model.max_seq_length` aligned.
3. If OOM, reduce `optim.per_device_batch_size`, then sequence length, then model width/layers.
4. If training is unstable, lower `optim.learning_rate` and keep clipping.
5. For real training, increase corpus size before scaling model size.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Corpus not found` | `data.text_file` path is wrong. | Point it at an existing UTF-8 text file. |
| Too few tokens for `seq_length` | Corpus token count is smaller than one sequence window. | Use a larger corpus or lower sequence length. |
| Loss falls unrealistically low | Toy corpus memorization. | Use a held-out corpus and much more text. |
| OOM | Batch, sequence length, or model size too high. | Lower batch first, then sequence/model size. |
| Token ids corrupt or invalid | Incorrect packing dtype would be unsafe for large vocabs, but code uses `uint32`. | Do not change packing dtype to `uint16`. |

## Related files

- `data/samples/corpus.txt` — demonstration corpus.
- `src/kumaru/train/config.py` — training and nano model schema.
- `src/kumaru/train/data.py` — raw-text packing and `PackedTokenDataset`.
- `configs/train/lora-small.yaml` — SFT smoke profile instead of pretraining.

