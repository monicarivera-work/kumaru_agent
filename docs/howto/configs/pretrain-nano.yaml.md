# `configs/train/pretrain-nano.yaml`

> Tiny from-scratch language-model pretraining profile for observing the full training loop on a readable corpus.

**Read this when:** you want to validate or study pretraining mechanics, token packing, small transformer architecture, or overfitting behavior.

---

## What it does

This config trains a small transformer from random initialization using raw text from `data/samples/corpus.txt`. It uses the Qwen2.5 0.5B tokenizer to pack the corpus into one flat token stream (`runs/nano/tokens.bin`), cuts that stream into 256-token windows, and trains a roughly 101M-parameter model under `runs/nano`: about 78M in the tied embedding table and about 24M in the eight transformer blocks.

It will not produce a useful assistant. The sample corpus is intentionally tiny (about 1,700 tokens, so six windows and three optimizer steps), and loss will fall because the model memorizes.

## When to use this profile

Use it for mechanism tests and education on one GPU. The comments note that proper Chinchilla-style training for 101M parameters wants about 2B tokens (~8 GB of text), far beyond the sample file. Point `data.text_file` at a real UTF-8 corpus for serious experiments, and give that run its own `output_dir` (see [Tuning it](#tuning-it)).

## Every field explained

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `task` | string | `pretrain` | Selects from-scratch pretraining. | Keep `pretrain`; use `lora` configs for adapters. |
| `output_dir` | string | `runs/nano` | Output directory. | Use a unique directory per run. |
| `run_name` | string | `nano-101m` | Run label. | Include model scale/version. |
| `data.train_file` | string | `data/samples/corpus.txt` | Required by the schema. Pretraining reads it only as the corpus when `data.text_file` is empty or null. | Leave it mirroring `text_file`. |
| `data.text_file` | string | `data/samples/corpus.txt` | Raw UTF-8 corpus for pretraining. | Replace with your real corpus path, and change `output_dir` with it. |
| `data.tokenizer` | string | `Qwen/Qwen2.5-0.5B-Instruct` | Tokenizer used to convert text to ids. | Reuse a strong BPE tokenizer unless training your own. |
| `model.vocab_size` | int | `0` | Vocabulary size; `0` inherits from tokenizer. | Keep `0` unless using a custom tokenizer/model head. |
| `model.n_layers` | int | `8` | Transformer block count. | More layers increase capacity and cost. |
| `model.n_heads` | int | `8` | Attention query head count. | Must divide `d_model`. |
| `model.n_kv_heads` | int | `4` | Key/value head count for grouped-query attention. | Use less than `n_heads` to reduce KV cache. |
| `model.d_model` | int | `512` | Hidden width. | Main capacity and compute knob. |
| `model.d_ff` | int | `1408` | Feed-forward width. | About 2.75x `d_model` matches SwiGLU convention. |
| `model.max_seq_length` | int | `256` | Length of every packed training window and the model's maximum context. The only sequence length pretraining reads. | `256` keeps the demo corpus at six windows; use 512-2048 on a real corpus if memory permits. Changing it needs no repack. |
| `model.dropout` | float | `0.0` | Dropout probability. | Large pretraining corpora usually need none; small toy runs overfit anyway. |
| `model.rope_theta` | float | `10000.0` | RoPE frequency base. | Leave default unless experimenting with context scaling. |
| `model.tie_embeddings` | bool | `true` | Shares input and output embeddings. | Keep true with large vocabularies to save parameters. |
| `optim.epochs` | float | `3` | Passes over packed windows. | Toy corpus overfits quickly; serious corpora use schedule-driven steps. |
| `optim.per_device_batch_size` | int | `4` | Sequences per GPU per micro-batch. | Reduce on OOM; raise only if memory allows. |
| `optim.gradient_accumulation_steps` | int | `4` | Micro-batches accumulated before each optimizer step. | Effective batch is 4 x 4 = 16 sequences = 4,096 tokens per step on one GPU, times the GPU count under `torchrun`. |
| `optim.learning_rate` | float | `3.0e-4` | Optimizer learning rate. | From-scratch training uses higher LR than fine-tuning. |
| `optim.lr_scheduler` | string | `cosine` | LR schedule. | Valid values are `cosine`, `linear`, `constant`. |
| `optim.warmup_ratio` | float | `0.02` | Warmup fraction. | Keep small but nonzero. |
| `optim.weight_decay` | float | `0.1` | Matrix weight decay. | Standard for pretraining; norms/biases are excluded by runner. |
| `optim.max_grad_norm` | float | `1.0` | Gradient clipping norm. | Keep to prevent spikes. |
| `optim.gradient_checkpointing` | bool | `false` | Activation recomputation. | Off because this tiny model should fit. |
| `optim.optimizer` | string | `adamw_torch` | Optimizer implementation. | Good default for from-scratch training. |
| `optim.bf16` | bool | `true` | bf16 training. | Use on supported GPUs. |
| `optim.seed` | int | `42` | Reproducibility seed. | Keep fixed for comparisons. |
| `logging_steps` | int | `5` | Optimizer steps between loss lines; a line prints when the step number is a multiple of it. | The demo runs only three steps, so it prints no loss lines at `5`; set `1` to watch every step. |
| `save_steps` | int | `500` | Checkpoint interval. | Lower for short tests, raise for large runs. |
| `save_total_limit` | int | `2` | Retained checkpoint count. | Increase if you need more rollback points. |

### One sequence-length setting

Pretraining reads only `model.max_seq_length`. It is both the length of each packed training window and the model's maximum context. `data.max_seq_length` is a LoRA setting (supervised examples longer than it are dropped), and `run_pretrain()` never reads it, so this profile leaves it out; adding it would change nothing. The resolved `runs/nano/train_config.yaml` still lists `data.max_seq_length: 2048`, because the saved copy records every schema field with its default.

## How to use it

Prerequisites, in the Kumaru folder:

```powershell
.\scripts\setup.ps1 -Gpu    # elsewhere: pip install -e ".[train]"
```

Validate:

```powershell
kumaru train configs/train/pretrain-nano.yaml --dry-run
```

A dry run checks the config and data, packs the corpus into `runs/nano/tokens.bin` if that file does not exist yet, builds the dataset and model plan, and stops before training. On a large corpus the packing is the slow part, and it is done once.

Train:

```powershell
kumaru train configs/train/pretrain-nano.yaml
```

Multi-GPU entry point from the source comments:

```powershell
torchrun --nproc_per_node=4 -m kumaru.train.pretrain_nano configs/train/pretrain-nano.yaml
```

Under `torchrun`, rank 0 packs the corpus while the other ranks wait at a barrier, and PyTorch aborts that wait after its default process-group timeout (10 minutes with NCCL, 30 with gloo). At the packing speed measured below, a corpus larger than about 800 MB outlasts the NCCL limit and one larger than about 2.5 GB outlasts gloo's. So pack first with the single-process dry run above, using the same `output_dir`, then launch `torchrun`. The `torchrun` entry point also skips the `kumaru train` preflight, which is a second reason to dry-run first.

## Tuning it

1. Replace `data.text_file` with a real UTF-8 corpus and set a new `output_dir` at the same time. The first run packs the corpus into `<output_dir>/tokens.bin`; every later run with that `output_dir` reuses the file without reading `text_file` again. A new corpus or `data.tokenizer` in an old `output_dir` therefore trains on the old tokens, so use a fresh directory or delete `tokens.bin`.
2. Packing streams to disk: it tokenizes 1,000,000 characters at a time and appends each chunk's token ids to the file, so RAM stays at about 0.2 GB however large the corpus is. The file takes 4 bytes per token (about 8 GB for 2B tokens), on top of the text itself. Packing is sequential; it ran at about 1.4 MB of text per second in local testing, so an 8 GB corpus takes about 1.5 hours.
3. Set the sequence length with `model.max_seq_length` only (see [One sequence-length setting](#one-sequence-length-setting)). Changing it needs no repack: the same `tokens.bin` is cut into windows of the new length.
4. If OOM, reduce `optim.per_device_batch_size`, then `model.max_seq_length`, then model width/layers.
5. If training is unstable, lower `optim.learning_rate` and keep clipping.
6. For real training, increase corpus size before scaling model size.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Training data not found: ... (data.text_file)` | `data.text_file` path is wrong. Preflight checks it before the tokenizer loads; with `text_file` unset it checks `data.train_file` instead. | Point it at an existing UTF-8 text file. Paths are resolved from the directory you run in. |
| `Corpus not found` | Same wrong path, reached through the `torchrun` entry point, which skips preflight. | Fix the path; dry-run with `kumaru train` first. |
| Trains on the wrong text after changing `text_file` or `tokenizer` | An existing `<output_dir>/tokens.bin` was reused. | Use a new `output_dir`, or delete `tokens.bin` to repack. |
| `tokens.bin.tmp` left in `output_dir` | Packing was killed before it finished. | Harmless: the next run packs again and overwrites it, and a partial file is never renamed to `tokens.bin`. Delete it to reclaim the disk space. |
| `... produced zero tokens` | The corpus file is empty. | Point `data.text_file` at a file with text in it. |
| Too few tokens for `seq_length` | Corpus token count is smaller than one sequence window. | Use a larger corpus or lower `model.max_seq_length`. |
| Collective or barrier timeout under `torchrun` on the first run | Rank 0 was still packing a large corpus when the other ranks' wait timed out. | Pack first with `kumaru train <config> --dry-run`, then launch `torchrun` with the same `output_dir`. |
| Loss falls unrealistically low | Toy corpus memorization. | Use a held-out corpus and much more text. |
| OOM | Batch, sequence length, or model size too high. | Lower batch first, then sequence/model size. |
| Token ids corrupt or invalid | Incorrect packing dtype would be unsafe for large vocabs, but code uses `uint32`. | Do not change packing dtype to `uint16`. |

## Related files

- `data/samples/corpus.txt` — demonstration corpus.
- `src/kumaru/train/config.py` — training and nano model schema.
- `src/kumaru/train/data.py` — raw-text packing and `PackedTokenDataset`.
- `configs/train/lora-small.yaml` — SFT smoke profile instead of pretraining.

