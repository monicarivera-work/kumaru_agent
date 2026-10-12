# Training

> Everything you need to train this model yourself: what the options are, what each one costs, how to run it, and how to tell whether it worked.

---

## First: do you actually need to train?

Fine-tuning is the third thing to try, not the first. Work down this list and stop when the problem is solved:

| Try | Cost | Fixes |
|---|---|---|
| 1. Better system prompt | seconds | Tone, format, verbosity, refusal behaviour |
| 2. Few-shot examples in the prompt | seconds | Output structure, edge-case handling |
| 3. Retrieval (RAG) | hours | Missing facts, private documents, anything that changes |
| 4. **Fine-tuning** | **an evening** | **Consistent house style, domain register, task-specific structure** |
| 5. Pretraining | millions of GPU-hours | Nothing you need |

The distinction that matters most:

- **Fine-tuning teaches *form*.** How to answer. Format, tone, structure, register.
- **Retrieval supplies *content*.** What to answer with. Facts, documents, current data.

Fine-tuning facts into weights is expensive, slow to update, and fails silently as the world changes. If your goal is "the model should know our internal docs", build retrieval first.

---

## The two paths

| | **LoRA fine-tune** | **From-scratch pretrain** |
|---|---|---|
| Config | `configs/train/lora-qwen.yaml` | `configs/train/pretrain-nano.yaml` |
| Starts from | Qwen2.5-7B-Instruct | Random weights |
| Time on a 3090 | 20–40 min/epoch on 1k examples | seconds on the demo corpus |
| Result | A genuinely better assistant | Noise, until you supply a real corpus |
| Why do it | You need it to behave differently | You need to *understand* how training works |

**Do the first one.** Read the second one's code and run it once, because reading a training loop with no framework in the way is the fastest way to understand what every framework is hiding.

**Which model you fine-tune.** Training needs a model's original PyTorch weights, so the LoRA path tunes `Qwen/Qwen2.5-7B-Instruct`, and the adapter it produces loads only on the `transformers` backend. The GGUF files that the `qwen36` and `qwen25-7b-gguf` profiles serve are compressed copies made for inference, not training. Tuning `qwen36` itself needs more than a 24 GB card: at 4 bits its frozen weights alone are ~18 GB before any activations ([why](SCALABILITY.md#where-the-memory-actually-goes)).

---

# Part 1 — LoRA fine-tuning

## How LoRA works, in one paragraph

Freeze the base model. Alongside each target weight matrix `W`, learn two small matrices `A` (r × d) and `B` (d × r), and use `W + (α/r)·B·A` instead of `W`. `B` starts at zero, so the model begins exactly as the base model and departs from it gradually. For a 7B model at r=16, that is ~40M trainable parameters instead of 7.6B — about 0.5%.

## Why that makes it fit on your GPU

| Component | Full fine-tune | QLoRA (r=16) |
|---|---|---|
| Base weights | 15.2 GB (bf16) | 4.5 GB (4-bit NF4) |
| Gradients | 15.2 GB | 0.08 GB |
| Adam moments (2 × fp32) | 60.8 GB | 0.32 GB |
| Activations (2k seq, bs 1) | ~4 GB | ~2.5 GB (checkpointed) |
| **Total** | **~95 GB** | **~7.5 GB** |

Adam stores two fp32 moments per **trainable** parameter — 8 bytes each. That term dominates full fine-tuning, and LoRA nearly eliminates it. 95 GB needs an H100 pair; 7.5 GB leaves headroom on a 3090.

---

## Step 1 — Build your dataset

JSONL. One conversation per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Multi-turn conversations and a leading `system` message both work. The Alpaca shape (`instruction` / `input` / `output`) is auto-converted, since many public datasets use it.

See [`data/samples/chat_train.jsonl`](../data/samples/chat_train.jsonl) for 30 working examples.

### How much data

| Goal | Examples |
|---|---|
| Style, tone, output format | 100–1,000 |
| New task behaviour | 1,000–10,000 |
| New domain competence | 10,000+ |

**Quality beats quantity, decisively.** 200 carefully written examples beat 5,000 scraped ones. The model learns the *distribution* of your data including its mistakes — inconsistent formatting in, inconsistent formatting out.

### Before you train, do this

1. Read 20 random examples end to end. Would you be happy if the model produced exactly that?
2. Check consistency — same format, same voice, same length conventions.
3. Hold out 10% and do not look at it during development.

Below ~50 examples, LoRA memorizes rather than generalizes. Drop `lora.r` to 4–8 and watch eval loss closely.

---

## Step 2 — The five-minute smoke run

**Do this every time. It is not optional.**

```powershell
kumaru train configs/train/lora-small.yaml --dry-run
```

`--dry-run` validates the config, loads and encodes the data, reports how many examples survived, and writes `runs/lora-small/example_preview.txt`. **Open that file.** It shows one encoded example split into two sections:

```
=== FULL SEQUENCE (model input) ===
<|im_start|>user
What is gradient accumulation?<|im_end|>
<|im_start|>assistant
Summing gradients over several forward/backward passes...<|im_end|>

=== SUPERVISED TOKENS (where loss is computed) ===
Summing gradients over several forward/backward passes...<|im_end|>
```

If `SUPERVISED TOKENS` contains the user's question, or is empty, **stop**. Your masking is wrong and the run would teach the model the wrong task. Catching that here costs five minutes; catching it from a loss curve costs an evening.

Then run it for real:

```powershell
kumaru train configs/train/lora-small.yaml
```

Five minutes, ~2 GB VRAM, on a 0.5B model. It proves the entire path: data loads, masking is right, loss decreases, the adapter saves, and it loads back into the server.

```powershell
kumaru chat --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/lora-small/adapter
```

---

## Step 3 — The real run

```powershell
.\scripts\setup.ps1 -Gpu -Quant    # adds bitsandbytes; elsewhere: pip install -e ".[train,quant]" in the Kumaru folder

kumaru train configs/train/lora-qwen.yaml --dry-run
kumaru train configs/train/lora-qwen.yaml
```

On an RTX 3090: ~9–11 GB VRAM, roughly 20–40 minutes per epoch on 1,000 examples of ~1k tokens each.

Use the adapter:

```powershell
kumaru chat --model Qwen/Qwen2.5-7B-Instruct --adapter runs/lora-qwen/adapter
kumaru serve --model Qwen/Qwen2.5-7B-Instruct --adapter runs/lora-qwen/adapter
```

Or permanently, in a config:

```yaml
backend:
  model: Qwen/Qwen2.5-7B-Instruct
  adapter_path: runs/lora-qwen/adapter
```

These commands pass no `-c`, so they run on the built-in defaults, whose backend is `transformers`: the only backend that attaches an adapter. The other backends refuse one with `error: The <name> backend cannot attach a LoRA adapter`, rather than quietly answering with the untuned model. To serve a tuned model from `llama-server` instead, llama.cpp can convert a PEFT adapter to GGUF (`convert_lora_to_gguf.py` in the llama.cpp repository), and `llama-server` loads one with `--lora`. Kumaru does not automate that yet.

---

## Step 4 — Did it work?

```powershell
kumaru eval --suite smoke --output after.json
kumaru eval --suite smoke --model Qwen/Qwen2.5-7B-Instruct --output before.json
```

Compare per-tag pass rates. A tag that regressed tells you exactly what you damaged.

**The most common fine-tuning outcome is a model that is better at your task and worse at everything else.** That is catastrophic forgetting, and you only detect it by measuring capabilities your dataset never covered.

For a verdict you can trust, run the same benchmark questions on both models and let `compare` judge the difference question by question ([how to read it](BENCHMARKS.md#4-reading-the-numbers)):

```powershell
pip install -e ".[bench]"
kumaru bench run golden gsm8k ifeval mmlu_pro -m Qwen/Qwen2.5-7B-Instruct -o runs/bench/base
kumaru bench run golden gsm8k ifeval mmlu_pro -m Qwen/Qwen2.5-7B-Instruct --adapter runs/lora-qwen/adapter -o runs/bench/tuned
kumaru bench compare runs/bench/base runs/bench/tuned --examples 3
```

- **Your task should improve.** Add questions about it to `golden` first ([how](BENCHMARKS.md#8-your-own-golden-questions)), or the comparison cannot see what you trained for.
- **The public benchmarks should not get worse.** A significant drop on one of them is forgetting.
- **100 questions resolve only large differences**, about 10 points. To catch smaller regressions, add `-n 500` or `--full`.
- **Expect it to be slow.** The `transformers` backend answers one question at a time, unlike the batched `llama-server` runs in BENCHMARKS.md. An interrupted run resumes when you repeat its command.

---

## Hyperparameters that matter

In order of impact:

### `optim.learning_rate` — 2e-4

| Value | Effect |
|---|---|
| 5e-5 | Very conservative. Use if you see forgetting. |
| **1e-4 – 3e-4** | **The working range for LoRA.** |
| 5e-4+ | Fast, unstable, often damages the base model. |

LoRA tolerates ~10× the learning rate of a full fine-tune because `B` starts at zero — the model begins as an exact copy of the base and departs gradually.

### `optim.epochs` — 3

2–4 is the range. At 10 you are memorizing. Watch eval loss: the step where it turns upward is where you should have stopped.

### `lora.r` — 16

| r | Use for |
|---|---|
| 4–8 | Style, tone, format. Also correct under 200 examples. |
| **16–32** | **Default. New vocabulary, task structure.** |
| 64–128 | Genuinely new capability, 10k+ examples only. |

Keep `alpha = 2 × r`. The update is scaled by `α/r`, so that convention holds the effective learning rate constant as you change rank — you do not have to retune the LR every time.

### `optim.gradient_accumulation_steps` — 16

Effective batch = `per_device_batch_size × gradient_accumulation_steps × num_gpus`. Target 16–64. This is how you get a large effective batch with a small memory footprint.

### `data.max_seq_length` — 2048

Examples longer than this are **dropped, not truncated**. Truncating mid-answer teaches the model to stop mid-sentence. If the dry run reports more than 10% dropped, raise this (and expect more VRAM) or shorten your examples.

---

## Reading the training log

```
step 40/120 | loss 1.8234 | ppl 6.2 | lr 1.94e-04 | gnorm 0.87 | 3,240 tok/s | eta 4m
```

| Field | Healthy | Warning sign |
|---|---|---|
| `loss` | Falls smoothly | Flat → LR too low or data broken. Spiky → LR too high. |
| `ppl` | exp(loss), falls with it | — |
| `lr` | Warms up, then decays | Not following the schedule → config bug |
| `gnorm` | Stable band after warmup | Repeated spikes → lower LR or clip harder |
| `tok/s` | Steady | Sudden drop → dataloader became the bottleneck |

**Watch `gnorm` most closely.** A gradient-norm spike precedes a NaN loss by several steps. It is your earliest warning that something is going wrong.

### The overfit test

Before any long run, train on 10 examples for 50 steps. Loss should approach zero. **If it cannot overfit 10 examples, your pipeline is broken, not your hyperparameters.** This five-minute check has saved more GPU-hours than any other habit.

---

## Out of memory? Change these in order

1. `optim.gradient_checkpointing: true` — ~40% less activation memory, ~30% slower
2. `load_in_4bit: true` — base weights from 15.2 GB to 4.5 GB
3. `data.max_seq_length: 1024` — activation memory scales with sequence length
4. `optim.per_device_batch_size: 1`, raise `gradient_accumulation_steps` to compensate
5. `optim.optimizer: paged_adamw_8bit` — 8-bit optimizer state, pages to CPU under pressure
6. `lora.target_modules: [q_proj, k_proj, v_proj, o_proj]` — attention only, drops the MLP adapters
7. Use a smaller base model

---

# Part 2 — Pretraining from scratch

## Read this before running it

This will **not** produce a useful assistant. The shipped corpus is ~1,700 tokens — the model memorizes it in three steps and emits noise. That is deliberate: it makes every mechanism concrete and observable in seconds, and it lets you *watch* overfitting happen on a corpus small enough to read in full. The profile logs every `logging_steps: 5` steps, so a three-step run prints no loss lines; set `logging_steps: 1` and raise `optim.epochs` to see the loss collapse step by step.

```powershell
kumaru train configs/train/pretrain-nano.yaml --dry-run
kumaru train configs/train/pretrain-nano.yaml
```

Point `data.text_file` at a real corpus (any UTF-8 text file, any size) to train something meaningful, and give that run a new `output_dir`. The first run packs the corpus into `<output_dir>/tokens.bin`, streaming the token ids to disk one chunk at a time: RAM stays at about 0.2 GB whatever the corpus size, and the file takes 4 bytes per token (about 8 GB for the 2B tokens this model wants). Later runs with the same `output_dir` reuse that file without reading the text again, so a changed `text_file` or `tokenizer` in an old `output_dir` would still train on the old tokens. Packing ran at about 1.4 MB of text per second in local testing, roughly 1.5 hours for 8 GB; `--dry-run` packs and stops, so you can do it up front.

The sequence length is `model.max_seq_length` (256 in the profile). Pretraining ignores `data.max_seq_length`, which only limits LoRA examples.

### Where the parameters go

The default config is ~101M parameters, and the split is instructive:

| Component | Parameters | Share |
|---|---|---|
| Embedding table (152k vocab × 512 d_model) | ~78M | 77% |
| 8 transformer blocks | ~24M | 23% |

**At small scale, the vocabulary dominates the model.** That is why `tie_embeddings: true` matters so much here — untying would add another 78M parameters that do almost no reasoning work. The ratio inverts as you scale: at 7B with the same vocabulary, embeddings are ~4% of the model.

`estimate_parameters()` in `model_nano.py` prints this breakdown without building the model, so you can size a run before spending GPU-hours on it.

## What you get to see

[`src/kumaru/train/model_nano.py`](../src/kumaru/train/model_nano.py) is a complete modern transformer in ~250 readable lines. It is the *same architecture* as Llama 3, Qwen 2.5, and Mistral — RMSNorm, rotary position embeddings, grouped-query attention, SwiGLU, pre-norm residuals — with smaller numbers. Scaling it to 7B changes `n_layers`, `d_model`, and `n_heads`. Nothing structural changes.

[`src/kumaru/train/pretrain_nano.py`](../src/kumaru/train/pretrain_nano.py) has no `Trainer` class. The optimizer step, LR schedule, gradient accumulation, clipping, mixed precision, and checkpointing are all written out, because these are exactly the mechanisms you need to understand when a run will not converge.

## Scaling the same loop

```powershell
# 1 GPU
kumaru train configs/train/pretrain-nano.yaml

# 8 GPUs, same file, same code
torchrun --nproc_per_node=8 -m kumaru.train.pretrain_nano configs/train/pretrain-nano.yaml

# multiple nodes
torchrun --nnodes=4 --nproc_per_node=8 --rdzv_endpoint=$MASTER:29500 -m kumaru.train.pretrain_nano cfg.yaml
```

Data-parallel training works because gradients are linear: each rank computes gradients on its shard, an all-reduce averages them, every rank applies an identical update. Mathematically equivalent to one giant batch.

Before the first `torchrun` launch on a large corpus, pack it with `kumaru train <config> --dry-run` (same `output_dir`). Under `torchrun`, rank 0 packs while the other ranks wait at a barrier, and PyTorch abandons that wait after its default process-group timeout: 10 minutes with NCCL, 30 with gloo. The `torchrun` entry point also skips the `kumaru train` preflight checks.

## Sizing a real run

Chinchilla-optimal is roughly **20 training tokens per parameter**:

| Model | Optimal tokens | Corpus size | On 8× A100 |
|---|---|---|---|
| 101M (the default) | 2B | ~8 GB text | ~1 hour |
| 350M | 7B | ~30 GB | hours |
| 1B | 20B | ~90 GB | ~2 days |
| 7B | 140B | ~600 GB | ~3 months |

To scale the config up, edit the `model` block in `configs/train/pretrain-nano.yaml`:

```yaml
model:
  n_layers: 24      # was 8
  n_heads: 16       # was 8
  n_kv_heads: 4
  d_model: 1024     # was 512
  d_ff: 2816        # keep ≈ 2.75 × d_model
  max_seq_length: 1024
```

`estimate_parameters()` in `model_nano.py` gives you the parameter count without building the model, so you can size a run before spending GPU-hours on it.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Loss is flat | LR too low, or labels all masked | Check `example_preview.txt`; raise LR |
| Loss is NaN | fp16 overflow, or LR too high | Use bf16; lower LR; check `gnorm` history |
| Loss drops then eval loss rises | Overfitting | Fewer epochs, lower `r`, more data |
| CUDA OOM | Too much memory requested | Follow the OOM list above, in order |
| "Every example was dropped" | `max_seq_length` too small | Raise it, or shorten examples |
| Adapter has no visible effect | Not loaded, or LR too low | Confirm `adapter_path` in `kumaru config`; check train loss actually fell |
| `error: The llamacpp_server backend cannot attach a LoRA adapter` | The profile serves a GGUF file (e.g. `-c qwen36`); only `transformers` loads PEFT adapters | Drop `-c` and pass the adapter's base model, as in Step 3; for GGUF, convert the adapter first (Step 3) |
| Model got worse at everything | Catastrophic forgetting | Lower LR, fewer epochs, mix in 10–20% general data |
| `bitsandbytes` import error | Optional dep missing | `.\scripts\setup.ps1 -Quant` (or `pip install -e ".[quant]"` in the Kumaru folder), or set `load_in_4bit: false` |

---

## Related reading

- [`howto/src/kumaru/train/data.md`](howto/src/kumaru/train/data.md) — loss masking and packing, in detail
- [`howto/src/kumaru/train/finetune_lora.md`](howto/src/kumaru/train/finetune_lora.md) — the LoRA trainer
- [`howto/src/kumaru/train/model_nano.md`](howto/src/kumaru/train/model_nano.md) — the architecture, component by component
- [`SCALABILITY.md`](SCALABILITY.md) §5 — the full training ladder
