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

**Distillation** ([Part 3](#part-3--distillation)) is the first path with generated data: a large model's verified answers teach a small, fast one.

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
- **Expect it to be slow.** With the default profile, the `transformers` backend answers one question at a time, unlike the batched `llama-server` runs in BENCHMARKS.md. It batches only greedy requests, and only when `backend.max_batch_size` is above 1, as in [`configs/student.yaml`](howto/configs/student.yaml.md). An interrupted run resumes when you repeat its command.

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

# Part 3 — Distillation

## What distillation is, in one paragraph

A large, slow model, the *teacher*, answers GSM8K's 7,473 training questions. A grader keeps only its correct answers, and a small, fast model, the *student*, is LoRA-tuned on them with the Part 1 trainer. The student learns the teacher's worked solutions, step by step, not just its final numbers. This is sequence-level knowledge distillation (Kim & Rush, 2016); keeping only graded-correct answers makes it rejection-sampling fine-tuning (Yuan et al., 2023). Here the teacher is Qwen3.6-35B-A3B (the `qwen36` profile: 95% on a 100-question GSM8K run, [BENCHMARKS.md §6.1](BENCHMARKS.md#61-the-truncation-confound)), and the student is Qwen2.5-1.5B-Instruct.

**Why it matters.** You pay for the big model's compute once; the small one then answers every question at its own speed. Whatever later makes the teacher better (a larger model, more answers per question) reaches the student through the same pipeline, and any task with a grader can be distilled the same way.

## The experiment: two students and a control

Practice on 7,000 GSM8K-style answers could help a student whoever wrote them. So the experiment trains two students that differ only in their data, and scores both against the untrained model:

| Arm | Recipe | Trained on | Scored as |
|---|---|---|---|
| Base | (none) | nothing | `student-base` |
| Teacher | [`distill-gsm8k.yaml`](howto/configs/distill-gsm8k.yaml.md) | the teacher's verified answers | `student-teacher` |
| Gold, the control | [`distill-gsm8k-gold.yaml`](howto/configs/distill-gsm8k-gold.yaml.md) | GSM8K's human-written solutions to the same questions | `student-gold` |

Teacher against base says whether training helped. Teacher against gold says whether the teacher was the reason. A test fails if the two recipes differ in anything but their data, output folder and run name. A third recipe, [`distill-gsm8k-0.5b.yaml`](howto/configs/distill-gsm8k-0.5b.yaml.md), is the teacher arm on Qwen2.5-0.5B: a cheap pilot that measures time and memory before the real runs.

Run every step from the Kumaru folder, after `.\scripts\setup.ps1 -Gpu` and `pip install -e ".[bench]"`, with the [`qwen36` profile](howto/configs/qwen36.yaml.md) working. Only Step 2 runs without the GPU.

## Step 1 — The teacher answers the training questions

```powershell
kumaru bench run gsm8k_train -c qwen36 --no-think -n 200 --max-tokens 1024 --context 8192 -o runs\distill\teacher-qwen36
kumaru bench run gsm8k_train -c qwen36 --no-think --full --max-tokens 1024 --context 8192 -o runs\distill\teacher-qwen36
```

- **`gsm8k_train`, never `gsm8k`.** `gsm8k` is the test split the student will be scored on. `kumaru distill` refuses runs of it, because training on test questions inflates exactly the score being measured.
- **`--no-think`.** With thinking on, the student would see each answer without the reasoning that produced it, so `kumaru distill` refuses thinking runs by default.
- **The 200-question pilot** checks the setup: `kumaru bench report runs\distill\teacher-qwen36` shows its accuracy and how many answers were cut off. The second command continues the same run and keeps those 200 answers.
- **16 hours**, measured: 7,473 answers of 365 tokens on average, at about 47 tokens per second, beside a game, with 18 expert layers in system RAM as in [BENCHMARKS.md §9](BENCHMARKS.md#9-run-mechanics). The pilot took 28 minutes and the rest 34 times as long, so on your machine the pilot gives the estimate. Stop it whenever you need the GPU; repeating the command resumes it.

## Step 2 — Build the two datasets

```powershell
kumaru distill build runs\distill\teacher-qwen36 -o runs\distill\data-teacher
kumaru distill gold -o runs\distill\data-gold --match runs\distill\data-teacher
```

Seconds, on the CPU. Each command writes `train.jsonl`, `val.jsonl` and `manifest.json`, and prints what it kept, what it dropped and why. Eleven checks keep only answers that are correct, complete and clean ([all of them](howto/src/kumaru/train/distill.md#2-the-checks-in-order)). Three shape the experiment:

- **Decontamination.** An answer is dropped when its question or its text shares a run of 13 words with a test question (the test of Brown et al., 2020). In the gold build, that removes 4 of the 7,473 questions.
- **Arithmetic.** A right final number reached through a wrong step would teach the wrong step, so every simple `a op b = c` is checked. It flags 24 of GSM8K's own human-written solutions, [17 of them real errors](howto/src/kumaru/train/distill.md#5-the-arithmetic-check-wrong_steps).
- **Matching.** `--match` gives the gold arm exactly the teacher arm's questions, each on the same side of the train/validation split.

## Step 3 — The pilot

```powershell
kumaru train configs\train\distill-gsm8k-0.5b.yaml --dry-run
kumaru train configs\train\distill-gsm8k-0.5b.yaml
```

58 minutes, measured. Afterwards, `runs\distill\qwen0.5b-teacher\metrics.json` holds two measurements: `train_seconds`, and `peak_gpu_memory_gb`, the most GPU memory the run held (21.6 GB). Each 1.5B arm took about twice as long but held less, because it takes one row per forward pass and the pilot two ([why that matters](howto/configs/distill-gsm8k.yaml.md#cost)). To read the pilot's own score, see [its how-to](howto/configs/distill-gsm8k-0.5b.yaml.md#reading-the-result).

## Step 4 — Score the untrained student

```powershell
kumaru bench run gsm8k -c student --full --max-tokens 1024 --label student-base -o runs\distill\eval-base
```

The [`student` profile](howto/configs/student.yaml.md) decodes greedily, 32 questions at a time, so all 1,319 test questions take about 20 minutes (measured), in about 5 GB (estimated). Its first run downloads Qwen2.5-1.5B-Instruct, about 3 GB. Measure this baseline rather than quoting one: the Qwen2.5 report gives about 73% for this model, but prompts and grading differ between setups.

## Step 5 — Train both arms

```powershell
kumaru train configs\train\distill-gsm8k.yaml --dry-run
kumaru train configs\train\distill-gsm8k.yaml
kumaru train configs\train\distill-gsm8k-gold.yaml
```

- **Check the dry run first**, as in [Part 1](#step-2--the-five-minute-smoke-run): the `SUPERVISED TOKENS` section of `runs\distill\qwen1.5b-teacher\example_preview.txt` must hold the answer and nothing of the question.
- **Each run ends on its best checkpoint**, the one with the lowest validation loss, saved as `adapter\` in its `output_dir`.
- **Out of memory, or steps far slower than ~17 seconds:** the arms already take one row per forward pass, and peaked at 13.8 GB (teacher) and 6.4 GB (gold). Free the GPU: close any game or other program using it. On Windows, running short shows as a slowdown rather than an error, because the driver spills into shared system memory (*Shared GPU memory* in Task Manager). Lowering `data.max_seq_length` in all three recipes also saves memory, but drops the longest answers ([details](howto/configs/distill-gsm8k.yaml.md#tuning-it)).

## Step 6 — Score both students and compare

```powershell
kumaru bench run gsm8k -c student --full --max-tokens 1024 --adapter runs\distill\qwen1.5b-teacher\adapter --label student-teacher -o runs\distill\eval-teacher
kumaru bench run gsm8k -c student --full --max-tokens 1024 --adapter runs\distill\qwen1.5b-gold\adapter --label student-gold -o runs\distill\eval-gold
kumaru bench compare runs\distill\eval-base runs\distill\eval-teacher
kumaru bench compare runs\distill\eval-base runs\distill\eval-gold
kumaru bench compare runs\distill\eval-gold runs\distill\eval-teacher -e 5
```

`--label` names each run in the tables; without it, all three would be called `student-default`. The comparisons ask, in order: did training on the teacher's answers help, did training on human answers help, and did the teacher's answers teach more than human ones. `-e 5` also prints up to five questions the two runs answered differently.

## What it costs

Measured on an RTX 3090 (24 GB) under Windows 11, in [the 2026-10-22 run](#results-2026-10-22), except where marked.

| Step | Time | GPU memory |
|---|---|---|
| 1. The teacher's answers | 16.0 h, beside a game | the `qwen36` profile's |
| 2. Both datasets | seconds | none (CPU) |
| 3. The pilot, 0.5B | 58 min | 21.6 GB, at 2 rows per pass |
| 4. Scoring the untrained student | 20 min | ~5 GB (estimated) |
| 5. Teacher arm, 1.5B | 117 min | 13.8 GB |
| 5. Gold arm, 1.5B | 134 min with a game sharing the GPU; about 2 h without | 6.4 GB |
| 6. Scoring each trained student | 78 min (teacher arm), 29 min (gold arm) | ~5 GB (estimated) |

- **Memory follows the rows per forward pass, not the model.** At every position the model scores all 151,936 tokens of its vocabulary, in fp32: about 1.5 MB per token, so one long answer needs ~2 GB for the loss alone. That is why the 0.5B pilot, at two rows per pass, held more than the 1.5B arms at one. The figures are what PyTorch reserved, its cache included, which is well above what the tensors needed: about 6 GB in the teacher arm, estimated ([the arithmetic](howto/configs/distill-gsm8k.yaml.md#cost)).
- **Time follows the passes.** At one row per pass, a step of 32 passes took ~16.6 seconds in both arms, though the teacher's rows are twice as long: the cost of a pass, not its length, sets the pace. The pilot's 16 passes per step took half that.
- **Scoring with an adapter is slower.** The adapter runs beside the base weights at every token, unmerged: about 2.6 times the base model's time per token. The teacher arm's answers are also longer, and more of them run to the 1,024-token limit. `$env:KUMARU_BACKEND__EXTRA__MERGE_ADAPTER = 'true'` folds the adapter into the weights at load ([`backend.extra`](howto/configs/default.yaml.md#backendextra)), which should remove most of that; it was not tried here.

## Reading the result

`kumaru bench compare` pairs two runs question by question: McNemar's exact test gives the p-value, and a paired bootstrap the interval ([how to read it](BENCHMARKS.md#4-reading-the-numbers)). At 1,319 questions, about 3 points is the smallest difference it reliably detects. Before concluding, train three seeds per arm: copy each recipe and change `optim.seed`, `output_dir` and `run_name`.

Read the teacher arm against base first: only a gain there is a gain. The arms against each other then say where it came from.

| Teacher arm against base | Teacher arm against gold | What it means |
|---|---|---|
| Above | Above | Distillation worked: the teacher's answers taught more than human ones. |
| Above | No difference | Practice on GSM8K-style questions helped; the teacher added nothing measurable. |
| Any | Below | Human solutions were better data for this student. Next: shorter teacher answers, or a mix of both datasets. |
| No difference | Above | No gain. The arms differ because the gold arm lost points, not because the teacher arm gained any. [The 2026-10-22 run](#results-2026-10-22) ended here. |
| No difference | No difference | Training changed nothing measurable. Check `example_preview.txt`, and that the training loss fell. |
| Below | Above or no difference | Training on the teacher's answers hurt. Look at the `-e 5` examples, and at how many answers were cut off (`kumaru bench report`). The [0.5B pilot](howto/configs/distill-gsm8k-0.5b.yaml.md#reading-the-result) fell this way against its untrained model. |

"Above" and "below" mean a difference with a p-value under 0.05; "no difference", one without.

- **Never compare the arms' validation losses.** Each is measured on its own kind of answer; only the GSM8K scores compare.
- **The arms differ in length by design.** The teacher arm trains on 2.26 million answer tokens, the gold arm on 0.62 million: 335 against 92 per answer. So a teacher win cannot say whether the teacher reasons better or only writes out more steps. A gold win is plausible too: Li et al. (2025) found that students of 3B parameters or fewer often learn more from short reasoning than from a strong teacher's long reasoning, and best from a mix of both ([details](howto/configs/distill-gsm8k-gold.yaml.md#cost-and-length)).

## Results, 2026-10-22

Every step above, run once on an RTX 3090 (24 GB) under Windows 11 with a Ryzen 7 5700X, at commit `fd2dadd`, with the recipes as they are now: only their batch split differed from that commit's. One seed per arm.

**Data.** The teacher answered all 7,473 training questions in 16.0 hours, 95.3% of them correctly. `kumaru distill build` kept 7,089 answers (6,751 for training, 338 for validation); it dropped 352 wrong, 21 unfinished, 7 that failed the arithmetic check and 4 as too close to a test question. `kumaru distill gold` matched 7,069 of those questions (6,732 and 337); for the other 20, GSM8K's own solution failed the arithmetic check.

**Training.** Each run took 422 steps of 32 rows.

| Run | Rows per pass × passes | Time | Peak GPU memory | Validation loss: first evaluation → kept checkpoint |
|---|---|---|---|---|
| Pilot, 0.5B, teacher's answers | 2 × 16 | 58 min | 21.6 GB | 0.331 → 0.285, last step |
| Teacher arm, 1.5B | 1 × 32 | 117 min | 13.8 GB | 0.260 → 0.221, last step |
| Gold arm, 1.5B | 1 × 32 | 134 min | 6.4 GB | 0.422 → 0.410, step 200 (0.411 at the last) |

The teacher arm's validation loss fell at every evaluation. The gold arm's was lowest after about one epoch, so the run kept that checkpoint. Until step ~175, its steps took as long as the teacher arm's; then a game took about 5 GB of the GPU, and they slowed by about a quarter.

**Scores.** All 1,319 GSM8K test questions, greedy, `--max-tokens 1024`:

| Student | Accuracy (95% CI) | Cut off | Tokens per answer | Scoring time |
|---|---|---|---|---|
| 1.5B, untrained | 72.3 (69.8–74.6) | 6 | 293 | 20 min |
| 1.5B, teacher arm | 71.9 (69.4–74.2) | 47 | 356 | 78 min |
| 1.5B, gold arm | 61.2 (58.5–63.8) | 9 | 97 | 29 min |
| 0.5B, untrained | 50.3 (47.6–53.0) | 17 | 307 | 21 min |
| 0.5B, teacher's answers (the pilot) | 44.7 (42.0–47.3) | 63 | 356 | 54 min |

| Comparison, A → B | B − A, points (95% CI) | Only A right / only B right | p | Verdict |
|---|---|---|---|---|
| Untrained → teacher arm | −0.4 (−2.9 to +2.1) | 148 / 143 | 0.815 | no clear difference |
| Untrained → gold arm | −11.1 (−13.8 to −8.3) | 258 / 112 | 2.19e-14 | untrained better |
| Gold arm → teacher arm | +10.7 (+7.8 to +13.6) | 128 / 269 | 1.24e-12 | teacher arm better |
| 0.5B untrained → 0.5B pilot | −5.6 (−8.3 to −2.9) | 206 / 132 | 6.74e-05 | untrained better |

**What the answers show.**

- **Each student took the format of its training answers.** The teacher arm opens steps with `**Step` headers (1,120 of 1,319 answers) and ends with an `Answer:` line (all 1,272 it finished), where the untrained model often boxes its answer instead (`\boxed`, 784 answers). The gold arm writes GSM8K's terse plain text: 82 tokens at the median, and an `Answer:` line in all 1,310 it finished.
- **The teacher arm ran long.** 47 of its answers hit the token limit (the untrained model's: 6), 12 of them repeating themselves; the untrained model got 22 of those 47 questions right. On the 1,266 questions where neither run was cut off, the untrained model got 931 right and the teacher arm 944: a difference picked after the fact, and within noise.
- **The gold arm stopped reaching the answer.** The questions only the untrained model got right, sorted by what went wrong:

| Student | Only the untrained model right | Cut off | Right number in the working, wrong final answer | Right number nowhere |
|---|---|---|---|---|
| 1.5B, teacher arm | 148 | 22 | 39 | 87 |
| 1.5B, gold arm | 258 | 4 | 36 | 218 |
| 0.5B, the pilot | 206 | 19 | 48 | 139 |

**What it means.**

- **Distillation added nothing measurable at 1.5B.** The teacher arm learned the teacher's format, not more math: fine-tuning teaches [form](#first-do-you-actually-need-to-train). A likely reason: Qwen2.5-Instruct's own fine-tuning already included step-by-step math solutions, filtered for correct answers by rejection sampling (Qwen2.5 technical report, 2024), much like the data this pipeline makes.
- **The gold arm lost 11 points.** Trained on GSM8K's terse solutions, the student learned to skip the working: its answers are a third as long, and in 218 of its 258 losses the right number appears nowhere.
- **So the teacher arm's lead over gold is not a teacher gain.** It is mostly long worked answers beating terse ones, as [the length caveat](#reading-the-result) warns.
- **The 0.5B lost 5.6 points where the 1.5B held even.** That fits Li et al. (2025): students of 3B parameters or fewer gain less from a strong teacher. The 0.5B also took the teacher's format (`**Step` in 1,134 answers) and was cut off more often (63 answers against 17).

**Caveats.** One seed per arm, where this guide recommends three. The 11-point gaps are far larger than a change of seed usually moves a score; teacher arm against base could move by a point or two. About 3 points is the smallest difference 1,319 questions reliably detect, so a smaller gain would go unseen.

**Decision.** Training on GSM8K stopped here. The pipeline works end to end, from the teacher's answers to the comparison, but this student, already tuned on math, gained nothing measurable from it on GSM8K. If you take it further, a task the student was not already tuned on is the more telling test ([below](#where-to-take-it-next), item 3).

## Where to take it next

Roughly in order of cost:

1. **More answers per question.** Run the teacher again with another `--seed` (it samples at temperature 0.7), and pass both runs to `kumaru distill build`. Yuan et al. (2023) found that more distinct correct solutions help, and help weaker students most.
2. **A bigger student.** Change `base_model` in all three recipes, and `backend.model` in `configs/student.yaml`; a test checks that they agree.
3. **Another task.** Any benchmark with a grader and a training split can be the source ([how](howto/src/kumaru/train/distill.md#how-to-extend-it)).
4. **Logit and on-policy distillation.** The student learns the teacher's full next-token probabilities, or practises on its own answers while the teacher scores them. Both change the trainer, not just the data ([how](howto/src/kumaru/train/distill.md#how-to-extend-it)).

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
| `error: <dir> holds answers to gsm8k, an evaluation set` | The teacher answered the test split | Run it on `gsm8k_train` ([Part 3, Step 1](#step-1--the-teacher-answers-the-training-questions)) |
| Size-mismatch error when scoring an adapter | The adapter was trained on another base model, e.g. a 0.5B adapter on the 1.5B student | Pass the adapter's base model with `-m` |

---

## Related reading

- [`howto/src/kumaru/train/data.md`](howto/src/kumaru/train/data.md) — loss masking and packing, in detail
- [`howto/src/kumaru/train/finetune_lora.md`](howto/src/kumaru/train/finetune_lora.md) — the LoRA trainer
- [`howto/src/kumaru/train/distill.md`](howto/src/kumaru/train/distill.md) — distillation data: every check, flag and file
- [`howto/configs/distill-gsm8k.yaml.md`](howto/configs/distill-gsm8k.yaml.md) — the distillation recipes, setting by setting, and what they cost
- [`howto/src/kumaru/train/model_nano.md`](howto/src/kumaru/train/model_nano.md) — the architecture, component by component
- [`SCALABILITY.md`](SCALABILITY.md) §5 — the full training ladder
