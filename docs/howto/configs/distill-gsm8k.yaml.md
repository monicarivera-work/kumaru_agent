# `configs/train/distill-gsm8k.yaml`

> The distillation recipe, teacher arm. It LoRA-tunes the student, Qwen2.5-1.5B-Instruct, on the teacher's verified answers to GSM8K's training questions. Its two siblings are the same recipe with exactly one thing changed.

**Read this when:** you are training the student on the teacher's answers, changing the recipe, or wondering why a setting has its value.

---

## What it does

This config trains a rank-16 LoRA adapter on every linear layer of `Qwen/Qwen2.5-1.5B-Instruct`: about 18.5 million trainable parameters, 1.2% of the model. Its training rows are the teacher's correct answers, which `kumaru distill build` writes to `runs/distill/data-teacher/`. The loss covers only the answer tokens. Every 50 steps, and at the last step, the trainer measures the loss on held-out questions, and the run ends on the checkpoint where that loss was lowest. That checkpoint is saved as `runs/distill/qwen1.5b-teacher/adapter`, which `kumaru bench run gsm8k -c student --adapter ...` then scores.

Three files, one recipe:

| File | What differs | The question it answers |
|---|---|---|
| `distill-gsm8k.yaml` | (the reference) | Does training on the teacher's answers raise the student's GSM8K score? |
| [`distill-gsm8k-gold.yaml`](distill-gsm8k-gold.yaml.md) | the data: GSM8K's human-written solutions to the same questions | Did the gain come from the teacher, or only from practice on GSM8K-style questions? |
| [`distill-gsm8k-0.5b.yaml`](distill-gsm8k-0.5b.yaml.md) | `base_model`: Qwen2.5-0.5B-Instruct; and the batch split, 2 × 16 | Does the pipeline work, and does distillation help at all, in half the time? |

A comparison is fair only if nothing but the tested factor changed, so `test_the_arms_and_the_pilot_differ_only_where_they_must` (in `tests/test_distill.py`) fails if any other setting differs. The pilot may split the batch of 32 rows differently, to fit its memory, but not change its size.

## When to use this profile

- After `kumaru distill build` has written `runs/distill/data-teacher/`. The whole experiment, from the teacher's answers to the final comparison, is [Part 3 of TRAINING.md](../../TRAINING.md#part-3--distillation).
- As the template for distilling another task: copy it, then change the data files, `output_dir`, `run_name`, and `data.max_seq_length` if the answers are longer.
- Not as a first test of a changed training pipeline: [`lora-small.yaml`](lora-small.yaml.md) takes minutes.

## Every field explained

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `task` | string | `lora` | LoRA fine-tuning. | Keep. |
| `base_model` | string | `Qwen/Qwen2.5-1.5B-Instruct` | The student. Its weights stay frozen; about 3 GB are downloaded on first use. | Must equal `backend.model` in [`configs/student.yaml`](student.yaml.md), which scores it; a test checks. Use an Instruct checkpoint: the trainer needs its chat template. |
| `output_dir` | string | `runs/distill/qwen1.5b-teacher` | Checkpoints, the archived `train_config.yaml`, `metrics.json`, and the final `adapter/`. | A new folder for every run (arm, seed or model), so no run mixes with another's checkpoints. |
| `run_name` | string | `distill-gsm8k-teacher` | The run's name in logs and TensorBoard. | Unique per run. |
| `load_in_4bit` | bool | `false` | Whether the frozen base is quantized to 4 bits (QLoRA). | Keep `false`. In bf16 the base is only ~3 GB, and the student is scored in bf16, so it trains on the same weights it is tested with. |
| `data.train_file` | string | `runs/distill/data-teacher/train.jsonl` | The training rows: one teacher answer each, as a user/assistant pair with no system message, like the requests `kumaru bench run` sends. | Written by `kumaru distill build`. Relative paths are read from the folder you run `kumaru train` in: the Kumaru folder. |
| `data.eval_file` | string | `runs/distill/data-teacher/val.jsonl` | The validation rows. | The held-out *questions*: none of them has any answer in `train.jsonl`, so the validation loss measures generalization, not recall. While it is set, `data.eval_split` is unused. |
| `data.max_seq_length` | int | `1536` | The longest row kept, in tokens, chat template included. Longer rows are dropped, not cut. | A row is the question with its instruction (up to ~280 tokens), the template and an answer of at most 1,024 tokens (`kumaru distill build --max-tokens`): ~1,330 at most. Lower values save memory but drop the longest answers, which belong to the hardest questions. |
| `data.train_on_completions_only` | bool | `true` | The loss covers only the answer tokens. | Keep: the student must learn to produce the answers, not to predict the questions. |
| `lora.r` | int | `16` | The adapter's rank: its capacity. | 16 on all seven projections is ~18.5M parameters, ample for ~7,000 answers. Consider 32 with several answers per question. |
| `lora.alpha` | int | `32` | The update's scale, `alpha / r` = 2. | Keep `2 × r`. |
| `lora.dropout` | float | `0.05` | Dropout on the adapter's input. | Keep. |
| `lora.target_modules` | list[string] | all seven projections | The layers that get an adapter: attention (`q`, `k`, `v`, `o`) and MLP (`gate`, `up`, `down`). | Keep all seven. In the QLoRA paper, adapters on every linear layer were needed to match full fine-tuning (Dettmers et al., 2023). |
| `lora.bias` | string | `none` | Whether biases train. | Keep `none`. |
| `optim.epochs` | float | `2` | Passes over the training rows. | Each question twice. The validation loss, logged every 50 steps, shows whether more would help: if it is still falling at the end, try 3. |
| `optim.per_device_batch_size` | int | `1` | Rows per forward pass. | It sets the peak memory ([Cost](#cost)). One row per pass fits either arm beside a game; the 0.5B pilot takes 2. |
| `optim.gradient_accumulation_steps` | int | `32` | Forward passes summed into one optimizer step. | Effective batch 1 × 32 = 32 rows: 211 steps per epoch. Any split of the 32 rows computes the same loss: transformers divides it by the answer tokens of all 32, not by the number of passes. |
| `optim.learning_rate` | float | `1.0e-4` | Peak learning rate. | The low end of LoRA's usual 1e-4 to 3e-4. The student is already a capable instruction model; the update should change how it reasons without erasing what it knows. |
| `optim.lr_scheduler` | string | `cosine` | How the learning rate decays after warmup. | Keep. |
| `optim.warmup_ratio` | float | `0.03` | Share of the steps over which the learning rate ramps up from 0. | 13 of ~420 steps. Must be below 1. |
| `optim.weight_decay` | float | `0.0` | Shrinks the trained weights each step. | 0, as in the other LoRA recipes. |
| `optim.max_grad_norm` | float | `1.0` | Gradient clipping. | Keep. |
| `optim.gradient_checkpointing` | bool | `true` | Recompute each layer's activations during the backward pass instead of storing them. | Keep. About 30% slower, but storing them takes roughly 100 KB per token per layer: ~4 GB for the longest row (estimated). |
| `optim.optimizer` | string | `adamw_torch` | PyTorch's AdamW. | Keep. Needs no bitsandbytes, and its state for 18.5M parameters is only ~150 MB. |
| `optim.bf16` | bool | `true` | bfloat16 weights and compute. | Keep on an RTX 30-series or newer. |
| `optim.seed` | int | `42` | LoRA initialization, row order and dropout. | Change it, with `output_dir` and `run_name`, for each repeat: three seeds per arm show how much of a difference is luck. The held-out questions do not move; `kumaru distill build --seed` chose them. |
| `logging_steps` | int | `10` | Training loss and learning rate are logged every 10 steps. | Keep. |
| `eval_steps` | int | `50` | The validation loss is measured every 50 steps, and at the last step. | About 9 evaluations in ~420 steps. |
| `save_steps` | int | `50` | A checkpoint every 50 steps, and at the last step. | Keep equal to `eval_steps`; a test checks. The run ends on the checkpoint with the lowest validation loss (`load_best_model_at_end`), and the trainer can only return to an evaluation it saved a checkpoint for. |
| `save_total_limit` | int | `3` | Checkpoints kept on disk. | The three newest, and the best one in any case. Each holds the adapter and its optimizer state, about 0.2 GB. |
| `report_to` | list[string] | `[]` | Where metrics go besides the console. | `["tensorboard"]` for curves, then `tensorboard --logdir runs/`. |

Fields not listed keep their defaults ([`config.md`](../src/kumaru/train/config.md)).

## Cost

Measured on an RTX 3090 (24 GB) under Windows 11, on 2026-10-22 ([the whole run](../../TRAINING.md#results-2026-10-22)).

- **Steps.** 7,089 of the teacher's answers passed the checks, 6,751 of them training rows: 211 steps per epoch, 422 in all. `--dry-run` prints the number for your data.
- **Time.** 117 minutes: about 16.6 seconds per step of 32 one-row passes. At this size a pass costs about the same whatever its length: the gold arm's steps, on rows half as long, took as long, until a game sharing the GPU slowed them by about a quarter.
- **Memory.** A peak of 13.8 GB, as `metrics.json` reports it (`peak_gpu_memory_gb`). That is what PyTorch reserved, its cache included; the tensors themselves need about 6 GB at the longest row (estimated):

| Part | At the longest row | Why |
|---|---|---|
| Base weights | 3.1 GB | 1.54B parameters × 2 bytes (bf16). |
| Adapter, gradients, AdamW state | ~0.3 GB | 18.5M parameters × 16 bytes, all in fp32. |
| Activations | ~0.3 GB | Only each layer's input is stored; one layer at a time is recomputed. |
| The loss over the vocabulary | 2.0-2.4 GB | 1.5-1.8 MB per token, over up to ~1,330 tokens: see below. |
| In use, at most | ~6 GB | The sum, estimated. |
| Reserved, measured | 13.8 GB | Most likely the rest is PyTorch's cache: see below. |

The loss is the surprise. At every position the model scores all 151,936 tokens of its vocabulary, and transformers upcasts those scores to fp32 to compute the loss: the bf16 scores, their fp32 copy and the log-probabilities together take about 1.5 MB per token, more in the backward pass. Encoded the way the trainer encodes them, the teacher's answers to a 100-question sample give rows of 455 tokens at the median (~0.7 GB for the loss) and up to ~1,330 (~2.4 GB). Rows that share a pass are padded to the longest of them, so the cost grows with the rows per pass: at 4, as this recipe first had, a pass holding one of the longest rows pads to ~5,300 positions, nearly 10 GB for the loss alone. That is why the rows per pass, not the model, set the peak. The 0.5B pilot scores the same vocabulary, and at 2 rows per pass it held 21.6 GB, more than this arm at 1.

**Why the reservation is so much larger.** `peak_gpu_memory_gb` is `torch.cuda.max_memory_reserved()`. PyTorch keeps freed memory reserved for reuse, and rows of every length leave blocks of every size, so the reservation most likely grows well past what is in use at any moment: 13.8 GB against ~6 here, 6.4 against ~4.4 in the gold arm, and 21.6 against ~6 for the pilot (estimates). PyTorch hands its cache back only when the card refuses an allocation. On Windows, by default, the driver does not refuse: past the card's memory, it hands out shared system memory (*sysmem fallback*), and training slows instead of failing. At 4 rows per pass, the GPU's memory was full within 9 steps (24,234 of 24,576 MiB in use across all processes, read with nvidia-smi), so that run was stopped.

## How to use it

Prerequisites, in the Kumaru folder:

```powershell
.\scripts\setup.ps1 -Gpu    # elsewhere: pip install -e ".[train]"
kumaru distill build runs\distill\teacher-qwen36 -o runs\distill\data-teacher
```

Validate, train, score:

```powershell
kumaru train configs\train\distill-gsm8k.yaml --dry-run
kumaru train configs\train\distill-gsm8k.yaml
kumaru bench run gsm8k -c student --full --max-tokens 1024 --adapter runs\distill\qwen1.5b-teacher\adapter --label student-teacher -o runs\distill\eval-teacher
```

- **The dry run** loads the tokenizer and the data, but not the model. It prints the plan (training and validation rows, effective batch 32, ~420 optimizer steps), says how many rows were too long, and writes one encoded row to `runs\distill\qwen1.5b-teacher\example_preview.txt`. Its `SUPERVISED TOKENS` section must hold the answer and nothing of the question.
- **Which checkpoint was kept:** `trainer_state.json`, in the last `checkpoint-*` folder, records `best_model_checkpoint` and `best_metric` (its validation loss). `metrics.json` records the training time, the peak GPU memory (`peak_gpu_memory_gb`) and the last 30 log entries.
- **The validation loss is not the score.** It picks the checkpoint; the score is `kumaru bench run gsm8k`, on test questions no training row came from.

To repeat an arm with another seed, copy the file, then change `optim.seed`, `output_dir` and `run_name`; make the same copy of the gold arm.

## Tuning it

Change all three files together; the arm test fails otherwise.

1. **Out of memory, or steps far slower than ~17 seconds:** this recipe already takes one row per pass. Free the GPU first: close any game, OBS or anything else that holds GPU memory. Next, lower `data.max_seq_length` in all three recipes; that drops the longest answers, often those to the hardest questions, and the dry run says how many. On Windows, setting NVIDIA Control Panel's *CUDA - Sysmem Fallback Policy* to *Prefer No Sysmem Fallback* should turn the slowdown into a refused allocation, which PyTorch answers by emptying its cache and retrying; this was not tried here.
2. **Validation loss still falling at the last evaluation:** `epochs: 3`.
3. **Validation loss lowest early, rising after:** the run already ends on the best checkpoint. For the next run, lower `learning_rate` to `5.0e-5`, or train 1 epoch.
4. **Two teacher answers per question** (`kumaru distill build` given two runs): about twice the rows, steps and time. `lora.r: 32` gives the adapter room for them.
5. **Another student:** change `base_model` here and in the gold arm, and `backend.model` in `configs/student.yaml`. Tests fail if they disagree.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Training data not found: runs\distill\data-teacher\train.jsonl` | The dataset is not built, or `kumaru train` ran in another folder. | `kumaru distill build ...` (above); run from the Kumaru folder. |
| `dropped N/M example(s) (x%) longer than 1536 tokens` | Answers longer than `kumaru distill build --max-tokens` allowed, or a much longer prompt. | A few are harmless. Many: rebuild with a lower `--max-tokens`, or raise `max_seq_length` (more memory). |
| CUDA out of memory, or steps far slower than ~17 seconds | What a game or OBS holds, on top of PyTorch's cache. On Windows, running short slows training instead of failing: *Shared GPU memory* rises in Task Manager. | Tuning, step 1. |
| `test_the_arms_and_the_pilot_differ_only_where_they_must` fails | One file was edited alone, or the pilot's batch split no longer makes 32 rows. | Make the same change in all three. The pilot may split the batch differently, but not change its size. |
| Size-mismatch error when scoring the adapter | It is scored on another base model (for example a 0.5B adapter on the 1.5B student). | Pass the matching `-m` to `kumaru bench run`. |
| The two arms score within a few points of each other | Possibly no real difference: at 1,319 questions, about 3 points is the smallest reliably detectable. | Read `kumaru bench compare`'s p-value; train three seeds per arm before concluding. |

## Related files

- `configs/train/distill-gsm8k-gold.yaml`, `configs/train/distill-gsm8k-0.5b.yaml` — the control arm and the pilot ([`distill-gsm8k-gold.yaml.md`](distill-gsm8k-gold.yaml.md), [`distill-gsm8k-0.5b.yaml.md`](distill-gsm8k-0.5b.yaml.md)).
- `configs/student.yaml` — scores the adapters ([`student.yaml.md`](student.yaml.md)).
- `src/kumaru/train/distill.py` — builds the datasets ([`distill.md`](../src/kumaru/train/distill.md)).
- `src/kumaru/train/finetune_lora.py` — the trainer, and how its settings reach transformers ([`finetune_lora.md`](../src/kumaru/train/finetune_lora.md)).
- `src/kumaru/train/config.py` — the schema of this file ([`config.md`](../src/kumaru/train/config.md)).
- `tests/test_distill.py` — the arm test; `tests/test_finetune_lora.py` — every setting reaches the trainer, warmup included.
- [`docs/TRAINING.md`](../../TRAINING.md#part-3--distillation) — the experiment, step by step.
