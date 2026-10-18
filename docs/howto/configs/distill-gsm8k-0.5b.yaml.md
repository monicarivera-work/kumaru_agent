# `configs/train/distill-gsm8k-0.5b.yaml`

> The distillation pilot: the teacher-arm recipe of [`distill-gsm8k.yaml`](distill-gsm8k.yaml.md) on the smaller student, Qwen2.5-0.5B-Instruct. It needs about a third of the compute, so it checks the whole pipeline, measures time and memory, and gives a first answer before the 1.5B run.

**Read this when:** you want a cheap first pass through the experiment, or are reading a 0.5B result.

---

## What it does

It trains a rank-16 LoRA adapter on the teacher's answers, exactly as the teacher arm does, but on `Qwen/Qwen2.5-0.5B-Instruct`. That model has 24 layers and a hidden size of 896, against 28 layers and 1,536 for the 1.5B. The adapter has 8,798,208 trainable parameters, 1.8% of the model. The base weights are about 1 GB, downloaded on first use.

A pilot run tells you four things before the run that matters:

1. **The pipeline works end to end:** data, training, adapter, scoring and comparison.
2. **Memory:** `peak_gpu_memory_gb` in `runs\distill\qwen0.5b-teacher\metrics.json`. The 1.5B run needs about 2-3 GB more.
3. **Time:** `train_seconds` in the same file. The 1.5B run takes about two to three times as long.
4. **A first answer:** whether training on the teacher's answers moves the score at all.

## When to use this profile

- First, before the 1.5B run, especially while a game shares the GPU.
- Again after changing the shared recipe: it is the quickest full check of the change.
- Not as the experiment's result. A 0.5B result is evidence about the 1.5B, not proof (see [Reading the result](#reading-the-result)).

## Every field explained

Only three fields differ from the teacher arm. Every other field, and why it has its value, is in [`distill-gsm8k.yaml.md`](distill-gsm8k.yaml.md#every-field-explained). `test_the_arms_and_the_pilot_differ_only_where_they_must` (in `tests/test_distill.py`) fails if anything else differs.

| Field | Here | In the teacher arm | Why it differs |
|---|---|---|---|
| `base_model` | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | The point of the pilot: a smaller student. Same tokenizer and chat template, so the same data fits. |
| `output_dir` | `runs/distill/qwen0.5b-teacher` | `runs/distill/qwen1.5b-teacher` | Each model keeps its own checkpoints and adapter. |
| `run_name` | `distill-gsm8k-teacher-0.5b` | `distill-gsm8k-teacher` | Names the run in logs. |

The batch stays at 4 rows × 8 accumulation steps. The smaller model would fit 8 × 4, the same effective batch, and run a little faster. Keeping 4 × 8 means the pilot runs exactly the same recipe.

## Cost

- **Steps:** the same ~420.
- **Time:** about 20-30 minutes, estimated. The 0.5B needs about a third of the 1.5B's compute per token. A small model uses the GPU less efficiently, though, so it takes more than a third of the time.
- **Memory:** a peak near 12 GB, and about 6.5 GB on a typical batch (estimated). It saves time, not memory. The weights shrink from 3.1 GB to 1 GB, but the loss over the vocabulary does not shrink: both models score all 151,936 tokens of the same vocabulary at every position. That costs 4-10 GB in both ([why](distill-gsm8k.yaml.md#cost)).

## How to use it

From the Kumaru folder, after `kumaru distill build` ([Part 3 of TRAINING.md](../../TRAINING.md#part-3--distillation)):

```powershell
kumaru train configs\train\distill-gsm8k-0.5b.yaml --dry-run
kumaru train configs\train\distill-gsm8k-0.5b.yaml
kumaru bench run gsm8k -c student -m Qwen/Qwen2.5-0.5B-Instruct --full --max-tokens 1024 --label student0.5b-base -o runs\distill\eval-0.5b-base
kumaru bench run gsm8k -c student -m Qwen/Qwen2.5-0.5B-Instruct --full --max-tokens 1024 --adapter runs\distill\qwen0.5b-teacher\adapter --label student0.5b-teacher -o runs\distill\eval-0.5b-teacher
kumaru bench compare runs\distill\eval-0.5b-base runs\distill\eval-0.5b-teacher -e 5
```

- **`-m` is required** when scoring. `configs/student.yaml` names the 1.5B model, and an adapter loaded onto the wrong base model fails with a size-mismatch error.
- **The 0.5B base run is its own baseline.** Compare the pilot's adapter only with the 0.5B base model, never with a 1.5B run.

### Reading the result

Qwen reports about 50% on GSM8K for the 0.5B model and about 73% for the 1.5B (Qwen2.5 technical report). Kumaru's prompt and grader give their own numbers, so compare only runs scored the same way. The 0.5B starts lower, so it has more room to gain.

- **A significant gain** (p below 0.05 in `kumaru bench compare`): the pipeline works, and the 1.5B run is worth its hour. The pilot has no control arm, so the gain could come from the teacher or from GSM8K practice alone; the 1.5B gold arm decides which.
- **No gain:** the 1.5B can still gain. Li et al. (2025, *Small Models Struggle to Learn from Strong Reasoners*) found that students of 3B parameters or fewer gain less from a strong teacher than larger students do. Check `example_preview.txt` and the losses first, though.
- **A loss:** the adapter hurt the 0.5B. Look at the `-e 5` examples; a broken answer format is a common cause, and the 1.5B would show it too.

To give the pilot a control arm, copy [`distill-gsm8k-gold.yaml`](distill-gsm8k-gold.yaml.md) and change `base_model`, `output_dir` and `run_name` as this file does.

## Tuning it

Change all three recipes together: the arm test fails otherwise. Tune the teacher arm ([Tuning it](distill-gsm8k.yaml.md#tuning-it)) and copy the change here and to the gold arm. The pilot exists to test such a change cheaply before the 1.5B run.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Size-mismatch error while loading the adapter, when scoring | The 0.5B adapter is being loaded onto the 1.5B model. | Pass `-m Qwen/Qwen2.5-0.5B-Instruct` to `kumaru bench run`. |
| `Training data not found: runs\distill\data-teacher\train.jsonl` | The teacher dataset is not built, or `kumaru train` ran in another folder. | Run `kumaru distill build` ([`distill.md`](../src/kumaru/train/distill.md#how-to-use-it)), from the Kumaru folder. |
| CUDA out of memory | The loss over the vocabulary on a batch of long rows, plus what a game holds. The 0.5B does not avoid it. | Halve the batch in all three recipes ([Tuning it](distill-gsm8k.yaml.md#tuning-it), step 1). |
| `test_the_arms_and_the_pilot_differ_only_where_they_must` fails | One recipe was edited alone. | Make the same change in all three. |

## Related files

- `configs/train/distill-gsm8k.yaml` — the teacher arm, whose recipe this runs: every field, the cost breakdown and tuning ([`distill-gsm8k.yaml.md`](distill-gsm8k.yaml.md)).
- `configs/train/distill-gsm8k-gold.yaml` — the control arm ([`distill-gsm8k-gold.yaml.md`](distill-gsm8k-gold.yaml.md)).
- `configs/student.yaml` — scores the students; `-m` swaps in the 0.5B ([`student.yaml.md`](student.yaml.md)).
- `src/kumaru/train/finetune_lora.py` — the trainer, which writes `metrics.json` ([`finetune_lora.md`](../src/kumaru/train/finetune_lora.md)).
- `tests/test_distill.py` — the arm test.
- [`docs/TRAINING.md`](../../TRAINING.md#part-3--distillation) — the experiment, step by step.
