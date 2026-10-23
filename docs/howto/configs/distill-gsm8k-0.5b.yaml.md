# `configs/train/distill-gsm8k-0.5b.yaml`

> The distillation pilot: the teacher-arm recipe of [`distill-gsm8k.yaml`](distill-gsm8k.yaml.md) on the smaller student, Qwen2.5-0.5B-Instruct. It needs about a third of the compute and took half the time of a 1.5B arm, so it checks the whole pipeline and gives a first answer before the 1.5B runs.

**Read this when:** you want a cheap first pass through the experiment, or are reading a 0.5B result.

---

## What it does

It trains a rank-16 LoRA adapter on the teacher's answers, exactly as the teacher arm does, but on `Qwen/Qwen2.5-0.5B-Instruct`. That model has 24 layers and a hidden size of 896, against 28 layers and 1,536 for the 1.5B. The adapter has 8,798,208 trainable parameters, 1.8% of the model. The base weights are about 1 GB, downloaded on first use.

A pilot run tells you four things before the runs that matter:

1. **The pipeline works end to end:** data, training, adapter, scoring and comparison.
2. **Memory:** `peak_gpu_memory_gb` in `runs\distill\qwen0.5b-teacher\metrics.json`: 21.6 GB on 2026-10-22, at 2 rows per forward pass. The arms take 1 row and held less, 13.8 GB at most, because the rows per pass, not the model, set the peak ([why](distill-gsm8k.yaml.md#cost)).
3. **Time:** `train_seconds` in the same file: 58 minutes. The 1.5B arms make twice as many passes per step and took about twice as long, 117 minutes for the teacher arm.
4. **A first answer:** whether training on the teacher's answers moves the score at all.

## When to use this profile

- First, before the 1.5B runs.
- Again after changing the shared recipe: it is the quickest full check of the change.
- Not as the experiment's result. A 0.5B result is evidence about the 1.5B, not proof (see [Reading the result](#reading-the-result)).

## Every field explained

Five fields differ from the teacher arm: three by design, and the two that split the batch. Every other field, and why it has its value, is in [`distill-gsm8k.yaml.md`](distill-gsm8k.yaml.md#every-field-explained). `test_the_arms_and_the_pilot_differ_only_where_they_must` (in `tests/test_distill.py`) fails if anything else differs.

| Field | Here | In the teacher arm | Why it differs |
|---|---|---|---|
| `base_model` | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | The point of the pilot: a smaller student. Same tokenizer and chat template, so the same data fits. |
| `output_dir` | `runs/distill/qwen0.5b-teacher` | `runs/distill/qwen1.5b-teacher` | Each model keeps its own checkpoints and adapter. |
| `run_name` | `distill-gsm8k-teacher-0.5b` | `distill-gsm8k-teacher` | Names the run in logs. |
| `optim.per_device_batch_size` | `2` | `1` | Two rows per forward pass fit the smaller model, in half the passes. |
| `optim.gradient_accumulation_steps` | `16` | `32` | Keeps the effective batch at 2 × 16 = 32 rows, as in the arms. |

The batch split changes the memory and the time, not what is learned. Each optimizer step still sees the same 32 rows, and the loss is the same however they are split: transformers divides it by the answer tokens of all 32, not by the number of passes. At 2 × 16, PyTorch's peak reservation was 21.6 GB. The recipes' first split, 4 × 8, was stopped after 9 steps, when nvidia-smi showed the GPU's memory full: 24,234 of 24,576 MiB in use, across all processes. The test allows the pilot this difference and no other: its rows per pass times its passes must equal the arms'.

## Cost

Measured on an RTX 3090 (24 GB) under Windows 11, on 2026-10-22.

- **Steps:** 422, as in the arms.
- **Time:** 58 minutes, about 8.2 seconds per step of 16 two-row passes: half the 1.5B teacher arm's 117 minutes. Scoring took 21 minutes for the untrained 0.5B and 54 with the adapter, which runs beside the base weights at every token ([why that is slower](../../TRAINING.md#what-it-costs)).
- **Memory:** a peak of 21.6 GB reserved, more than either 1.5B arm (13.8 and 6.4 GB), because it takes two rows per pass. A smaller model saves little memory: the weights shrink from 3.1 GB to 1 GB, but the loss over the vocabulary does not, since both models score all 151,936 tokens of the same vocabulary at every position ([why](distill-gsm8k.yaml.md#cost)).

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

Qwen reports about 50% on GSM8K for the 0.5B model and about 73% for the 1.5B (Qwen2.5 technical report); Kumaru measured 50.3% and 72.3%. Prompts and graders differ between setups, so compare only runs scored the same way. The 0.5B starts lower, so it has more room to gain.

- **A significant gain** (p below 0.05 in `kumaru bench compare`): the pipeline works, and the 1.5B runs are worth their two hours each. The pilot has no control arm, so the gain could come from the teacher or from GSM8K practice alone; the 1.5B gold arm decides which.
- **No gain:** the 1.5B can still gain. Li et al. (2025, *Small Models Struggle to Learn from Strong Reasoners*) found that students of 3B parameters or fewer gain less from a strong teacher than larger students do. Check `example_preview.txt` and the losses first, though.
- **A loss:** the adapter hurt the 0.5B. Look at the `-e 5` examples, and at how many answers were cut off (`kumaru bench report`). A broken answer format is a common cause, and the 1.5B would show it too.

**The 2026-10-22 run gave a loss.** The pilot scored 44.7% (42.0–47.3) against 50.3% (47.6–53.0) untrained: −5.6 points (−8.3 to −2.9), with 206 questions only the untrained model got right against 132 only the pilot did, p 6.74e-05. Its training went as planned: the validation loss fell from 0.331 to 0.285, lowest at the last step. Its format was not broken either. It took the teacher's, with `**Step` headers in 1,134 answers and an `Answer:` line in every one it finished, where the untrained model boxed its answer (`\boxed`) in 1,203. It also ran long more often: 63 answers hit the token limit, against 17. But most of its 206 losses were not cut off: 19 were, 48 had the right number in the working but a wrong final answer, and 139 never reached it. The 1.5B teacher arm then held even with its untrained model (−0.4 points, p 0.815), so the loss at 0.5B did not carry over to 1.5B, which fits Li et al. (2025) ([the whole run](../../TRAINING.md#results-2026-10-22)).

To give the pilot a control arm, copy [`distill-gsm8k-gold.yaml`](distill-gsm8k-gold.yaml.md) and change `base_model`, `output_dir`, `run_name` and the batch split as this file does.

## Tuning it

Change all three recipes together: the arm test fails otherwise. Tune the teacher arm ([Tuning it](distill-gsm8k.yaml.md#tuning-it)) and copy the change here and to the gold arm. The pilot exists to test such a change cheaply before the 1.5B runs.

The batch split is the exception: here it may differ from the arms', as long as it makes 32 rows. If a game shares the GPU, use the arms' 1 × 32: it should hold less, but may take up to twice as long (not measured for the 0.5B).

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Size-mismatch error while loading the adapter, when scoring | The 0.5B adapter is being loaded onto the 1.5B model. | Pass `-m Qwen/Qwen2.5-0.5B-Instruct` to `kumaru bench run`. |
| `Training data not found: runs\distill\data-teacher\train.jsonl` | The teacher dataset is not built, or `kumaru train` ran in another folder. | Run `kumaru distill build` ([`distill.md`](../src/kumaru/train/distill.md#how-to-use-it)), from the Kumaru folder. |
| CUDA out of memory, or steps far slower than ~8 seconds | Two rows per pass, plus what a game holds: the 0.5B does not avoid the cost of the vocabulary. On Windows, running short slows training instead of failing (*Shared GPU memory* rises in Task Manager). | Close any game, or split the batch 1 × 32 here, as the arms do ([Tuning it](distill-gsm8k.yaml.md#tuning-it), step 1). |
| `test_the_arms_and_the_pilot_differ_only_where_they_must` fails | One recipe was edited alone, or this file's batch split no longer makes 32 rows. | Make the same change in all three; keep rows per pass × passes at 32 here. |

## Related files

- `configs/train/distill-gsm8k.yaml` — the teacher arm, whose recipe this runs: every field, the cost breakdown and tuning ([`distill-gsm8k.yaml.md`](distill-gsm8k.yaml.md)).
- `configs/train/distill-gsm8k-gold.yaml` — the control arm ([`distill-gsm8k-gold.yaml.md`](distill-gsm8k-gold.yaml.md)).
- `configs/student.yaml` — scores the students; `-m` swaps in the 0.5B ([`student.yaml.md`](student.yaml.md)).
- `src/kumaru/train/finetune_lora.py` — the trainer, which writes `metrics.json` ([`finetune_lora.md`](../src/kumaru/train/finetune_lora.md)).
- `tests/test_distill.py` — the arm test.
- [`docs/TRAINING.md`](../../TRAINING.md#part-3--distillation) — the experiment, step by step.
