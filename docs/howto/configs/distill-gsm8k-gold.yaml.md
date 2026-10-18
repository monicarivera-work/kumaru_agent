# `configs/train/distill-gsm8k-gold.yaml`

> The control arm of the distillation experiment: the student and recipe of [`distill-gsm8k.yaml`](distill-gsm8k.yaml.md), trained on GSM8K's human-written solutions to the same questions instead of the teacher's answers.

**Read this when:** you are training the control arm, or deciding what a teacher-versus-gold result means.

---

## What it does

A student trained on the teacher's answers can beat the base model for two reasons. The teacher's answers may have taught it something; or any practice on GSM8K-style questions, in the answer format the grader expects, would have done the same. A comparison with the base model cannot tell these apart. This arm changes only who wrote the answers, so comparing the two arms isolates what the teacher added.

`kumaru distill gold --match runs\distill\data-teacher` writes its data. For every question in the teacher dataset, it takes GSM8K's own solution and rewrites it in the teacher's format: calculator notes such as `<<48/2=24>>` are removed, and `#### 72` becomes `Answer: 72`. Each question stays on the same side of the train/validation split. The arithmetic check runs on these solutions too. It flags 24 of them, [17 real errors](../src/kumaru/train/distill.md#5-the-arithmetic-check-wrong_steps) in GSM8K's reference answers among them, so up to 24 questions are in the teacher arm only.

## When to use this profile

- After the teacher dataset is built; `--match` reads it. Train it alongside the teacher arm: without it, a teacher-arm gain has two possible explanations.
- Not for the 0.5B pilot, which has no control arm. To add one, copy this file and change `base_model`, `output_dir` and `run_name` as [`distill-gsm8k-0.5b.yaml`](distill-gsm8k-0.5b.yaml.md) does.

## Every field explained

Only four fields differ from the teacher arm. Every other field, and why it has its value, is in [`distill-gsm8k.yaml.md`](distill-gsm8k.yaml.md#every-field-explained). `test_the_arms_and_the_pilot_differ_only_where_they_must` (in `tests/test_distill.py`) fails if anything else differs.

| Field | Here | In the teacher arm | Why it differs |
|---|---|---|---|
| `data.train_file` | `runs/distill/data-gold/train.jsonl` | `runs/distill/data-teacher/train.jsonl` | The tested factor: who wrote the answers. |
| `data.eval_file` | `runs/distill/data-gold/val.jsonl` | `runs/distill/data-teacher/val.jsonl` | The same held-out questions, with human answers, so each arm picks its checkpoint on its own kind of answer. |
| `output_dir` | `runs/distill/qwen1.5b-gold` | `runs/distill/qwen1.5b-teacher` | Each arm keeps its own checkpoints and adapter. |
| `run_name` | `distill-gsm8k-gold` | `distill-gsm8k-teacher` | Names the run in logs. |

## Cost and length

GSM8K's solutions are short. These counts were measured with the trainer's own encoding, on the full gold dataset and on the teacher's answers to a 100-question sample:

| | Gold | Teacher |
|---|---|---|
| Answer, mean | 93 tokens (250 characters) | 333 tokens (1,040 characters) |
| Whole row, median | 206 tokens | 455 tokens |
| Longest row | 506 tokens | up to ~1,330 |

- **Steps:** the same ~420 as the teacher arm, when the teacher dataset holds one answer per question.
- **Time:** a batch is about half as long, so the run takes about half the teacher arm's time, roughly half an hour (estimated).
- **Memory:** no batch exceeds ~2,050 positions, so the loss needs at most ~3.7 GB, and the whole run ~9 GB (estimated).

**The arms differ in length by design.** With equal rows and steps, the gold arm trains on about a quarter as many answer tokens. That is part of what is being compared, since writing out every step is part of what a teacher gives. But it means a teacher win cannot say whether the teacher reasons better, or only writes more steps. The opposite can happen too. Li et al. (2025, *Small Models Struggle to Learn from Strong Reasoners*) found that students of 3B parameters or fewer often learn more from short reasoning than from a strong teacher's long reasoning. So the short human solutions can win.

## How to use it

From the Kumaru folder, after `kumaru distill build` and the base and teacher-arm evaluations ([Part 3 of TRAINING.md](../../TRAINING.md#part-3--distillation)):

```powershell
kumaru distill gold -o runs\distill\data-gold --match runs\distill\data-teacher
kumaru train configs\train\distill-gsm8k-gold.yaml --dry-run
kumaru train configs\train\distill-gsm8k-gold.yaml
kumaru bench run gsm8k -c student --full --max-tokens 1024 --adapter runs\distill\qwen1.5b-gold\adapter --label student-gold -o runs\distill\eval-gold
kumaru bench compare runs\distill\eval-gold runs\distill\eval-teacher -e 5
kumaru bench compare runs\distill\eval-base runs\distill\eval-gold
```

- **The build report** counts `not_matched` for every GSM8K question that is not in the teacher dataset: mostly the questions the teacher got wrong. The rows should then nearly equal the teacher dataset's.
- **The first comparison** is the experiment: the gold arm is the baseline (A), the teacher arm the candidate (B). The second shows what GSM8K practice alone did.
- **Never compare the arms' validation losses.** Each is measured on its own kind of answer, and the teacher's long, regular text is easier to predict per token. Only the GSM8K scores compare.

### Reading the result

"Significant" means `kumaru bench compare` reports a p-value below 0.05. At 1,319 questions, a difference of about 3 points is the smallest it reliably detects. Train three seeds per arm before concluding ([how](distill-gsm8k.yaml.md#how-to-use-it)).

| Result | What it means |
|---|---|
| Teacher arm significantly above gold | The teacher's answers taught more than human ones: distillation worked. |
| No significant difference, both above base | The gain came from GSM8K-style practice; the teacher added nothing measurable. |
| Gold significantly above the teacher arm | Human solutions were better data for this student, likely for the reason above. Next: shorter teacher answers, or both datasets mixed. |
| Neither arm above base | Training changed nothing measurable. Check `example_preview.txt`, and whether the training loss fell (`log_history` in the last checkpoint's `trainer_state.json`). |

## Tuning it

Do not tune this file alone. Any change except the four fields above belongs in all three recipes: tune the teacher arm ([Tuning it](distill-gsm8k.yaml.md#tuning-it)) and copy the change here. Otherwise the arm test fails, and the comparison is no longer fair.

One case needs care. If the teacher dataset holds several answers per question, the gold arm still has one per question, so it gets fewer rows and fewer steps. Either report that as part of the result, or build the teacher dataset with `--per-question 1`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `...\data-teacher\train.jsonl not found` | The teacher dataset is not built yet, or `--match` points elsewhere. | Run `kumaru distill build` first; pass its `-o` folder to `--match`. |
| `runs\distill\data-gold already holds a dataset` | A previous build. | Add `--force` to replace it. |
| `not_matched` near 7,473 | `--match` points at a small dataset, such as one built from the 200-question pilot. | Rebuild the teacher dataset from the full run, then this one with `--force`. |
| `Training data not found: runs\distill\data-gold\train.jsonl` | This dataset is not built, or `kumaru train` ran in another folder. | Run `kumaru distill gold` (above), from the Kumaru folder. |
| `test_the_arms_and_the_pilot_differ_only_where_they_must` fails | One recipe was edited alone. | Make the same change in all three. |
| The gold arm's validation loss is far above the teacher arm's | Expected (see above). | Compare GSM8K scores. |

## Related files

- `configs/train/distill-gsm8k.yaml` — the teacher arm: every shared field, the cost breakdown and tuning ([`distill-gsm8k.yaml.md`](distill-gsm8k.yaml.md)).
- `src/kumaru/train/distill.py` — `build_gold`, `gold_rationale` and the arithmetic check ([`distill.md`](../src/kumaru/train/distill.md)).
- `src/kumaru/eval/benchmarks/gsm8k.py` — `gsm8k_train`, whose prompts and ids both arms share ([`gsm8k.md`](../src/kumaru/eval/benchmarks/gsm8k.md)).
- `tests/test_distill.py` — the arm test.
- [`docs/TRAINING.md`](../../TRAINING.md#part-3--distillation) — the experiment, step by step.
