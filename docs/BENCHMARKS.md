# Benchmarks: is Kumaru actually smarter?

> You asked what would make Kumaru "incredibly smart". The biggest single lever is the model, and the only honest way to know a change helped is to measure it. This document is both: the measured result of replacing Qwen2.5-7B with Qwen3.6-35B-A3B, and the manual for measuring every change you make from now on.

**Read this when:** you want the numbers; you are about to change a model, prompt or setting and want to know whether it helped; or you want to add a benchmark.

---

## TL;DR

Accuracy (%) on the same questions, same engine, same harness. In parentheses: the 95% confidence interval.

| Benchmark (questions) | Qwen2.5-7B (before) | Qwen3.6-35B-A3B, Think off | Qwen3.6-35B-A3B, Think on |
|---|---|---|---|
| golden (27): your own questions | 96.3 (81.7–99.3) | 96.3 (81.7–99.3) | 100.0 (87.5–100.0) |
| GSM8K (100): grade-school math | 92.0 (85.0–95.9) | 95.0 (88.8–97.8) | 97.0 (91.5–99.0) |
| MMLU-Pro (100): knowledge + reasoning | 59.0 (49.2–68.1) | 84.0 (75.6–89.9) | 79.0 (70.0–85.8) |
| IFEval (100): instruction following | 72.0 (62.5–79.9) | 80.0 (71.1–86.7) | not run |
| HumanEval (100): Python coding | 81.0 (72.2–87.5) | 99.0 (94.6–99.8) | not run |
| tokens per answer (mean) | 283 | 611 | 2,039 (Think off: 856 on the same three benchmarks) |

**Verdicts** (paired significance tests, [§4](#4-reading-the-numbers)): Qwen3.6 with Think off beats the 7B on MMLU-Pro (+25 points) and HumanEval (+18), both p < 0.001; the golden, GSM8K and IFEval differences are within noise. Think on against Think off: no clear difference on golden (+3.7), GSM8K (+2.0) or MMLU-Pro (−5.0, p = 0.5), at 2.4 times the tokens. The MMLU-Pro dip is not evidence that thinking hurts: 24 of its 100 Think-on answers hit the 4,096-token thinking cap, including 5 of the 6 questions that Think off got right and Think on got wrong.

**What to use:** Qwen3.6 (`-c qwen36`) with Think off, the profile's default. It is the measured upgrade over the 7B, and in this run thinking bought no significant accuracy for 2.4 times the tokens. Think on is still worth trying on hard multi-step problems: chat does not cap thinking, so uncapped thinking is not yet measured. A rerun with a larger `--thinking-budget` would measure it.

**Other programming languages:** HumanEval measures Python only. [§11](#11-programming-languages-multipl-e) adds MultiPL-E, the same problems in JavaScript, TypeScript, C#, Java, C++, Go and Rust, graded by running each language's unit tests. On 100 of them, Qwen3.6 with Think off scored 98 in Python, 96 in TypeScript, 95 in C++ (96 graded on Linux, in the Docker sandbox), 94 in JavaScript and C#, 93 in Java and Go, and 92 in Rust. None is significantly below Python, and each translation loses 1 or 2 of its points to defects in its tests. Half its mistakes in C#, Java, C++, Go and Rust were compile errors ([§11.5](#115-results)).

The Think-on run was stopped after three benchmarks to free the GPU. To finish IFEval and HumanEval, run its original command again with `--out`; finished questions are skipped ([§9](#9-run-mechanics)):

```powershell
$env:KUMARU_BACKEND__MODELS_DIR = 'D:\kumaru\models'      # where that run's model is stored
$env:KUMARU_BACKEND__LLAMA_SERVER__N_CPU_MOE = '18'       # the GPU split that run used (§3)
kumaru bench run golden gsm8k mmlu_pro ifeval humaneval -c qwen36 --context 65536 --think --thinking-budget 4096 -n 100 --allow-code-exec --out runs\bench\20261010-134047-qwen36-think4096
```

Why a bigger model can be this fast: Qwen3.6-35B-A3B is a *mixture of experts*. It has 35B parameters (5x the 7B), and knowledge scales with total parameters, but a router activates only ~3B of them per token, and speed scales with *active* parameters. It is also two model generations newer, with an optional *thinking* mode that reasons privately before answering. The full explanation, including why it fits in 24 GB, is in [`configs/qwen36.yaml`](../configs/qwen36.yaml).

---

## 1. Measure something in five minutes

```powershell
pip install -e ".[bench]"                       # once: datasets + IFEval's checkers
kumaru bench list                               # what is available
kumaru bench run golden -c qwen36               # your 27 questions, about two minutes
kumaru bench run gsm8k mmlu_pro -c qwen36 -n 100
kumaru bench report  runs\bench\<run folder>
kumaru bench compare runs\bench\<run A> runs\bench\<run B> --examples 3
```

The first run of a public benchmark downloads its dataset (cached afterwards). `qwen36` needs `.\scripts\get-llama-server.ps1` once, and downloads the 17.7 GB model on first use; any profile works, including your fine-tuned ones.

`run` prints one progress line per benchmark and saves every answer. `report` shows a saved run in detail. `compare` answers the question you actually care about: *did B really beat A, or is that noise?*

## 2. The benchmarks

| Name | What it measures | Full size | Chance | Score |
|---|---|---|---|---|
| `golden` | Your own regression questions ([§8](#8-your-own-golden-questions)) | 27, grow it | - | accuracy |
| `gsm8k` | Multi-step grade-school arithmetic (Cobbe et al., 2021) | 1,319 | ~0% | final number correct |
| `mmlu_pro` | Knowledge plus reasoning across 14 subjects, up to 10 options (Wang et al., 2024) | 12,032 | ~11% | letter correct |
| `ifeval` | Following verifiable instructions: "no commas", "exactly 3 bullet points" (Zhou et al., 2023) | 541 prompts, 834 instructions, 25 types | - | prompt-level strict |
| `humaneval` | Writing a correct Python function from its docstring (Chen et al., 2021) | 164 | ~0% | pass@1: hidden unit tests pass |
| `multipl_e` | HumanEval's problems in 8 languages: Python, JavaScript, TypeScript, C#, Java, C++, Go, Rust (Cassano et al., 2023) | 152 per language | ~0% | pass@1, per language |

What each is good for:

- **MMLU-Pro** separates models best: it is hard (10 options, many need calculation), broad, and far from saturated. Weight it most.
- **GSM8K** is nearly saturated by modern models (both score in the 90s). It is a smoke test: a big drop means something broke. Small models still have room: the Qwen2.5 report gives about 73% for Qwen2.5-1.5B, so [distillation](TRAINING.md#part-3--distillation) is measured on it.
- **IFEval** is the closest proxy for "does what I asked". It reports four scores: *prompt-level* (every instruction in the prompt followed) and *instruction-level* (share of individual instructions followed), each *strict* (checked literally) or *loose* (forgiving markdown asterisks or a "Sure, here it is" first line). The headline is prompt-level strict, the strictest and the one model cards usually quote.
- **HumanEval** runs the model's code against unit tests, so it measures working code, not plausible-looking code. It is opt-in (`--allow-code-exec`, [§7](#7-caveats-read-before-quoting-these-numbers)).
- **MultiPL-E** asks the same problems in every language, so a language that scores well below Python is a weak language, not a harder test. `multipl_e` is a suite of eight benchmarks, `multipl_e_py` to `multipl_e_rs`. Each needs its language's compiler or runtime; a language without one is skipped, not scored 0 ([§11](#11-programming-languages-multipl-e)).
- **golden** is the only benchmark that cannot have leaked into a model's training data, and the only one that measures what *you* care about.

`kumaru bench list` also shows **`gsm8k_train`**, GSM8K's 7,473 training problems. It is not a score: a teacher model answers it to make distillation data, and `kumaru distill` accepts only runs of training splits like it ([TRAINING.md Part 3](TRAINING.md#part-3--distillation)).

## 3. How the comparison was run

Everything was held fixed except the model, and in the third column, thinking:

| | Setting |
|---|---|
| Engine | llama.cpp `llama-server` build b11541, started and managed by Kumaru (`backend.name: llamacpp_server`) |
| Weights | Qwen2.5-7B-Instruct Q4_K_M; Qwen3.6-35B-A3B UD-IQ4_XS. Both are ~4-bit quantizations |
| Questions | 100 per benchmark (golden: all 27), drawn with seed 1234. MMLU-Pro is stratified so all 14 subjects appear in proportion |
| Pairing | Every model answers the same questions with the same per-question sampling seed |
| Prompting | Zero-shot (no worked examples) and no system prompt: this measures the model, not Kumaru's chat persona. `--use-agent-prompt` adds the persona if you want to measure the whole agent |
| Answer budget | Up to 8,192 tokens per answer, plus 4,096 for thinking when it is on |
| Concurrency | Up to 4 questions at once, batched into one forward pass; fewer if they would not all fit in the KV cache (the 7B, with a 32k window, ran 3) |
| Hardware | RTX 3090 24 GB, Ryzen 7 5700X, 64 GB RAM, Windows 11 |

**Prompts.** Each benchmark asks for its answer in a fixed, parseable form:

| Benchmark | Prompt |
|---|---|
| GSM8K | `Solve the following math problem. Think step by step, then write the final answer on its own last line in the form "Answer: <number>", with no units.` + the problem |
| MMLU-Pro | `The following is a multiple choice question about {subject}. Think step by step, then finish your answer with "The answer is (X)" where X is the letter of the correct option.` + question and options |
| IFEval | The dataset prompt, unchanged |
| HumanEval | ``Complete the following Python function. Return the complete function, including its signature and any imports it needs, in a single ```python code block.`` + the stub |
| MultiPL-E | ``Complete the following {language} function. Keep its name and signature, and return the complete function, plus any helper functions or imports it needs, in a single ```{fence} code block. Do not include a main function, tests or example usage.`` + the stub in that language, fenced |

The extractors accept the common ways models deviate (`**Answer:** $1,080`, `\boxed{C}`, a bare function body) and record which method found the answer; `report` shows the counts.

**Decoding.** Each model uses its publisher's recommended sampling preset, not one shared setting: models are tuned for their presets, and greedy decoding, the usual "neutral" choice, sends Qwen3 models into repetition loops.

| Run | temperature | top_p | top_k | min_p | presence penalty | repetition penalty |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 0.7 | 0.8 | 20 | 0 | 0 | 1.05 |
| Qwen3.6, Think off | 0.7 | 0.8 | 20 | 0 | 1.5 | 1.0 |
| Qwen3.6, Think on | 1.0 | 0.95 | 20 | 0 | 1.5 | 1.0 |

HumanEval and MultiPL-E turn both penalties off for every model, because code repeats tokens legitimately (indentation, names, brackets).

**The thinking budget** is enforced by llama-server. When a model has thought for 4,096 tokens, the server appends *"I have used my thinking budget, so I will now give my best final answer."* and makes it answer. Chat in the UI does not cap thinking (up to the profile's 32,768-token `max_tokens`); the benchmark cap exists to bound run time, so it understates what unlimited thinking can do on hard questions.

**Sharing the GPU.** The Qwen3.6 runs shared the GPU with a running game, so they kept 18 layers of experts in system RAM (`KUMARU_BACKEND__LLAMA_SERVER__N_CPU_MOE=18`) with a 64k context. That changes speed, not the model.

## 4. Reading the numbers

**A single score** is shown as `60.0 (50.2-69.1)`: accuracy and its 95% Wilson confidence interval. At 100 questions the interval is about ±10 points in the middle of the range and narrower near 0% or 100%. So two scores 5 points apart, taken alone, tell you almost nothing.

**Comparisons are paired, and that is where the power comes from.** Both runs answered the same questions with the same seeds, so `compare` looks question by question. Questions both runs got right, or both got wrong, say nothing about which is better. Only the disagreements do. A row of `kumaru bench compare <7B run> <Qwen3.6 run>`:

```
benchmark   n     A %    B %    B - A (95% CI)           only A / only B   p (Holm)   verdict    tokens A -> B
mmlu_pro    100   59.0   84.0   +25.0 (+15.0 to +36.0)   5 / 30            8.94e-05   B better   416 -> 1,472
```

The two runs agreed on 65 questions and disagreed on 35: the 7B alone was right on 5, Qwen3.6 alone on 30. If the two models were equally good, each disagreement would be a coin flip. McNemar's exact test asks how surprising the actual split is: 35 fair coin flips come out 5 / 30 or more lopsided about twice in 100,000 tries (p = 2.2 × 10⁻⁵; the Holm correction below multiplies it by 4, giving the 8.94e-05 shown). The interval on the difference comes from a paired bootstrap, which resamples the questions 10,000 times.

**Several benchmarks means several chances of a fluke.** `compare` applies Holm's correction across the benchmarks it compares, so the chance of *any* false "better" stays under 5%. The `p (Holm)` column is already corrected.

| Verdict | Meaning |
|---|---|
| `B better` / `A better` | Significant after Holm's correction (adjusted p < 0.05). Believe it |
| `B better? (not after correction)` | p < 0.05 for this benchmark alone. A lead: re-test with more questions |
| `no clear difference` | Either no difference, or too few questions to see one |

**Rule of thumb for 100 questions:** two runs typically disagree on 20-30 questions. To be significant, those must split at least 5-15 or 9-21, which is a 10- to 12-point difference. For smaller effects, use more questions (`-n 500`, or `--full`).

**The noise floor.** Running the *same* model twice does not give the same score ([§6.2](#62-the-noise-floor-the-same-model-twice)). Two identical 7B runs differed by up to 7 points, and `compare` called every one of those differences noise. Never act on a difference that `compare` does not call significant.

## 5. Are the graders right?

A benchmark score is only as trustworthy as its grader. One that misreads `$1,080`, or a function written without its signature, silently turns right answers into wrong ones and biases against models that format differently. Every grader was checked against the dataset's own answer key before any model was scored:

| Benchmark | Check | Result |
|---|---|---|
| GSM8K | All 1,319 reference answers, each phrased 5 ways (the dataset's own solution, `Answer: N`, `**Answer:** $N,NNN`, `\boxed{N{,}NNN}`, `The answer is N.`), must grade correct; the reference answer + 1 must grade wrong | 6,595 / 6,595 correct; 1,319 / 1,319 wrong answers rejected (after the fix described below) |
| MMLU-Pro | All 12,032 gold letters, each phrased 6 ways, must grade correct; a wrong letter must grade wrong; removing junk options must never move the gold one | 72,192 / 72,192 correct; no wrong letter accepted |
| IFEval | Kumaru's implementation against Google's reference implementation, on the 541 GPT-4 responses published with the paper | Identical inputs 541/541; instruction verdicts agree 833/834; prompt-level strict 77.08 vs 76.89 and loose 79.67 vs 79.48 (Kumaru vs reference); word and sentence counts match on 2,253 texts |
| HumanEval | All 164 reference solutions, in 3 forms (whole function, body only, unindented body), must pass; a wrong solution must fail; an infinite loop must time out; the guard must block `os.remove` | 492 / 492 pass (69 s at `--workers 2`; 319 s in the Docker sandbox); all three safety checks hold, in both |
| MultiPL-E | The other languages have no reference solutions, so Kumaru re-grades answers that MultiPL-E's authors graded with their own harness. Up to 200 per language, spread over the problems: half that MultiPL-E passed, half that it failed, covering every kind of failure (wrong result, crash, compile error, timeout). Each must get MultiPL-E's verdict | Python, JavaScript, Java, C++ and Rust 200 / 200; C# 199 / 200; Go 198 / 200; TypeScript 195 / 200. The same in the Docker sandbox, except TypeScript 196 / 200. All 8 disagreements on Windows, and the 7 in Docker, are expected and explained in [§11.4](#114-are-its-graders-right) |

The one IFEval disagreement is deliberate. For "the letter *x* should appear at least *n* times" where *x* is not a letter, the reference implementation silently substitutes a random letter; Kumaru counts the character that was actually asked for.

The GSM8K check caught a real bug. The first grader compared numbers with a relative tolerance of one part in a million, so on the two largest answers in the test set, 1,450,000 and 2,880,000 (`gsm8k/611` and `gsm8k/796`), the wrong answer "reference + 1" graded correct. GSM8K answers are exact, so numbers now have to match exactly. Formatting such as `$1,450,000.00` still never matters, and a 10⁻⁹ allowance absorbs floating-point noise such as 0.1 + 0.2. Re-grading every saved GSM8K and golden answer in the runs reported here with the exact rule changed no verdict, so the bug inflated no score in this document.

Three golden format graders were later tightened, because each accepted a reply its prompt rules out: `banana-caps` took `BANANA.` although the prompt says "nothing else", `three-colors` took the same colour three times, and `json-ada` took a third key. Re-grading the 135 golden answers saved by the five runs reported here with the stricter patterns changed no verdict either.

Reproduce the first, second and fourth rows with:

```powershell
python scripts\check_graders.py        # exit code 0 = every check passed
```

and the MultiPL-E row, once the languages' toolchains are installed ([§11.2](#112-run-it)), with `python scripts\check_graders.py multipl_e --workers 2`. It downloads about 12 MB of graded answers and takes about 40 minutes.

The IFEval cross-check needs Google's reference code (`instruction_following_eval` in the `google-research` repository) in its own virtual environment: grade the GPT-4 responses published there (`data/input_response_data_gpt4_20231107_145030.jsonl`) with both implementations, then compare the verdicts instruction by instruction. The unit tests `tests/test_bench_extract.py`, `test_bench_graders.py` and `test_bench_ifeval.py` pin the edge cases found along the way.

## 6. Lab notebook: what we learned along the way

### 6.1 The truncation confound

The first full run (v1) gave each benchmark its own answer cap: 1,024 tokens for golden and GSM8K, 1,536 for HumanEval, 2,048 for MMLU-Pro and IFEval. The caps barely touched the 7B (one IFEval answer out of 427). They did touch Qwen3.6, which deliberates longer even with thinking off. On MMLU-Pro, 16 of its 100 answers were cut off mid-reasoning and 12 of those were scored wrong: more than half of its 23 misses. On GSM8K, 3 of its 7 misses were cut off. A cut-off answer usually never reaches its final line, so it is scored wrong because the model ran out of room, not because it was wrong. (A few are still graded right, when the answer appeared before the cut.)

That is a *confound*: a cap that is harmless for one model and binding for another biases the comparison against the more thorough model. The fix was one generous default for every benchmark (8,192 tokens, plus the thinking budget), and the run summary now shows a **cut off** column so a binding cap is visible instead of silent.

| Qwen3.6, Think off | v1 (capped) | v2 (8,192) | cut off, v1 → v2 |
|---|---|---|---|
| golden | 96.3 | 96.3 | 1 → 0 |
| GSM8K | 93.0 | 95.0 | 3 → 0 |
| MMLU-Pro | 77.0 | 84.0 | 16 → 1 |

(v1 was stopped during IFEval once the problem was found.)

A cap changes nothing until it binds, so the v2 answer lengths show how often each v1 cap would have bitten. For Qwen3.6, 17 MMLU-Pro answers ran past 2,048 tokens, 5 GSM8K answers past 1,024, 3 IFEval answers past 2,048 and 1 HumanEval answer past 1,536. For the 7B, one answer in 427 did (on IFEval), which is why the problem was invisible from its side.

**Follow the questions, not the score.** The headline change on MMLU-Pro, +7 points, is *not* significant on its own: the two runs disagreed on only 15 questions, split 4 / 11 (p = 0.12). The question-level records settle it:

| MMLU-Pro questions Qwen3.6 got wrong in v1 | Right in v2 |
|---|---|
| cut off by the v1 cap (12) | 10 |
| not cut off (11) | 1 |

Recovering 10 of 12 against 1 of 11 is not chance (Fisher's exact test, p = 0.0006). On the 84 questions v1 did not cut off, the two runs disagreed just 4 times (3 / 1), which is ordinary sampling noise. GSM8K shows the same pattern on a smaller scale: 2 of its 3 cut-off misses became right.

**Lesson:** check the cut-off column before believing any comparison. When an aggregate test is inconclusive but you suspect a cause, look at *which* questions changed: the records in `items.jsonl` can show a mechanism that a score hides.

### 6.2 The noise floor: the same model twice

The 7B reached a v1 cap only once (that one IFEval answer), so its v2 run is an *A/A test*: same weights, questions and seeds. Only the batch size differed (4 questions at a time in v1, 3 in v2).

| Qwen2.5-7B | v1 | v2 | only v1 right / only v2 right | verdict |
|---|---|---|---|---|
| golden | 96.3 | 96.3 | 0 / 0 | no clear difference |
| GSM8K | 96.0 | 92.0 | 5 / 1 | no clear difference |
| MMLU-Pro | 60.0 | 59.0 | 8 / 7 | no clear difference |
| IFEval | 65.0 | 72.0 | 3 / 10 | no clear difference |
| HumanEval | 80.0 | 81.0 | 2 / 3 | no clear difference |

Only 104 of the 427 answers came out identical word for word, and most of the rest had diverged within their first few hundred characters. Fixed seeds do not make GPU inference reproducible. Which requests share a batch changes the order of floating-point additions, which nudges the token probabilities, which occasionally flips a sampled token, and from there the answer diverges. (Every flipped grade came with a different answer, so the graders were not the cause.) Swings of 4 points on GSM8K and 7 on IFEval, from nothing at all, are exactly why a single run's difference means little. `compare` judged every row correctly: no clear difference.

## 7. Caveats (read before quoting these numbers)

- **Subsets.** 100 questions per benchmark detect big jumps, not small ones ([§4](#4-reading-the-numbers)). Use `--full` for numbers you want to publish.
- **Not comparable to model-card scores.** Publishers use their own prompts, few-shot examples, unquantized weights, larger budgets and sometimes several attempts per question; differences of several points are routine. Compare models *within* this harness.
- **Quantization.** Both models are ~4-bit. That costs a little accuracy against the original weights, equally for both, and it is how you run them.
- **Contamination.** GSM8K and HumanEval have been public since 2021, MultiPL-E's translations of HumanEval since 2022 and MMLU-Pro since 2024; newer models may have seen them in training. Your golden set is immune; grow it.
- **The thinking cap.** Think-on scores used a 4,096-token thinking budget to bound run time ([§3](#3-how-the-comparison-was-run)). It cut short 24 of the 100 MMLU-Pro answers, so it likely understates Think on. Chat thinks without that cap.
- **One run per configuration**, with the run-to-run noise measured in [§6.2](#62-the-noise-floor-the-same-model-twice).
- **HumanEval and MultiPL-E run model-written code** on your machine, as you. Each program gets a fresh temporary folder, a scrubbed environment and time limits (HumanEval: 15 s; MultiPL-E: 120 s to compile, 15 s to run), and everything it started is killed when it ends. On Windows its memory is also capped (4 GB). HumanEval's Python also gets a guard that disables destructive `os`/`shutil`/`subprocess` calls. That stops accidents, not a determined adversary: it is not a sandbox. `--sandbox docker` runs each program in a throwaway container with no network, a read-only root and CPU, memory and process limits ([§11.3](#113-where-the-code-runs)). Evaluate models you do not trust with it, or in a container or VM.

## 8. Your own golden questions

`data/bench/golden.jsonl` holds one question per line, graded by a rule rather than by a model:

```json
{"id": "pens", "prompt": "Pens cost $3 each, or $10 for a pack of 4. What is the least you can pay for exactly 10 pens? Finish with \"Answer: <number>\".", "grader": "numeric", "answer": 26, "tags": ["math"]}
```

Graders: `exact`, `contains`, `contains_any`, `not_contains`, `regex`, `numeric` and `choice`. `numeric` is exact unless the question sets a relative `tolerance`; use one only for answers that are approximate by nature, such as `0.001` for a rounded unit conversion (within 0.1%: for 37.8 it accepts 37.763 to 37.837, not 37.7 or 37.9). The first tag groups the question in `report`. Lines starting with `//` are comments. The full format is in [`docs/howto/src/kumaru/eval/benchmarks/golden.md`](howto/src/kumaru/eval/benchmarks/golden.md).

The habit that compounds: **whenever Kumaru gets something wrong that it should get right, add it as a golden question.** Rerun golden (about two minutes) after every change to a model, prompt or setting. Any other `.jsonl` file in the same format also works as a benchmark: `kumaru bench run my_questions.jsonl -c qwen36`.

## 9. Run mechanics

**What a run saves.** `runs\bench\<timestamp>-<label>\` holds:

- `run.json`: the model, every setting and the software versions;
- `<benchmark>\items.jsonl`: every question as sent, the full answer (and thinking), the extracted answer, the grade, token counts, latency and seed;
- `<benchmark>\summary.json`: the scores.

Every number in this document can be traced to individual answers.

**Resuming.** A run that dies (crash, reboot, Ctrl+C) resumes with `-o <its folder>` and the same flags. Answered questions are skipped, and a line half-written by a hard kill is repaired. Resuming with different settings (model, decoding parameters, seed, prompt mode, sandbox, compiler version or memory cap) is refused, so one folder never mixes two configurations.

**Useful flags for `kumaru bench run`:**

| Flag | Effect |
|---|---|
| `-c <profile>` | Which model and settings (`qwen36`, `qwen25-7b-gguf`, your own YAML) |
| `-n 100` / `--full` | Questions per benchmark, or all of them |
| `--seed 1234` | Picks the questions and each question's sampling seed. Keep it fixed across runs you will compare |
| `--think` / `--no-think` | Force thinking on or off (default: the profile's) |
| `--thinking-budget 4096` | Cap the thinking tokens per question |
| `--max-tokens N` | Override the output cap for every question |
| `--context 65536` | Shrink the context window (and the KV cache) to free VRAM |
| `-j N` | Questions in flight (default: as many as the backend can batch) |
| `--use-agent-prompt` | Include Kumaru's system prompt, as chat does |
| `--allow-code-exec` | Required for `humaneval` and `multipl_e`: they run model-written code |
| `--sandbox docker` | Run that code in a container without network instead of on your machine ([§11.3](#113-where-the-code-runs); build its image first) |
| `-o <folder>` | Write to (or resume) this folder |

**How long it takes** on the hardware above, for golden plus 100 questions of each public benchmark:

| Run | Wall time | Slowest benchmark |
|---|---|---|
| Qwen2.5-7B | 11 min | MMLU-Pro, 4 min |
| Qwen3.6, Think off | 100 min | MMLU-Pro, 56 min |
| Qwen3.6, Think on | 152 min for golden, GSM8K and MMLU-Pro (stopped there) | MMLU-Pro, 91 min |

The 9-fold gap to the 7B has two factors. Qwen3.6 generated about a quarter as many tokens per second in these runs (about 44 against 185, all requests together), partly because 18 layers of its experts ran from system RAM to leave room for a game ([§3](#3-how-the-comparison-was-run)). And it writes about twice as many tokens per answer (3.5 times as many on MMLU-Pro).

MultiPL-E's eight languages, 100 problems each, took 95 minutes with Qwen3.6 and Think off: from 7 minutes for JavaScript to 16 for Java and C#, in step with the length of their answers ([§11.5](#115-results)).

## 10. Where evaluation goes next

Evaluation has to scale with the model, or you lose the ability to tell whether changes help. In rough order of value:

1. **More questions.** `--full` runs all ~14,000 questions. It turns ±10-point intervals into about ±1 point on MMLU-Pro and GSM8K, ±4 on IFEval and ±6 on HumanEval (only 164 problems exist) and on each MultiPL-E language (152).
2. **Harder benchmarks**, as the current ones saturate: GPQA-Diamond (graduate-level science), MATH-500 and AIME (competition mathematics), LiveCodeBench (code problems published after a model's training cutoff, so contamination-resistant), BFCL (function calling, once Kumaru has tools) and RULER (long-context retrieval and reasoning).
3. **Open-ended quality.** Chat answers have no single right answer. The standard tool is a strong *LLM judge* comparing two answers pairwise, with positions swapped to cancel its bias toward the first one (Arena-Hard works this way).
4. **Test-time compute.** pass@k and majority voting over k samples measure how much extra compute buys. Thinking budgets are the first instance of this; the scores in the TL;DR are the first data point.
5. **A regression gate.** Run golden in CI on every change and fail the build if `compare` says the change made it worse.

**Adding a benchmark** takes about 50 lines: subclass `Benchmark` in `src/kumaru/eval/benchmarks/` (implement `load()` and `grade()`; optionally `aggregate()` for extra metrics and `generation_overrides()`), then register it in `src/kumaru/core/registry.py` with `BENCHMARKS.register_lazy(...)`. A separate package can also register through the `kumaru.benchmarks` entry point without touching Kumaru. Start from [`docs/howto/src/kumaru/eval/benchmarks/base.md`](howto/src/kumaru/eval/benchmarks/base.md). The evaluation axis of scaling is in [SCALABILITY.md §7](SCALABILITY.md#7-evaluation-scale).

## 11. Programming languages: MultiPL-E

HumanEval asks for Python only. It cannot say how well the model writes JavaScript for a website, C# for a game or Rust for an app, or which of those languages it is weak in. MultiPL-E (Cassano et al., 2023) translated HumanEval's problems, docstrings and unit tests into other languages. Kumaru runs seven of them, chosen for building apps, websites and games, plus Python as the baseline:

| Benchmark | Language | Needs | Checked here with |
|---|---|---|---|
| `multipl_e_py` | Python | nothing: the Python that runs Kumaru | Python 3.12.6 |
| `multipl_e_js` | JavaScript | Node.js 18+ | Node.js 24.17.0 |
| `multipl_e_ts` | TypeScript | Node.js 22.7+; `tsc` to check types | Node.js 24.17.0, without `tsc` |
| `multipl_e_cs` | C# | the .NET 10 SDK | .NET SDK 10.0.301 |
| `multipl_e_java` | Java | a JDK 17+ and `javatuples-1.2.jar` | Temurin JDK 21.0.12.1 |
| `multipl_e_cpp` | C++ | g++ (the tests need GCC's standard library) | g++ 16.2.0 (WinLibs) |
| `multipl_e_go` | Go | Go | Go 1.27.3 |
| `multipl_e_rs` | Rust | `rustc` | rustc 1.99.1 |

Where to get each: [`multipl_e.md`](howto/src/kumaru/eval/benchmarks/multipl_e.md#install-the-toolchains).

### 11.1 What it measures

- **The same problems in every language.** MultiPL-E has no translation of 12 of HumanEval's 164 problems in at least one of these languages, so every language gets the other 152, in the same order. With the same `--seed`, `-n 100` asks every language the same 100 problems. A language that scores well below Python is then a weak language, not a harder test.
- **Working code.** Each answer is compiled and run against the translated unit tests, and scores only if every test passes (pass@1). A failure is recorded as `failed` (a test failed or the program crashed), `compile_error` or `timeout`, so `report` shows how a language goes wrong: `tests: passed 120, failed 20, compile_error 9, timeout 3`.
- **`multipl_e_py` is not `humaneval`.** It asks the 152 shared problems with the same instruction as the other languages, which makes it their baseline. `humaneval` asks all 164 with its own instruction. Do not compare one's score with the other's.
- **What it does not measure.** These are short, single-function problems that use only the standard library. Building an app, a website or a game also takes frameworks, many files, build tools, user interfaces and debugging, none of which MultiPL-E tests. A low score flags a language to be careful with. A high score shows the model writes correct code in that language, not that it can build a whole application in it.

### 11.2 Run it

```powershell
kumaru bench run multipl_e -c qwen36 --no-think --allow-code-exec -n 100            # every language with a toolchain
kumaru bench run multipl_e_rs multipl_e_go -c qwen36 --allow-code-exec --full       # chosen languages, all 152 problems
kumaru bench report runs\bench\<run folder>
```

`multipl_e` (or `multipl-e`) expands to the eight benchmarks. Before the model loads, each language grades a known right and a known wrong answer with its toolchain. A language whose toolchain is missing, or fails that self-test, is skipped with how to fix it:

```
skipped multipl_e_rs: Rust needs rustc, which is not on PATH. Install Rust: https://rustup.rs
```

It is listed again under `not run:` when the run ends, and is never scored 0. A language named on its own (`multipl_e_rs`) stops the run with that message instead.

Programs compile and run on the CPU, at most `KUMARU_CODE_WORKERS` at a time (default: half the cores, from 1 to 4). Lower it to leave room for a game, or if compiles start timing out.

### 11.3 Where the code runs

By default, on your machine: `--allow-code-exec` is your consent. Every program, HumanEval's included, goes through [`codeexec`](howto/src/kumaru/eval/benchmarks/codeexec.md):

- a fresh temporary folder, deleted afterwards, and a scrubbed environment (no API keys);
- 120 s to compile and 15 s to run. When a step ends or times out, everything it started is killed, not just its own process;
- on Windows, a job object per step, which caps its memory at 4 GB (`KUMARU_CODE_MEMORY_MB`) and its processes at 512, and suppresses crash dialogs;
- compilers and runtimes found only in `PATH`'s absolute folders, never in the current folder, and run by absolute path;
- a random marker placed after the last test. A program passes only if it exits with 0 *and* prints the marker, so code that exits early cannot pass untested.

That stops accidents, not adversaries: the program runs as you, can read your files and can reach the network.

`--sandbox docker` runs each program in a throwaway container instead: no network, a read-only root, an unprivileged user without capabilities, 2 CPUs, 2 GB of memory and 512 processes. That is a real boundary. The image, built from [`docker/sandbox/Dockerfile`](howto/docker/sandbox/Dockerfile.md), carries every toolchain, because nothing can be downloaded inside it. The sandbox and the compiler versions are part of a run's fingerprint, so one run never mixes programs graded inside and outside it.

**Status: built and checked** on 2026-10-23, with Docker Desktop 4.39.0 on WSL 2. The image built the first time, as written: 3.54 GB, on Ubuntu 24.04.5, with Python 3.12.3, Node.js 24.22.1 with TypeScript 5.9.3, .NET SDK 10.0.402, OpenJDK 21.0.12.1, g++ 13.3.0, Go 1.25.14 and rustc 1.90.0 (the Ubuntu packages aren't pinned; [`Dockerfile.md`](howto/docker/sandbox/Dockerfile.md#what-it-does) lists their exact versions). Its graders agree with HumanEval's and MultiPL-E's as closely as the local ones do, and in it, C++'s `HumanEval_114` and `HumanEval_139`, which Windows's 32-bit `long` breaks, get MultiPL-E's verdict on all 100 of its published answers ([§11.4](#114-are-its-graders-right)). Starting a container adds about a second to each program. Building takes about 11 GB of Docker's disk ([`Dockerfile.md`](howto/docker/sandbox/Dockerfile.md#how-to-use-it)). Build it, check its graders, then use it:

```powershell
docker build -t kumaru-sandbox -f docker/sandbox/Dockerfile .
python scripts\check_graders.py humaneval multipl_e --sandbox docker --workers 2
kumaru bench run humaneval multipl_e -c qwen36 --no-think --allow-code-exec --sandbox docker
```

### 11.4 Are its graders right?

Turning a chat model's reply into a program is where a code grader goes wrong. Models add imports, helper functions, a `main`, their own tests and example calls, and any of those can clash with the hidden tests. Each language's assembler keeps what the code needs and drops the rest ([`multipl_e.md`](howto/src/kumaru/eval/benchmarks/multipl_e.md#how-each-languages-program-is-assembled)); a mistake there fails working code.

These languages have no reference solutions to check against. Instead, MultiPL-E's authors published answers they had graded with their own harness ([`nuprl/MultiPL-E-completions`](https://huggingface.co/datasets/nuprl/MultiPL-E-completions)): StarCoder2-15B's, and for Python replit-code-v1-3b's. `scripts\check_graders.py` replays up to 200 per language through `kumaru bench`'s own path (extraction, assembly, compile, run) and asks for MultiPL-E's verdict. The sample is spread over the problems. Half are answers MultiPL-E passed; the other half covers every way it saw them fail (a wrong result or a crash, a compile error, and up to 10 timeouts). Measured on Windows 11, with 2 programs at a time, on a CPU shared with a model:

| Language | Same verdict as MultiPL-E | Problems | Toolchain | Time |
|---|---|---|---|---|
| Python | 200 / 200 | 104 | Python 3.12.6 | 48 s |
| JavaScript | 200 / 200 | 129 | Node.js 24.17.0 | 91 s |
| TypeScript | 195 / 200; the other 5 expected | 120 | Node.js 24.17.0, types stripped | 115 s |
| C# | 199 / 200; the other one expected | 121 | .NET SDK 10.0.301 | 780 s |
| Java | 200 / 200 | 132 | Temurin JDK 21.0.12.1, javatuples 1.2 | 208 s |
| C++ | 200 / 200 | 126 | g++ 16.2.0 (WinLibs, MinGW-w64 UCRT) | 489 s |
| Go | 198 / 200; the other 2 expected | 125 | Go 1.27.3 | 274 s |
| Rust | 200 / 200 | 133 | rustc 1.99.1 | 186 s |

The same 200 answers per language, graded in the Docker sandbox ([§11.3](#113-where-the-code-runs)) on Linux, with the image's toolchains, 2 at a time:

| Language | Same verdict as MultiPL-E | Toolchain | Time |
|---|---|---|---|
| Python | 200 / 200 | Python 3.12.3 | 161 s |
| JavaScript | 200 / 200 | Node.js 24.22.1 | 191 s |
| TypeScript | 196 / 200; the other 4 expected | Node.js 24.22.1, `tsc` 5.9.3 | 336 s |
| C# | 199 / 200; the other one expected | .NET SDK 10.0.402 | 403 s |
| Java | 200 / 200 | OpenJDK 21.0.12.1, javatuples 1.2 | 184 s |
| C++ | 200 / 200 | g++ 13.3.0 | 276 s |
| Go | 198 / 200; the other 2 expected | Go 1.25.14 | 141 s |
| Rust | 200 / 200 | rustc 1.90.0 | 191 s |

The image's Go and Rust are older than the ones used on Windows (1.25 against 1.27, and 1.90 against 1.99), and its g++ is 13 against 16; no verdict changed. None of the disagreements is a grader error. Each comes from a different toolchain or from a deliberate difference from MultiPL-E's harness (listed below):

- **TypeScript, 5 on Windows and 4 in Docker.** MultiPL-E checks types with `tsc`, which rejected these 5 answers. In 4 of them, what it rejected was a line after the function: a file path such as `/src/kata/7kyu/sum.ts`, which `tsc` reads as a regular expression, or an example call cut off mid-string. Kumaru keeps only an answer's declarations and imports (below), so it drops that line, and the tests pass, with `tsc` or without. The 5th has a type error in the function itself. Without `tsc`, Kumaru runs TypeScript with Node's built-in type stripping, which doesn't check types, so its tests pass; in the Docker sandbox, `tsc` rejects it, as MultiPL-E's did.
- **C#, 1.** An answer to `HumanEval_94` timed out under MultiPL-E's 5 s limit on Mono. On .NET 10 it takes about 4 s and passes, on Windows and in Docker.
- **Go, 2.** An answer to `HumanEval_114` calls `min`, which has been built into Go since version 1.21; MultiPL-E's older Go rejected it. An answer to `HumanEval_136` timed out under MultiPL-E's 30 s limit, which covers building as well as running; here it builds and passes. Both are the same on Windows and in Docker.

The check grades each answer on the tests as MultiPL-E published them, because those gave its verdicts, so the comparison is like with like. `kumaru bench run` grades C# on fixed tests instead (below); on the published ones, MultiPL-E and Kumaru fail a right List or Dictionary result alike. C++'s 200 happen to include none of the 24 answers to `HumanEval_114` that MultiPL-E passed. On Windows those fail (below), so a larger sample would show disagreements that the check doesn't yet explain. In the Docker sandbox, all 100 of MultiPL-E's answers to 114 and 139 get its verdict.

Agreement compares verdicts, pass or fail. The kind of failure can differ without changing a verdict. Of the 10 C# timeouts sampled, for example, Kumaru reports 7 as plain failures: 4 programs stop at their first failed `Debug.Assert`, where Mono's carries on, and 3 runaway loops hit the 4 GB memory cap before the time limit.

**What the check does not cover.** MultiPL-E's answers continue its prompt, as base models do, so they rarely bring the extras a chat model's answer does. How the assemblers handle those is covered by unit tests and by each language's self-test, not by this agreement. The check caught two assembler bugs. Go's `[]interface{}` was taken for a block, which cut functions in two. And in JavaScript and TypeScript, a function body that names a local variable after its function (`let intersection = …` inside `intersection`) was read as the whole function, so the program left the function out. Agreement missed that one: it touched 2 of the check's 200 answers per language, which fail either way. In the Docker sandbox, `tsc` rejected TypeScript's 2 for a type error, where MultiPL-E had run them, and that gave it away. Of all MultiPL-E's completions, it misread 143 (81 JavaScript, 62 TypeScript), and failed the 52 of them that MultiPL-E passed. Read correctly, all 143 get MultiPL-E's verdict, on Windows and in the sandbox. None of Qwen3.6's answers was affected: they all define their function. Tests of whole answers caught two more bugs: a JavaScript `import { … } from` was cut at its `}`, and a Java or C# `void main` was missed, which left two `main`s. All four are fixed and pinned by unit tests. After a run, read a few of each language's `compile_error`s in its `items.jsonl` before believing a low score: each keeps its `mode`, its `stage` and the compiler's output. In Qwen3.6's run ([§11.5](#115-results)), every compile error was the model's own, apart from Go's on `HumanEval_87`, a defect (below).

**Deliberate differences from MultiPL-E's harness.**
- Answers may be whole functions, as chat models write them, as well as continuations.
- JavaScript, TypeScript and C# keep only an answer's declarations and imports, so a stray statement after the function, such as an example call, can't fail it.
- Time limits: 120 s to compile and 15 s to run, where MultiPL-E allows 5 to 30 s by language.
- C# builds with .NET 10 instead of Mono.
- Rust compiles as edition 2021 with warnings allowed, instead of 2015.
- TypeScript is type-checked only when `tsc` is installed.
- On Windows, C++ and Rust get the 8 MB stack Linux gives them, and C++ links statically.
- C#'s tests of a List or Dictionary result are fixed, as MultiPL-E has since fixed its translator (below).

The full table is in [`multipl_e.md`](howto/src/kumaru/eval/benchmarks/multipl_e.md#differences-from-multipl-es-own-harness).

**Defects in the dataset.** The pinned dataset (revision 28441b6, July 2025, still the newest on the Hub) has tests that no answer can pass, and Windows adds two in C++. Kumaru applies only fixes MultiPL-E has made itself, and only when they treat every language alike:

- **Fixed: C#'s List and Dictionary results, 43 of the 152 problems** ([nuprl/MultiPL-E#71](https://github.com/nuprl/MultiPL-E/issues/71)). The tests compare them with `.Equals`, which .NET (and Mono) answers by reference, so a right answer fails: `new List<long>().Equals(new List<long>())` is false. MultiPL-E fixed its translator in January 2026 (commit 85833ac) but has not regenerated the dataset. Kumaru rewrites each of those assertions as the fixed translator writes them (`SequenceEqual` for a List; `Count` and `Except` for a Dictionary), and refuses to load unless exactly those 43 problems change. Unfixed, an answer can pass at most 107 of C#'s 152 problems; fixed, 150, on a par with the other languages. C# results graded before the fix can't be resumed with it.
- **Not fixed: `HumanEval_92` (`any_int`), in all 7 translations.** The answer turns on telling `3` from `3.0`, which Python can and the translations can't, so each asks for the same call to be both true and false: JavaScript and TypeScript have one number type, C#, Java, C++ and Go pass floating-point numbers, and Rust asserts `candidate(3.0, 4.0, 7.0)` both ways. MultiPL-E's fix ([#142](https://github.com/nuprl/MultiPL-E/issues/142)) drops contradictory assertions only when they are written alike, which only Rust's are. Kumaru doesn't apply it: Rust alone would gain a problem that the other six can't pass, whereas 92 unsolvable in all seven favours none of them. #71, by contrast, puts C# back on a par with the rest.
- **Not fixed: `HumanEval_136` (`largest_smallest_integers`), in C# and Java.** The function returns a pair of optional numbers, and the tests compare it with values of other types, which are never equal. C#'s expect values such as `Tuple.Create((Nullable<long>)null, 1L)`, a `Tuple<Nullable<long>,long>`. Java's wrap them once too often: `Optional.of(Optional.empty())` for a missing number, and `Optional.of(Pair.with(-2l, 1l))` for a whole pair. MultiPL-E fixed Java's other `Optional` tests ([#47](https://github.com/nuprl/MultiPL-E/issues/47)), but its translator still writes these; it hasn't fixed C#'s either.
- **Not fixed: `HumanEval_51` (`remove_vowels`), in JavaScript.** A test string holds a raw line break, a syntax error in JavaScript.
- **Not fixed: `HumanEval_87` (`get_row`), in Go.** The function returns pairs, which Go's translation writes as `[]interface{}`. But some expected values put those pairs inside a `[][]int` literal, and one test passes a `[]interface{}` where the function takes `[][]int`. The test file never compiles, so no answer passes. MultiPL-E has no issue for it.
- **Can't pass on Windows: `HumanEval_114` (`minSubArraySum`) and `HumanEval_139` (`special_factorial`), in C++.** The C++ translations write Python's integers as `long`, which is 64 bits on Linux, where MultiPL-E grades, but 32 bits on Windows, with MinGW as with MSVC. 114's test passes −9999999999999999 in a `std::vector<long>`, which cuts it to 32 bits before the call, so no answer passes on Windows. 139's expects 125411328000, so an answer passes only if it widens the prompt's `long` return type to `long long`. A right answer graded by hand bears both out. On Linux both can pass, as they do in the Docker sandbox: there, all 50 of MultiPL-E's published answers to each problem get its verdict (24 of 114's pass, none of 139's), and right answers that keep `long` pass, Qwen3.6's to 139 among them ([§11.5](#115-results)). No other C++ test holds a number beyond 32 bits.

So with all 152 problems, the best possible score is 152 in Python; 151 in TypeScript and Rust, and in C++ on Linux; and 150 in JavaScript, C#, Java and Go, and in C++ on Windows, where 139 also needs that wider type. The fix for #71 and the scan for numbers beyond 32 bits covered all 152 problems. The other defects were found by reading every failure of the run below, which asked 100 of them, and by grading a right answer written by hand wherever no published answer had passed either; the other 52 problems haven't been audited, so a language's best score may be lower.

### 11.5 Results

Qwen3.6-35B-A3B with Think off, on the same 100 problems in every language, with the settings of [§3](#3-how-the-comparison-was-run) and the toolchains in [§11](#11-programming-languages-multipl-e)'s table. The programs ran on the machine itself, without `--sandbox`, 2 at a time.

| Language | pass@1 (95% CI) | Best possible | The model's mistakes | Defects ([§11.4](#114-are-its-graders-right)) | Tokens per answer | Minutes |
|---|---|---|---|---|---|---|
| Python | 98.0 (93.0–99.4) | 100 | 64, 92 | none | 301 | 10.5 |
| JavaScript | 94.0 (87.5–97.2) | 98 | 95, 113, 126, 134 | 51, 92 | 194 | 7.0 |
| TypeScript | 96.0 (90.2–98.4) | 99 | 95, 126, 134 | 92 | 256 | 9.1 |
| C# | 94.0 (87.5–97.2) | 98 | 49\*, 129\*, 132\*, 134 | 92, 136 | 437 | 15.9 |
| Java | 93.0 (86.3–96.6) | 98 | 54, 95, 113, 129\*, 134 | 92, 136 | 448 | 16.0 |
| C++ | 95.0 (88.8–97.8) | 99 | 75\*, 134, 156\* | 92; 139 on Windows | 352 | 12.9 |
| Go | 93.0 (86.3–96.6) | 98 | 7\*, 29\*, 95, 126, 134\* | 87\*, 92 | 368 | 13.2 |
| Rust | 92.0 (85.0–95.9) | 99 | 37, 68\*, 95, 121\*, 122\*, 126, 159 | 92 | 281 | 9.9 |

- Problems are numbered as in HumanEval (134 is `HumanEval_134_check_if_last_char_is_a_letter`), and \* marks a compile error.
- **Best possible** counts the problems among these 100 that an answer can pass. C++'s includes 139, which passes on Windows only if the answer widens the prompt's `long` return type to `long long`. Qwen3.6's answer kept the signature, as the instruction asks, so its failure counts as a defect, not a mistake. On Linux it passes (below).
- **The model's mistakes** are failures that some answer shows to be passable: Qwen3.6's own in its other C# run, an answer MultiPL-E published, or a right answer written by hand.
- **TypeScript ran with its types stripped** (no `tsc`), so a type error would not have failed it. Graded again with `tsc`, no answer had one (below).
- **C#'s row comes from a second run**, after the fix to its tests ([§11.4](#114-are-its-graders-right)). The next paragraph compares it with the first.

**C#, published tests against fixed ones.** The first run predates the fix and graded C# on the tests as published: 70.0 (60.4–78.1), 28 points below Python. All 27 of the 100 problems with a List or Dictionary result failed, right or wrong. Regraded on the fixed tests by the same grader, the same answers score 96.0 (90.2–98.4): 26 of those 27 pass, and the 27th, 74, returns the second list on a tie where the docstring asks for the first. No verdict changed on the other 73 problems. The second run, with the fix, gave the table's 94.0. Only 25 of its answers match the first run's word for word (and get the same grades), and the two runs' mistakes don't overlap: only the first run's answers pass 49, 129, 132 and 134, only the second's 74 and 95 (p = 0.688). That is the run-to-run noise of [§6.2](#62-the-noise-floor-the-same-model-twice).

**Graded again in the Docker sandbox.** Both runs' 900 answers, as stored, were graded again in the Docker sandbox ([§11.3](#113-where-the-code-runs)), on Linux with the image's toolchains, 2 at a time, in 15 minutes. Every verdict and every kind of failure stood but one: C++'s 139, whose answer kept the prompt's `long`, passes where `long` is 64 bits. So on Linux, C++ scores 96.0 (90.2–98.4), and is 3 to 1 against Python (p = 0.625). TypeScript was type-checked this time, by `tsc` 5.9.3, and all 100 of its verdicts stood, so none of its answers had a type error.

**What it shows.**

- **No language is clearly weaker than Python.** Problem by problem (McNemar's exact test, [§4](#4-reading-the-numbers)), Rust comes closest: Python alone passed 7 problems and Rust alone 1, p = 0.0703, or 0.492 after Holm's correction for seven languages. Java and Go are 6 to 1 (p = 0.125), JavaScript and C# 5 to 1 (p = 0.219), C++ 4 to 1 (p = 0.375) and TypeScript 3 to 1 (p = 0.625). The 1 is always 64, the only problem that Python failed and every other language passed. A hundred problems detect only large gaps ([§7](#7-caveats-read-before-quoting-these-numbers)); `--full` asks all 152.
- **Four problems account for 17 of the 31 mistakes in the translations**, and Python passed all four:
  - **134**, in six languages. Every failing answer strips trailing spaces before it looks at the last character, though the docstring's own example says `"apple pi e "` gives false; Go's also holds two attempts and doesn't compile. Python's answer stripped them too, and passed only because its last check looked at the unstripped text.
  - **95**, in five, and in C#'s first run. Python's test has an integer key, `5`, which the translations had to write as the string `"5"`. Python's answer turned the integer away for not being a string. Every failing answer took `"5"`, which has no letters, for a lower-case key, so the test's dictionary passed as all lower case. The prompt's own example, a key `8` that makes the answer false, hints at the rule. It is passable, so not a defect, but only the translations set this trap.
  - **126**, in four: "more than 1 duplicate of the same number" read as any duplicate, though the docstring's own example says `[1, 2, 2, 3, 3, 4]` is sorted.
  - **113**, in two: the expected sentence is the docstring's, with each "i" replaced by a count. JavaScript's answer wrote "o" where the "i"s go and replaced every "o", and Java's put spaces beside the "i"s.
- **Half the mistakes in the compiled languages are compile errors**, 12 of 24 in C#, Java, C++, Go and Rust: a name declared twice or reused from an enclosing scope (C# 129 and 132, Java 129, Go 134); a parameter the prompt names `strings`, which hides Go's `strings` package (Go 7 and 29); a type error (C# 49's `BigInteger` not cast to `long`, Rust 68's `usize` where an `isize` was due, Rust 121's `%` on a `&&isize`); an assignment to an immutable argument (Rust 122); a header that doesn't exist (C++ 156's `<pair>`); and `vector` without `std::` (C++ 75). The compiler points at each, so a model that compiled its code and read the errors could fix them, but the benchmark grades the first answer as written. One wrong answer is a Rust trap: 159 calls `saturating_sub` on a signed `isize`, which saturates at `isize::MIN`, not at 0, so it returned −6 carrots left where 0 was due.
- **Time follows answer length.** The answers came at 45 to 48 tokens a second, all requests together, and the programs compiled and ran alongside, so each language took about as long as its answers took to write: 7 minutes for JavaScript's 194 tokens per answer, 16 for Java's 448.

**The runs.**

```powershell
$env:KUMARU_BACKEND__MODELS_DIR = 'D:\kumaru\models'      # where the model is stored
$env:KUMARU_BACKEND__LLAMA_SERVER__N_CPU_MOE = '18'       # the GPU split of §3
$env:KUMARU_CODE_WORKERS = '2'                            # programs at a time (§11.2)
kumaru bench run multipl_e -c qwen36 --context 65536 --no-think -n 100 --allow-code-exec --out runs\bench\20261022-082009-qwen36-nothink
kumaru bench run multipl_e_cs -c qwen36 --context 65536 --no-think -n 100 --allow-code-exec --out runs\bench\20261022-095557-qwen36-nothink-cs
```

The second, after the fix, gave the C# row. The first took 95 minutes, the second 16. Neither had an error, an answer cut off at the token cap, a timeout or a skipped language.

## Related files

- `src/kumaru/eval/benchmarks/`: the benchmarks, the runner and the statistics, each with a how-to page under `docs/howto/src/kumaru/eval/benchmarks/`
- `data/bench/golden.jsonl`: your golden set
- `scripts/check_graders.py`: re-validates the graders
- `configs/qwen36.yaml`, `configs/qwen25-7b-gguf.yaml`: the two profiles compared here
- `docs/SCALABILITY.md` §7: evaluation as a scaling axis
