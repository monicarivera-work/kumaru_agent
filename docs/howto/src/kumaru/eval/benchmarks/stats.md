# `src\kumaru\eval\benchmarks\stats.py`

> Puts error bars on a score and tests whether a difference is real: the Wilson score interval, the exact McNemar test, a paired percentile bootstrap, and the Holm–Bonferroni correction. Uses only the standard library; numpy makes the bootstrap faster.

**Read this when:** you are deciding how many questions to run, reading a `95% CI` or `p (Holm)` column, or tempted to call a 3-point gain an improvement.

---

## What it does

| Function | Exact method | Used for |
|---|---|---|
| `wilson_interval` | **Wilson score interval** for a binomial proportion, z = 1.96 (95%). | The `95% CI` column of `bench run`/`report`, and the per-run CIs inside a comparison. |
| `mcnemar_exact` | **Exact McNemar test**, two-sided: an exact binomial (sign) test on the discordant pairs, p = min(1, 2·Σ_{i≤min(b,c)} C(b+c, i) / 2^(b+c)). | The raw p of `bench compare` (`Comparison.p_value`, not shown as a column; it drives the "(not after correction)" verdict). |
| `paired_bootstrap_ci` | **Paired percentile bootstrap** of the mean difference (B − A): 10,000 resamples of whole questions, seed 0. | The `B - A (95% CI)` column. |
| `holm_adjust` | **Holm–Bonferroni** step-down adjustment. | `p (Holm)` across the benchmarks of one comparison. The verdict reads this value. |

## Why it exists

A score on 100 questions is an estimate. Re-run on 100 other questions from the same benchmark and it moves by several points purely by chance. Without error bars, every change looks like either progress or a regression.

### The intuition behind each method

- **Wilson, not "p ± 1.96·√(p(1−p)/n)".** The textbook interval assumes the estimate is roughly normal. That breaks near 0% and 100%, exactly where good models score: at 99/100 its upper bound passes 100%, and at 100/100 it has zero width. Wilson's interval stays in [0, 1], is asymmetric near the edges, and has close to the advertised 95% coverage even for small n.
- **McNemar, because the runs are paired.** Both runs answered the same questions (same `-n`, `--seed`). Questions both got right, or both got wrong, say nothing about which model is better. Only the disagreements do: b = only A right, c = only B right. If there were no real difference, each disagreement would be a fair coin flip, so the p-value is a binomial tail. Pairing removes "this sample was easy" from the comparison. That makes it far more sensitive than comparing two independent CIs.
- **Bootstrap for the size of the gain.** Resampling questions with replacement, keeping each question's A and B results together, shows how much B − A would move with a different sample of questions. It reports how big the gain is. McNemar reports whether there is one. With numpy, the code draws how many −1/0/+1 differences each resample contains (a multinomial). That is equivalent to drawing questions and about 1000× faster. Without numpy, it falls back to `random.choices` and still draws every requested resample. In pure Python that is slow: at the default 10,000 resamples, about 1 second for GSM8K's 1,319 questions and about 9 seconds for MMLU-Pro's 12,032.
- **Holm, because you test five benchmarks at once.** At p < 0.05 per test, the chance that at least one of five benchmarks "wins" by luck is about 23%, not 5%. Holm multiplies the smallest p by m, the next by m − 1, and so on, keeping the order monotone. That keeps the family-wise false-alarm rate at 5%, and it is never less powerful than plain Bonferroni.

## Mental model: what the numbers mean in practice

**Wilson 95% intervals:**

| n | at 50% | at 70% | at 90% |
|---|---|---|---|
| 27 (golden) | 34.0–69.3 | 51.5–84.1 | 71.9–96.1 |
| 100 (default `-n`) | 40.4–59.6 | 60.4–78.1 | 82.6–94.5 |
| 200 | 43.1–56.9 | 63.3–75.9 | 85.1–93.4 |
| 500 | 45.6–54.4 | 65.8–73.9 | 87.1–92.3 |
| 1000 | 46.9–53.1 | 67.1–72.8 | 88.0–91.7 |

- **At n = 100, a 95% CI is about ±9–10 points for accuracies near 50–70%.** Two runs at 64% and 70% are therefore not distinguishable by their CIs.
- Overlapping CIs do **not** prove "no difference". Run a paired comparison instead.
- About 320 questions get you to ±5 at 70%, and about 900 to ±3.

**McNemar, the least uneven split that is still significant** (two-sided p < 0.05):

| disagreements (b + c) | least uneven significant split | net gain needed on n = 100 |
|---|---|---|
| 6 | 0 : 6 (p = 0.031) | 6 points |
| 10 | 1 : 9 (p = 0.021) | 8 points |
| 20 | 5 : 15 (p = 0.041) | 10 points |
| 30 | 9 : 21 (p = 0.043) | 12 points |
| 40 | 13 : 27 (p = 0.038) | 14 points |

So on 100 paired questions, a real gain shows up at about 8–14 points, depending on how much the models disagree. Smaller gains need more questions (`-n 500` or `--full`). With 5 or fewer disagreements nothing can be significant: even 0 : 5 gives p = 0.0625. Holm then multiplies the smallest of five p-values by 5.

**The CI and the verdict can disagree at the margin.** With 0 : 5 disagreements on 100 questions, the bootstrap gives +5.0 (+1.0 to +10.0): it excludes zero. Yet McNemar p = 0.0625. A percentile bootstrap is slightly optimistic when the discordant count is tiny. The verdict follows the exact, Holm-corrected p.

**The intervals cover sampling noise only.** Systematic error doesn't widen them: a binding token cap, a broken chat template or an extraction failure moves accuracy while the CI stays as narrow as ever. The commonest is the cap. Every benchmark's default answer budget is 8,192 tokens (plus the thinking budget when thinking is on), and an answer cut there is graded as it stands and usually fails. `n_truncated`, the **cut off** column, is how you spot a binding cap. Check it, and the `answers found by:` line (`tests:` for HumanEval), before reading a CI or a verdict. If one side of a comparison has many more cut-offs, the comparison measures the budget, not the model.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `Z95` | `float = 1.959963984540054` | Two-sided 95% normal quantile. |
| `wilson_interval` | `wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]` | `(low, high)` as fractions. `n <= 0` returns `(0.0, 1.0)`. |
| `mcnemar_exact` | `mcnemar_exact(b: int, c: int) -> float` | Two-sided exact p. `b + c == 0` returns `1.0`. Exact integer arithmetic, so it never underflows. |
| `paired_bootstrap_ci` | `paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], *, resamples: int = 10_000, seed: int = 0, confidence: float = 0.95) -> tuple[float, float, float]` | `(mean(b) - mean(a), low, high)`. Unequal lengths raise `ValueError`; empty input returns `(0.0, 0.0, 0.0)`. Draws exactly `resamples` resamples, with or without numpy. Deterministic for a given seed. |
| `holm_adjust` | `holm_adjust(pvalues: Sequence[float]) -> list[float]` | Adjusted p-values in input order, monotone, capped at 1. |

## How to use it

### Through the CLI (the normal path)

```powershell
kumaru bench compare runs\bench\<A> runs\bench\<B>
```

`compare_runs` pairs questions by id. It computes the bootstrap delta on the boolean `correct` (for IFEval that is prompt-level strict), McNemar on the discordant counts, and Holm across the benchmarks the two runs share. See [`runner.md`](runner.md#compare-two-runs) for the table and the verdict rules.

### In Python, for numbers outside a run

```python
from kumaru.eval.benchmarks.stats import wilson_interval, mcnemar_exact, paired_bootstrap_ci, holm_adjust

wilson_interval(70, 100)          # (0.604..., 0.781...)
mcnemar_exact(6, 23)              # 0.0023  (6 only-A vs 23 only-B)
holm_adjust([0.01, 0.04, 0.03])   # [0.03, 0.06, 0.06]

a = [1]*60 + [0]*40               # baseline right/wrong per question
b = [1]*55 + [0]*5 + [1]*22 + [0]*18
paired_bootstrap_ci(a, b)         # (0.17, 0.07, 0.27)  -> +17 points, CI +7 to +27
```

### Plan the sample size before a run

- Accuracy CI: aim for the width you need. `-n 100` gives ±9–10 near 50–70%, `-n 500` gives ±4.
- A/B decision: assume the models disagree on about 25% of questions. Then a 5-point gain needs roughly 500 paired questions to reach p < 0.05 (125 disagreements split 50 : 75 give p = 0.031; at 400 questions, 40 : 60 gives p = 0.057). Use `-n 500`, or `--full` when the benchmark is small (IFEval 541, HumanEval 164).
- Same `-n` and `--seed` for both runs, or the comparison loses pairs (`unpaired` in the table).

## How to extend it

- **Another metric to test** (a graded score, not right/wrong): `paired_bootstrap_ci` accepts any floats. McNemar is only for binary outcomes. Use a sign test or a permutation test, and add it here with a test.
- **Different confidence level:** `wilson_interval(z=...)` and `paired_bootstrap_ci(confidence=...)` take it. The runner and table headers assume 95%.
- **Keep it dependency-free:** the module must work without numpy. `tests\test_bench_stats.py::TestPairedBootstrap::test_pure_python_fallback_agrees_with_numpy` pins that the two paths agree, and `test_pure_python_fallback_draws_every_requested_resample` pins that the fallback honours `resamples` (it once capped every request at 2,000) and stays deterministic.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| B is 4 points better but the verdict is "no clear difference" | Expected at n = 100 (see the table). | Run more questions with the same seed, then compare again. |
| Verdict "B better? (not after correction)" | Raw p < 0.05 but Holm-adjusted p ≥ 0.05. | Treat it as unconfirmed. Re-test that benchmark alone on a larger `-n`; a pre-registered single test needs no correction. |
| CI excludes 0 but verdict says no difference | Small discordant counts; the bootstrap is optimistic (example above). | Trust the exact p. |
| A significant "A better" where B shows `cut off` > 0 | B's answers hit the token cap (8,192 by default, plus any thinking budget) and failed. The test is right about the scores, but the cause is the setting. | Re-run B in a new directory with a larger `--max-tokens` or `--thinking-budget` (a resume refuses changed budgets), then compare again. |
| `ValueError: paired samples must have equal length` | Lists passed in different lengths. | Pair by question id first (as `compare_runs` does). |
| Bootstrap CI differs slightly between machines | numpy is missing on one of them, so it falls back to pure-Python resampling: the same number of resamples, but a different random stream and percentile rule. | Install numpy (the `bench` and `dev` extras include it). The test suite pins the two paths to within 4 points of each other at the CI edges; the delta itself is identical. |
| `bench compare` takes seconds per benchmark | numpy is missing, so the bootstrap runs in pure Python (about 9 seconds for 12,000 questions at 10,000 resamples). | Install numpy: `pip install -e ".[bench]"`. |

## Related files

- `src\kumaru\eval\benchmarks\runner.py` — `summarize` (Wilson), `compare_runs` and `verdict` ([`runner.md`](runner.md)).
- `tests\test_bench_stats.py` — `TestWilsonInterval` (pins 50/100 → 0.4038–0.5962 and 96/100 → 0.9016–0.9843), `TestMcNemarExact`, `TestPairedBootstrap` and `TestHolmAdjust`.
- [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md) — results, each with its CI.
