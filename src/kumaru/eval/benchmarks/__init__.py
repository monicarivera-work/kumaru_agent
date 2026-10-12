"""Public benchmarks with honest statistics: `kumaru bench`.

`kumaru eval` answers "is this setup working, and how fast?". This package
answers "how *good* is the model, and did a change really help?" — on public
datasets, with every answer kept for audit and every comparison tested for
significance.

Modules, in reading order:

* `base`      — the `Benchmark` contract and the deterministic question sampler.
* `extract`   — pulling the final answer out of free-form model text.
* `stats`     — confidence intervals and paired significance tests.
* `gsm8k`, `mmlu_pro`, `ifeval` (+ `ifeval_checks`), `humaneval`, `golden`
              — one module per benchmark.
* `runner`    — runs benchmarks, resumes interrupted runs, compares results.

Benchmarks are registered lazily in `kumaru.core.registry.BENCHMARKS`, so
importing this package stays cheap; the dataset library loads only when a
benchmark actually runs.
"""

from kumaru.eval.benchmarks.base import BenchItem, Benchmark, Grade, select_items

__all__ = ["BenchItem", "Benchmark", "Grade", "select_items"]
