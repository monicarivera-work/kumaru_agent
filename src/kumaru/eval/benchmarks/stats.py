"""Statistics for benchmark results: is a difference real, or noise?

A score on 100 questions is an estimate, not a fact. Re-run with 100 different
questions and it moves by several points purely by chance. These four tools
put error bars on that uncertainty, using only the standard library (numpy
speeds up the bootstrap when installed):

* `wilson_interval` — a 95% confidence interval for one accuracy. Preferred
  over the textbook "p ± 1.96·sqrt(p(1-p)/n)" because that formula breaks near
  0% and 100%, exactly where good models live.
* `mcnemar_exact` — "did model B really beat model A?" when both answered the
  *same* questions. Only the questions where they *disagree* carry information;
  the test asks whether those disagreements split more unevenly than a fair coin.
* `paired_bootstrap_ci` — a confidence interval for the size of the gain
  (B − A), by re-sampling questions with replacement thousands of times.
* `holm_adjust` — when you compare on five benchmarks at once, one of them
  "wins" at p < 0.05 by luck far more often than 5% of the time. Holm's
  correction raises each p-value just enough to keep the overall false-alarm
  rate at 5%.

Rule of thumb for 100-question runs: an accuracy near 70% carries about ±9
points of uncertainty, and a paired comparison reliably detects gains of
roughly 10+ points. 500 questions shrink the interval to about ±4.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence

Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%)."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    `b` = questions only the first system got right, `c` = only the second.
    Under "no real difference" each disagreement is a fair coin flip, so the
    p-value is a binomial tail. Python's integers are exact at any size, so
    this never underflows.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return `(mean(b) - mean(a), low, high)` from a percentile bootstrap.

    Resampling whole questions (not the two systems separately) keeps the
    pairing, so the interval reflects only uncertainty about the *difference*.
    Both paths draw exactly `resamples` resamples; without numpy that is pure
    Python and slow (about 9 s for 10,000 resamples of 12,000 questions).
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0

    diffs = [float(y) - float(x) for x, y in zip(a, b, strict=True)]
    delta = sum(diffs) / n
    alpha = (1.0 - confidence) / 2

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - the bench, dev and torch extras install numpy
        np = None  # type: ignore[assignment]

    if np is not None:
        # Differences take few distinct values (-1, 0, +1 for right/wrong), so
        # drawing *how many* of each value a resample contains (a multinomial)
        # is mathematically identical to drawing n questions — and ~1000x faster.
        counts = Counter(diffs)
        values = np.array(list(counts), dtype=float)
        probs = np.array([counts[v] / n for v in counts], dtype=float)
        rng = np.random.default_rng(seed)
        means = rng.multinomial(n, probs, size=resamples) @ values / n
        low, high = np.quantile(means, [alpha, 1 - alpha])
        return delta, float(low), float(high)

    rng_py = random.Random(seed)
    means = sorted(sum(rng_py.choices(diffs, k=n)) / n for _ in range(resamples))
    low = means[int(alpha * resamples)]
    high = means[min(resamples - 1, int(math.ceil((1 - alpha) * resamples)) - 1)]
    return delta, low, high


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Sort ascending; multiply the k-th smallest (0-based) by (m - k); then make
    the sequence non-decreasing so a larger raw p never gets a smaller adjusted
    one. Uniformly more powerful than plain Bonferroni, with the same guarantee.
    """
    m = len(pvalues)
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(sorted(range(m), key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (m - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted
