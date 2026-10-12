"""Protect the statistics behind every "B beat A" claim, against hand-checked values."""

import random
import sys
from statistics import NormalDist
from typing import Any

import pytest

from kumaru.eval.benchmarks.stats import (
    Z95,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap_ci,
    wilson_interval,
)


class TestWilsonInterval:
    @pytest.mark.parametrize(
        ("successes", "n", "low", "high"),
        [
            (50, 100, 0.403832, 0.596168),
            (96, 100, 0.901629, 0.984337),
            (0, 10, 0.0, 0.277533),
            (10, 10, 0.722467, 1.0),
        ],
    )
    def test_known_values(self, successes: int, n: int, low: float, high: float) -> None:
        got = wilson_interval(successes, n)
        assert got == pytest.approx((low, high), abs=1e-6)

    @pytest.mark.parametrize(("successes", "n"), [(3, 17), (50, 100), (96, 100), (499, 500)])
    def test_bounds_solve_the_wilson_equation(self, successes: int, n: int) -> None:
        # Definition: the bounds are the p where (p_hat - p)^2 = z^2 p (1 - p) / n.
        p_hat = successes / n
        for p in wilson_interval(successes, n):
            assert (p_hat - p) ** 2 == pytest.approx(Z95**2 * p * (1 - p) / n, rel=1e-9)

    def test_no_data_means_total_uncertainty(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_z_is_the_two_sided_95_percent_normal_quantile(self) -> None:
        assert NormalDist().inv_cdf(0.975) == pytest.approx(Z95, abs=1e-12)

    def test_interval_always_contains_the_estimate_and_stays_in_bounds(self) -> None:
        for successes in range(21):
            low, high = wilson_interval(successes, 20)
            assert 0.0 <= low <= successes / 20 <= high <= 1.0


class TestMcNemarExact:
    @pytest.mark.parametrize(
        ("b", "c", "expected"),
        [(0, 6, 0.03125), (1, 9, 0.021484375), (3, 3, 1.0), (0, 0, 1.0)],
    )
    def test_known_values(self, b: int, c: int, expected: float) -> None:
        assert mcnemar_exact(b, c) == pytest.approx(expected, abs=1e-12)

    def test_is_symmetric_in_its_arguments(self) -> None:
        assert mcnemar_exact(2, 7) == mcnemar_exact(7, 2)

    def test_huge_counts_neither_raise_nor_leave_the_unit_interval(self) -> None:
        assert mcnemar_exact(1000, 1000) == 1.0
        assert 0.0 <= mcnemar_exact(0, 2000) < 1e-300


class TestPairedBootstrap:
    def test_unequal_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            paired_bootstrap_ci([1.0], [1.0, 0.0])

    def test_empty_samples_return_zeros(self) -> None:
        assert paired_bootstrap_ci([], []) == (0.0, 0.0, 0.0)

    def test_identical_runs_have_a_zero_width_interval_at_zero(self) -> None:
        a = [1.0, 0.0, 1.0, 1.0]
        assert paired_bootstrap_ci(a, list(a), resamples=500) == (0.0, 0.0, 0.0)

    def test_interval_brackets_a_real_gain(self) -> None:
        a = [0.0] * 50 + [1.0] * 50
        b = [1.0] * 100
        delta, low, high = paired_bootstrap_ci(a, b, resamples=4000, seed=1)
        assert delta == 0.5
        assert 0.35 < low < 0.5 < high < 0.65

    def test_same_seed_gives_the_same_interval(self) -> None:
        a = [0.0, 1.0] * 30
        b = [1.0, 1.0, 0.0] * 20
        assert paired_bootstrap_ci(a, b, seed=7, resamples=2000) == paired_bootstrap_ci(
            a, b, seed=7, resamples=2000
        )

    def test_pure_python_fallback_agrees_with_numpy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("numpy")
        a = [0.0] * 50 + [1.0] * 50
        b = [1.0] * 70 + [0.0] * 30
        fast = paired_bootstrap_ci(a, b, resamples=4000, seed=3)
        monkeypatch.setitem(sys.modules, "numpy", None)  # makes `import numpy` fail
        slow = paired_bootstrap_ci(a, b, resamples=4000, seed=3)
        assert slow[0] == fast[0]
        assert slow[1] == pytest.approx(fast[1], abs=0.04)
        assert slow[2] == pytest.approx(fast[2], abs=0.04)

    def test_pure_python_fallback_draws_every_requested_resample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The fallback used to cap any request at 2,000 resamples, silently.
        draws = 0
        choices = random.Random.choices

        def counting_choices(rng: random.Random, *args: Any, **kwargs: Any) -> list[Any]:
            nonlocal draws
            draws += 1
            return choices(rng, *args, **kwargs)

        monkeypatch.setitem(sys.modules, "numpy", None)  # makes `import numpy` fail
        monkeypatch.setattr(random.Random, "choices", counting_choices)
        a = [0.0, 1.0] * 10
        b = [1.0, 1.0, 0.0, 1.0] * 5
        first = paired_bootstrap_ci(a, b, resamples=5_000, seed=3)
        assert draws == 5_000
        assert paired_bootstrap_ci(a, b, resamples=5_000, seed=3) == first


class TestHolmAdjust:
    def test_known_values_keep_input_order(self) -> None:
        assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

    def test_empty_and_single(self) -> None:
        assert holm_adjust([]) == []
        assert holm_adjust([0.02]) == [0.02]

    def test_adjusted_values_are_capped_at_one_and_never_below_raw(self) -> None:
        raw = [0.5, 0.6, 0.001, 0.2]
        adjusted = holm_adjust(raw)
        assert adjusted[0] == adjusted[1] == 1.0
        assert all(p_adj >= p for p, p_adj in zip(raw, adjusted, strict=True))
