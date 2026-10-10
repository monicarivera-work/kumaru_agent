"""The contract every benchmark implements, and the deterministic item sampler.

A benchmark is three decisions, and keeping them separate is what makes results
trustworthy:

1. **Which questions** (`load`) — fixed, public datasets, so numbers are
   comparable with anyone else who ran the same set.
2. **How they are asked** (`messages`) — the exact prompt is part of the
   measurement; change it and you have a different benchmark.
3. **How answers are scored** (`grade`) — deterministic code, never vibes, and
   it records *what it extracted* so every verdict can be audited by hand.

Adding a benchmark is additive: subclass `Benchmark`, then register it in
`kumaru.core.registry.BENCHMARKS` (or ship it in your own package under the
`kumaru.benchmarks` entry-point group). Nothing else changes.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

from kumaru.core.errors import DependencyMissingError, KumaruError
from kumaru.core.types import Message

if TYPE_CHECKING:
    from kumaru.eval.benchmarks.runner import ItemRecord


class BenchItem(BaseModel):
    """One question, in the shape the runner needs."""

    # Stable across runs and machines. Paired comparisons join on it.
    id: str
    # The user message sent to the model.
    prompt: str
    # Gold answer, in whatever form this benchmark's `grade` expects.
    answer: Any = None
    # Category for per-group accuracy and stratified sampling (e.g. a subject).
    group: str | None = None
    system: str | None = None
    # Per-item answer budget in tokens; overrides `Benchmark.answer_tokens`.
    max_tokens: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Grade(BaseModel):
    """A grader's verdict on one answer."""

    correct: bool
    # Partial credit in [0, 1]. None means "same as correct" (1.0 or 0.0).
    score: float | None = None
    # What the grader read as the model's answer. Recorded so a human can check
    # whether a "wrong" answer was really wrong or just badly extracted.
    extracted: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class Benchmark(ABC):
    """Base class for every benchmark."""

    # Registry name and results-directory name. A class attribute that an
    # instance may refine (a golden set loaded from another file renames itself).
    name: str
    description: ClassVar[str]
    # Dataset id and paper, shown by `kumaru bench list` and stored in results.
    source: ClassVar[str]
    # Output budget when thinking is off; thinking adds its own budget on top.
    # Generous on purpose: a cut-off answer is graded as it stands, and usually
    # fails because it never reached its final line, so a tight cap measures
    # your settings, not the model (at 1,024-2,048 tokens Qwen3.6 had 3-16% of
    # its answers cut mid-reasoning). `n_truncated` shows when it binds.
    answer_tokens: ClassVar[int] = 8192
    # True if grading runs model-written code (requires --allow-code-exec).
    needs_code_execution: ClassVar[bool] = False
    # True if `grade` blocks (subprocesses); the runner moves it to a thread.
    blocking_grader: ClassVar[bool] = False

    def preflight(self) -> None:  # noqa: B027 - optional hook; most benchmarks need no checks
        """Raise before any model loads if something this benchmark needs is missing."""

    @abstractmethod
    def load(self) -> list[BenchItem]:
        """Return every item of the evaluation split, in a stable order."""

    def messages(self, item: BenchItem) -> list[Message]:
        """The conversation sent for one item. Override for multi-turn formats."""
        out: list[Message] = []
        if item.system:
            out.append(Message.system(item.system))
        out.append(Message.user(item.prompt))
        return out

    @abstractmethod
    def grade(self, item: BenchItem, output: str) -> Grade:
        """Score one answer. Must be deterministic."""

    def generation_overrides(self) -> dict[str, Any]:
        """Decoding settings this task needs regardless of model (e.g. no
        repetition penalties for code, where repetition is correct)."""
        return {}

    def aggregate(self, records: list[ItemRecord], items: dict[str, BenchItem]) -> dict[str, float]:
        """Extra metrics beyond accuracy (e.g. IFEval's four official scores).

        `items` maps each record's id to its question, for benchmarks whose
        metrics depend on item metadata. Records include failed requests.
        """
        return {}


def load_hf_rows(path: str, name: str | None, split: str) -> list[dict[str, Any]]:
    """Load a Hugging Face dataset split as a list of plain dicts.

    The first call downloads a few MB into the Hugging Face cache; later calls
    are offline. Imported lazily because `datasets` takes seconds to import.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DependencyMissingError("datasets", extra="bench") from exc

    try:
        dataset = load_dataset(path, name, split=split)
    except Exception as exc:
        raise KumaruError(
            f"Could not load dataset '{path}' (split '{split}'): {exc}",
            hint="The first run needs internet access; afterwards it is served "
            "from the Hugging Face cache (HF_HOME, default ~/.cache/huggingface).",
        ) from exc
    return [dict(row) for row in dataset]


def select_items(items: list[BenchItem], limit: int | None, seed: int) -> list[BenchItem]:
    """Deterministic, stratified subsample of `limit` items.

    The same `(items, limit, seed)` always returns the same questions. That is
    what makes two runs *paired* — both models answer identical questions — and
    pairing is where a comparison gets its statistical power: it cancels out
    "this sample happened to be easy".

    When items carry a `group`, each group gets a share proportional to its
    size (largest-remainder rounding), so a 100-question sample of a 14-subject
    benchmark still covers every subject in the right proportion.
    """
    if limit is None or limit >= len(items):
        return list(items)
    if limit <= 0:
        return []

    rng = random.Random(seed)
    groups: dict[str, list[BenchItem]] = defaultdict(list)
    for item in items:
        groups[item.group or ""].append(item)

    if len(groups) == 1:
        chosen = rng.sample(items, limit)
    else:
        keys = sorted(groups)
        quotas = {k: limit * len(groups[k]) / len(items) for k in keys}
        allocation = {k: int(quotas[k]) for k in keys}
        leftover = limit - sum(allocation.values())
        by_remainder = sorted(keys, key=lambda k: quotas[k] - allocation[k], reverse=True)
        for k in by_remainder[:leftover]:
            allocation[k] += 1
        chosen = []
        for k in keys:
            chosen.extend(rng.sample(groups[k], allocation[k]))

    order = {item.id: index for index, item in enumerate(items)}
    return sorted(chosen, key=lambda item: order[item.id])
