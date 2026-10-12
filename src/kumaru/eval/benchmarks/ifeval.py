"""IFEval — can the model follow precise, checkable instructions?

541 prompts, each carrying one to three verifiable instructions ("write at
least 300 words", "no commas", "end with this exact phrase"). It measures the
property that makes an assistant feel *obedient* rather than merely smart, and
it catches regressions that fine-tuning or a new system prompt can introduce.

Four official numbers, all reported in `metrics`:

* prompt-level strict — every instruction in the prompt followed, exact text
  (this is the headline `accuracy`);
* instruction-level strict — share of individual instructions followed;
* the same two under the *loose* reading (see `ifeval_checks.loose_variants`).

Source: Zhou et al., 2023, "Instruction-Following Evaluation for Large Language
Models"; dataset `google/IFEval`, split `train` (the only split; it is the
evaluation set). Needs `pip install -e ".[bench]"` (langdetect for language
checks, NLTK for counting); the first run downloads NLTK's Punkt model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from kumaru.core.errors import KumaruError
from kumaru.eval.benchmarks.base import BenchItem, Benchmark, Grade, load_hf_rows
from kumaru.eval.benchmarks.ifeval_checks import (
    CHECKS,
    check_instructions,
    count_sentences,
    detect_language,
)

if TYPE_CHECKING:
    from kumaru.eval.benchmarks.runner import ItemRecord


class IFEval(Benchmark):
    name = "ifeval"
    description = "Verifiable instruction following: 25 instruction types, 541 prompts."
    source = "google/IFEval (541 prompts) - Zhou et al. 2023"

    def preflight(self) -> None:
        # Fail (or download) before any model time is spent, not at grading.
        detect_language("This sentence is plainly English.")
        count_sentences("Dr. Smith arrived. He sat down.")

    def load(self) -> list[BenchItem]:
        items = []
        for row in load_hf_rows("google/IFEval", None, "train"):
            instruction_ids = list(row["instruction_id_list"])
            unknown = [i for i in instruction_ids if i not in CHECKS]
            if unknown:
                raise KumaruError(
                    f"IFEval prompt {row['key']} uses unknown instructions: {unknown}",
                    hint="The dataset changed upstream; add checkers in ifeval_checks.py.",
                )
            # Arrow stores every possible argument on every row, mostly as None.
            kwargs = [
                {key: value for key, value in (entry or {}).items() if value is not None}
                for entry in row["kwargs"]
            ]
            items.append(
                BenchItem(
                    id=f"ifeval/{row['key']}",
                    prompt=row["prompt"],
                    meta={"instructions": instruction_ids, "kwargs": kwargs},
                )
            )
        return items

    def grade(self, item: BenchItem, output: str) -> Grade:
        strict, loose = check_instructions(item.meta["instructions"], item.meta["kwargs"], output)
        return Grade(
            correct=all(strict),
            score=sum(strict) / len(strict),
            detail={"strict": strict, "loose": loose},
        )

    def aggregate(self, records: list[ItemRecord], items: dict[str, BenchItem]) -> dict[str, float]:
        prompts = strict_prompts = loose_prompts = 0
        instructions = strict_followed = loose_followed = 0
        by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for record in records:
            ids = items[record.id].meta["instructions"]
            # A failed request answered nothing, so it followed nothing.
            strict = record.detail.get("strict") or [False] * len(ids)
            loose = record.detail.get("loose") or [False] * len(ids)
            prompts += 1
            strict_prompts += all(strict)
            loose_prompts += all(loose)
            instructions += len(ids)
            strict_followed += sum(strict)
            loose_followed += sum(loose)
            for instruction_id, followed in zip(ids, strict, strict=True):
                tally = by_kind[instruction_id.split(":")[0]]
                tally[0] += followed
                tally[1] += 1

        metrics: dict[str, Any] = {
            "prompt_level_strict": strict_prompts / max(1, prompts),
            "instruction_level_strict": strict_followed / max(1, instructions),
            "prompt_level_loose": loose_prompts / max(1, prompts),
            "instruction_level_loose": loose_followed / max(1, instructions),
        }
        for kind, (followed, total) in sorted(by_kind.items()):
            metrics[f"strict/{kind}"] = followed / total
        return metrics
