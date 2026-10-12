"""GSM8K — grade-school math word problems.

1,319 test problems, each needing 2-8 steps of arithmetic reasoning, with an
integer answer. It measures whether a model can *chain* simple steps without
dropping one — the same skill that makes multi-step answers trustworthy.
Frontier models score 95%+, so it is a good first check that a local model
reasons rather than pattern-matches.

Protocol: zero-shot chain of thought. The model may reason freely and must end
with "Answer: <number>". Grading compares numbers, not strings, so "18",
"18.00" and "$18" all match a gold answer of 18.

Source: Cobbe et al., 2021, "Training Verifiers to Solve Math Word Problems";
dataset `openai/gsm8k` (config `main`, split `test`).
"""

from __future__ import annotations

from kumaru.eval.benchmarks.base import BenchItem, Benchmark, Grade, load_hf_rows
from kumaru.eval.benchmarks.extract import extract_final_number, numbers_equal

PROMPT = (
    "Solve the following math problem. Think step by step, then write the final "
    'answer on its own last line in the form "Answer: <number>", with no units.\n\n'
    "{question}"
)


def gold_answer(solution: str) -> str:
    """GSM8K solutions end with '#### <answer>'."""
    return solution.split("####")[-1].strip().replace(",", "")


class GSM8K(Benchmark):
    name = "gsm8k"
    description = "Grade-school math word problems: multi-step arithmetic reasoning."
    source = "openai/gsm8k (main/test, 1,319 problems) - Cobbe et al. 2021"

    def load(self) -> list[BenchItem]:
        rows = load_hf_rows("openai/gsm8k", "main", "test")
        return [
            BenchItem(
                id=f"gsm8k/{index}",
                prompt=PROMPT.format(question=row["question"].strip()),
                answer=gold_answer(row["answer"]),
            )
            for index, row in enumerate(rows)
        ]

    def grade(self, item: BenchItem, output: str) -> Grade:
        value, method = extract_final_number(output)
        return Grade(
            correct=numbers_equal(value, item.answer),
            extracted=value,
            detail={"method": method},
        )
