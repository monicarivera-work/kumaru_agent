"""MMLU-Pro — hard, 10-option multiple choice across 14 subjects.

12,032 questions spanning math, physics, chemistry, law, engineering, health,
psychology, economics, business, biology, philosophy, computer science,
history and "other". Compared with the original MMLU it has ten options instead
of four (random guessing scores 10%, not 25%), removes trivial and noisy
questions, and adds reasoning-heavy ones, so it separates strong models that
plain MMLU no longer can.

Protocol: zero-shot chain of thought, ending with "The answer is (X)", graded
with the reference script's extraction order. One deliberate difference: when
no letter can be extracted, the reference script scores a *random guess*; we
score it wrong, so results are deterministic (a slightly conservative bias,
bounded by the extraction-failure rate shown in every summary).

Sampling is stratified by subject (see `select_items`), so small runs keep the
benchmark's subject mix.

Source: Wang et al., 2024, "MMLU-Pro: A More Robust and Challenging Multi-Task
Language Understanding Benchmark"; dataset `TIGER-Lab/MMLU-Pro`, split `test`.
"""

from __future__ import annotations

from kumaru.eval.benchmarks.base import BenchItem, Benchmark, Grade, load_hf_rows
from kumaru.eval.benchmarks.extract import extract_choice

LETTERS = "ABCDEFGHIJ"

PROMPT = (
    "The following is a multiple choice question about {subject}. Think step by step, "
    'then finish your answer with "The answer is (X)" where X is the letter of the '
    "correct option.\n\nQuestion: {question}\nOptions:\n{options}"
)


def clean_options(options: list[str], answer_index: int) -> list[str]:
    """Drop "N/A" padding (as the reference script does) unless it would shift the answer."""
    kept = [option for option in options if option != "N/A"]
    if answer_index < len(kept) and kept[answer_index] == options[answer_index]:
        return kept
    return options


class MMLUPro(Benchmark):
    name = "mmlu_pro"
    description = "Hard 10-option multiple choice, 14 subjects: knowledge plus reasoning."
    source = "TIGER-Lab/MMLU-Pro (test, 12,032 questions) - Wang et al. 2024"

    def load(self) -> list[BenchItem]:
        items = []
        for row in load_hf_rows("TIGER-Lab/MMLU-Pro", None, "test"):
            options = clean_options(list(row["options"]), int(row["answer_index"]))
            letters = LETTERS[: len(options)]
            # Some options carry stray trailing newlines that would split the list.
            listing = "\n".join(
                f"{letter}. {text.strip()}" for letter, text in zip(letters, options, strict=True)
            )
            subject = str(row["category"])
            items.append(
                BenchItem(
                    id=f"mmlu_pro/{row['question_id']}",
                    prompt=PROMPT.format(subject=subject, question=row["question"].strip(), options=listing),
                    answer=LETTERS[int(row["answer_index"])],
                    group=subject,
                    meta={"letters": letters},
                )
            )
        return items

    def grade(self, item: BenchItem, output: str) -> Grade:
        letter, method = extract_choice(output, item.meta.get("letters", LETTERS))
        return Grade(correct=letter == item.answer, extracted=letter, detail={"method": method})
