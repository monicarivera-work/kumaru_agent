"""Protect distillation data: what is kept, what is dropped and why, what is refused.

Teacher runs are written by hand as benchmark result folders, the evaluation
set is a two-question golden set, and GSM8K's rows are patched in, so no
model, GPU or network is needed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from kumaru.cli import app, main
from kumaru.core.config import load_config
from kumaru.core.errors import ConfigError
from kumaru.core.registry import BENCHMARKS
from kumaru.eval.benchmarks import gsm8k
from kumaru.eval.benchmarks.gsm8k import PROMPT
from kumaru.eval.benchmarks.runner import BenchSummary, ItemRecord
from kumaru.train.config import load_train_config
from kumaru.train.distill import (
    REASONS,
    DistillSettings,
    Manifest,
    NearCopyIndex,
    build_from_runs,
    build_gold,
    gold_rationale,
    split_of,
    wrong_steps,
)

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TRAIN_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "train"

# Two evaluation questions, and training questions written to be easy to confuse with them.
JULIE = (
    "Julie reads 12 pages of a book every day for a week and then reads 20 more "
    "pages on Sunday. How many pages does she read in total?"
)
BAKER = (
    "A baker makes 24 muffins in the morning and twice as many in the afternoon, "
    "then sells all but 10 of them. How many muffins does he sell?"
)
# Same instruction and the same first word as JULIE, but a different problem.
JULIE_SIBLING = (
    "Julie buys 3 notebooks for 2 dollars each and a pen that costs 1 dollar more "
    "than a notebook. How much does she spend altogether?"
)
# JULIE in other words: no 13-word run in common.
PARAPHRASE = (
    "Each day for seven days, Julie gets through a dozen pages, and on Sunday she "
    "adds 20 pages more. What is her page total?"
)
MARBLES = "Tom has 4 marbles and his sister gives him 5 more. How many marbles does Tom have now?"
# Two 14-word questions that open alike: without "John has" they are under 13 words.
APPLES = "John has 5 apples and buys 7 more. How many apples does he have?"
PENS = "John has 3 pens and then loses 1. How many pens does he have?"
QUESTIONS = [
    "Ann has 2 apples and buys 2 more. How many apples does she have?",
    "A bus carries 10 people and 2 get off. How many are left on the bus?",
    "Sam earns 5 dollars a day. How much does he earn in 2 days?",
    "A box holds 6 eggs. How many eggs are in 3 boxes?",
]


def prompt(question: str) -> str:
    """A question as the gsm8k benchmarks ask it."""
    return PROMPT.format(question=question)


def answer(id_: str, question: str, output: str = "2 + 2 = 4\nAnswer: 4", **fields: Any) -> ItemRecord:
    """A graded teacher answer; by default correct, complete and well formatted."""
    values: dict[str, Any] = {
        "correct": True,
        "finish_reason": "stop",
        "completion_tokens": 40,
        "detail": {"method": "answer_tag"},
    }
    return ItemRecord(id=id_, prompt=prompt(question), output=output, **(values | fields))


def write_run(
    folder: Path,
    records: list[ItemRecord],
    *,
    benchmark: str = "gsm8k_train",
    thinking: bool | None = False,
    label: str = "teacher",
) -> Path:
    """One benchmark's result folder, as `kumaru bench run` writes it."""
    folder.mkdir(parents=True)
    summary = BenchSummary(
        benchmark=benchmark,
        label=label,
        model="teacher-model",
        backend="llama_server",
        thinking=thinking,
        n=len(records),
        complete=True,
        item_ids=[record.id for record in records],
    )
    (folder / "summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    lines = "".join(record.model_dump_json() + "\n" for record in records)
    (folder / "items.jsonl").write_text(lines, encoding="utf-8")
    return folder


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def eval_set(scratch_dir: Path) -> Path:
    """A golden set standing in for the test split; its items are golden-eval/<id>."""
    path = scratch_dir / "eval.jsonl"
    rows = [
        {"id": "julie", "prompt": prompt(JULIE), "grader": "numeric", "answer": 104},
        {"id": "baker", "prompt": prompt(BAKER), "grader": "numeric", "answer": 62},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def settings(eval_set: Path) -> DistillSettings:
    return DistillSettings(eval_sets=[str(eval_set)], val_fraction=0.0)


# ---------------------------------------------------------------------------
# Arithmetic check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "step"),
    [
        ("3 x 4 = 13", "3 x 4 = 13"),
        ("192*100=1920", "192*100=1920"),
        ("6/3=.5", "6/3=.5"),
        ("6*2.5 = 18.00", "6*2.5 = 18.00"),
        ("1,500 + 250 = 1,800", "1,500 + 250 = 1,800"),
        ("She pays 4 × $2 = $9 in all.", "4 × 2 = 9"),
        ("\\(3 \\times 4 = 13\\)", "3 × 4 = 13"),
        ("**3 x 4 = 13**", "3 x 4 = 13"),
        ("10 \u2212 3 = 6", "10 - 3 = 6"),
        # The first step goes on ("= 5 + 4"), so only the second is whole.
        ("2 + 3 = 5 + 4 = 10", "5 + 4 = 10"),
        ("1 / 4 = 30%", "1 / 4 = 30"),
    ],
)
def test_wrong_steps_finds_a_wrong_result(text: str, step: str) -> None:
    assert wrong_steps(text) == [step]


@pytest.mark.parametrize(
    "text",
    [
        "3 x 4 = 12",
        "1,500 + 250 = 1,750",
        "\\$1,500 + \\$250 = \\$1,750",
        "10 / 3 = 3.33",  # rounded
        "7 / 2 = 3",  # truncated: whole boxes
        "1 / 4 = 25%",  # read as a fraction
        "2 + 3 x 4 = 14",  # part of a longer expression
        "17 x 16 = 17 x (10 + 6)",
        "13 / 3 = 4.333, and 4.333 x 3 = 13",  # an input rounded to 3 decimals
        "0900 + 30 = 0930",  # clock times
        "9 x 2 = 18 000",  # the number goes on
        "1:30 + 2 = 4",
        "4 x 2 = 8/3",
        "3 x 4 = 13km",
        "5 / 0 = 0",
    ],
)
def test_wrong_steps_passes_right_steps_and_skips_what_it_cannot_judge(text: str) -> None:
    assert wrong_steps(text) == []


# ---------------------------------------------------------------------------
# Near-copy check
# ---------------------------------------------------------------------------


def test_near_copies_are_found_and_shared_instructions_are_not() -> None:
    index = NearCopyIndex({"e/julie": prompt(JULIE), "e/baker": prompt(BAKER)})
    assert index.find(prompt(JULIE)) == "e/julie"
    assert index.find("Warm-up: " + JULIE.upper()) == "e/julie"  # case and context do not hide it
    assert index.find(prompt(BAKER)) == "e/baker"
    # The instruction every prompt starts with is instruction text, and a run may
    # not straddle it: "...with no units. Julie" must not tie JULIE_SIBLING to JULIE.
    assert index.find(prompt(JULIE_SIBLING)) is None
    assert index.find(prompt("")) is None
    assert index.grams(prompt(JULIE)) == index.grams(JULIE)
    # A known limit: a paraphrase shares no 13-word run.
    assert index.find(prompt(PARAPHRASE)) is None


def test_instruction_text_is_also_learned_from_the_training_prompts() -> None:
    # One evaluation question cannot show what is instruction text; the training prompts can.
    eval_prompts = {"e/julie": prompt(JULIE)}
    assert NearCopyIndex(eval_prompts).find(prompt(MARBLES)) == "e/julie"

    train = {"t/1": prompt(JULIE_SIBLING), "t/2": prompt(MARBLES), "t/3": prompt(JULIE)}
    index = NearCopyIndex(eval_prompts, train)
    assert index.find(prompt(MARBLES)) is None
    assert index.find(prompt(JULIE_SIBLING)) is None
    assert index.find(prompt(JULIE)) == "e/julie"


def test_questions_that_open_alike_are_still_found() -> None:
    eval_prompts = {
        "e/apples": prompt(APPLES),
        "e/pens": prompt(PENS),
        "e/julie": prompt(JULIE),
        "e/baker": prompt(BAKER),
    }
    index = NearCopyIndex(eval_prompts)
    # "John has" opens half of these questions, not more: it is question text.
    assert index.find(prompt(APPLES)) == "e/apples"
    assert index.find(prompt(PENS)) == "e/pens"
    assert index.find("Warm-up: " + APPLES) == "e/apples"
    assert index.find(prompt(MARBLES)) is None

    # With only the two, "John has" is in every prompt and is taken for
    # instruction text, leaving 12 words; exact copies are still found.
    pair = NearCopyIndex({"e/apples": prompt(APPLES), "e/pens": prompt(PENS)})
    assert pair.grams(prompt(APPLES)) == []
    assert pair.find(prompt(APPLES).upper()) == "e/apples"
    assert pair.find(prompt(PENS) + " ") == "e/pens"
    assert pair.find(prompt("")) is None


def test_a_question_copied_into_the_training_prompts_is_not_instruction_text() -> None:
    eval_prompts = {"e/julie": prompt(JULIE)}
    copy, variant = prompt(JULIE), prompt("Warm-up: " + JULIE)
    train = {
        "t/copy": copy,
        "t/variant": variant,
        "t/marbles": prompt(MARBLES),
        "t/sibling": prompt(JULIE_SIBLING),
        "t/baker": prompt(BAKER),
    }
    index = NearCopyIndex(eval_prompts, train)
    assert index.find(copy) == "e/julie"
    assert index.find(variant) == "e/julie"
    assert index.find(prompt(MARBLES)) is None
    assert index.find(prompt(JULIE_SIBLING)) is None
    # With only the copy and the variant, the question is in every training
    # prompt and is taken for instruction text; the exact copy is still found.
    assert NearCopyIndex(eval_prompts, {"t/copy": copy, "t/variant": variant}).find(copy) == "e/julie"


def test_each_benchmark_template_is_found_among_its_own_prompts() -> None:
    # Two templates, each in only half of all the prompts but in all of its own.
    other = "Read the problem below carefully and reply with the final number only, nothing else. Problem: {}"
    eval_prompts = {
        "gsm8k/0": prompt(JULIE),
        "gsm8k/1": prompt(BAKER),
        "other/0": other.format(MARBLES),
        "other/1": other.format(JULIE_SIBLING),
    }
    index = NearCopyIndex(eval_prompts)
    assert index.find(prompt(PARAPHRASE)) is None
    assert index.find(other.format(PARAPHRASE)) is None
    assert index.find(other.format(JULIE)) == "gsm8k/0"
    assert index.find(prompt(MARBLES)) == "other/0"


# ---------------------------------------------------------------------------
# Split and gold answers
# ---------------------------------------------------------------------------


def test_the_split_holds_out_questions_by_a_stable_hash() -> None:
    ids = [f"gsm8k_train/{i}" for i in range(4000)]
    sides = [split_of(i, 42, 0.05) for i in ids]
    assert sides == [split_of(i, 42, 0.05) for i in ids]
    assert 0.04 < sides.count("val") / len(ids) < 0.06
    assert [split_of(i, 7, 0.05) for i in ids] != sides  # the seed picks which ones
    assert {split_of(i, 42, 0.0) for i in ids} == {"train"}


def test_gold_rationale_drops_calculator_notes_and_ends_like_the_teacher() -> None:
    solution = (
        "She sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "She sold 1,000+24 = <<1000+24=1024>>1,024 in all.\n#### 1,024"
    )
    assert gold_rationale(solution) == (
        "She sold 48/2 = 24 clips in May.\nShe sold 1,000+24 = 1,024 in all.\nAnswer: 1024"
    )
    assert gold_rationale("no final line") is None
    assert gold_rationale("#### 5") is None


# ---------------------------------------------------------------------------
# Building from teacher runs
# ---------------------------------------------------------------------------


def test_every_check_drops_what_it_should_and_the_counts_add_up(
    scratch_dir: Path, settings: DistillSettings
) -> None:
    q = [f"Question {i}: a train leaves at {i} and travels {i + 1} miles. How far is it?" for i in range(11)]
    first = write_run(scratch_dir / "a", [
        answer("gsm8k_train/0", q[0]),
        answer("gsm8k_train/1", q[1], error="HTTP 500", correct=False),
        answer("gsm8k_train/2", q[2], correct=False),
        answer("gsm8k_train/3", q[3], finish_reason="length"),
        answer("gsm8k_train/4", q[4], detail={"method": "last_number"}),
        answer("gsm8k_train/5", q[5], output="   "),
        answer("gsm8k_train/6", q[6], completion_tokens=2000),
        answer("gsm8k_train/7", JULIE),
        answer("gsm8k_train/8", q[8], output=f"The problem: {BAKER}\nAnswer: 62"),
        answer("gsm8k_train/9", q[9], output="3 x 4 = 13\nAnswer: 13"),
        answer("gsm8k_train/10", q[10], output="5 + 5 = 10\nAnswer: 10"),
    ])  # fmt: skip
    second = write_run(scratch_dir / "b", [
        answer("gsm8k_train/0", q[0], output="2 + 2 =  4\n\nAnswer: 4"),  # same, bar spacing
        answer("gsm8k_train/10", q[10], output="10 / 2 = 5, 5 x 2 = 10\nAnswer: 10"),
    ], label="teacher-seed2")  # fmt: skip
    out = scratch_dir / "data"
    manifest = build_from_runs([first, second], out, settings.model_copy(update={"per_question": 1}))

    assert manifest.dropped == {reason: 1 for reason in REASONS if reason != "not_matched"} | {
        "contaminated": 2
    }
    assert list(manifest.dropped) == [reason for reason in REASONS if reason in manifest.dropped]
    assert manifest.read == 13 == manifest.kept + sum(manifest.dropped.values())
    assert manifest.examples["contaminated"] == [
        {"id": "gsm8k_train/7", "matches": "golden-eval/julie"},  # the question
        {"id": "gsm8k_train/8", "matches": "golden-eval/baker"},  # quoted in the answer
    ]
    assert manifest.examples["arithmetic"] == [{"id": "gsm8k_train/9", "step": "3 x 4 = 13"}]
    assert [source["label"] for source in manifest.sources] == ["teacher", "teacher-seed2"]
    assert (manifest.kept, manifest.train_rows, manifest.val_rows, manifest.questions) == (2, 2, 0, 2)
    assert manifest.mean_answer_tokens == 40.0

    assert read_rows(out / "train.jsonl") == [
        {
            "messages": [
                {"role": "user", "content": prompt(q[0])},
                {"role": "assistant", "content": "2 + 2 = 4\nAnswer: 4"},
            ],
            "id": "gsm8k_train/0",
            "source": "teacher",
        },
        {
            "messages": [
                {"role": "user", "content": prompt(q[10])},
                {"role": "assistant", "content": "5 + 5 = 10\nAnswer: 10"},
            ],
            "id": "gsm8k_train/10",
            "source": "teacher",
        },
    ]
    assert read_rows(out / "val.jsonl") == []

    unchecked = settings.model_copy(update={"per_question": 1, "check_arithmetic": False})
    assert build_from_runs([first, second], scratch_dir / "unchecked", unchecked).kept == 3


def test_the_manifest_proves_which_bytes_were_written(scratch_dir: Path, settings: DistillSettings) -> None:
    run = write_run(scratch_dir / "run", [answer(f"gsm8k_train/{i}", q) for i, q in enumerate(QUESTIONS)])
    out = scratch_dir / "data"
    manifest = build_from_runs([run], out, settings.model_copy(update={"val_fraction": 0.5}))

    assert set(manifest.files) == {"train.jsonl", "val.jsonl"}
    for name, digest in manifest.files.items():
        data = (out / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        assert b"\r\n" not in data  # the same bytes, and hash, on every OS
    saved = Manifest.model_validate_json((out / "manifest.json").read_text(encoding="utf-8"))
    assert saved.model_dump() == manifest.model_dump()
    assert saved.settings.eval_sets == settings.eval_sets


def test_every_answer_to_a_question_lands_on_the_same_side(scratch_dir: Path) -> None:
    questions = [f"Question {i}: what is {i} plus 1?" for i in range(80)]
    runs = [
        write_run(scratch_dir / "a", [
            answer(f"gsm8k_train/{i}", q, f"{i} + 1 = {i + 1}\nAnswer: {i + 1}") for i, q in enumerate(questions)
        ]),
        write_run(scratch_dir / "b", [
            answer(f"gsm8k_train/{i}", q, f"Adding one to {i} gives {i + 1}.\nAnswer: {i + 1}") for i, q in enumerate(questions)
        ], label="teacher-seed2"),
    ]  # fmt: skip
    out = scratch_dir / "data"
    manifest = build_from_runs(runs, out, DistillSettings(val_fraction=0.3, decontaminate=False))

    sides: dict[str, set[str]] = {}
    for side in ("train", "val"):
        for row in read_rows(out / f"{side}.jsonl"):
            sides.setdefault(row["id"], set()).add(side)
    assert all(len(found) == 1 for found in sides.values())
    val_ids = {i for i, found in sides.items() if found == {"val"}}
    assert 0 < len(val_ids) == manifest.val_questions < 80
    assert manifest.val_rows == 2 * len(val_ids)
    assert (manifest.kept, manifest.questions) == (160, 80)
    assert val_ids == {f"gsm8k_train/{i}" for i in range(80) if split_of(f"gsm8k_train/{i}", 42, 0.3) == "val"}


# ---------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("benchmark", ["golden-eval", "gsm8k", "golden", "humaneval", "not-registered"])
def test_answers_to_evaluation_sets_are_refused_as_training_data(
    scratch_dir: Path, settings: DistillSettings, benchmark: str
) -> None:
    run = write_run(scratch_dir / "run", [answer("x/0", QUESTIONS[0])], benchmark=benchmark)
    out = scratch_dir / "data"
    with pytest.raises(ConfigError, match=f"answers to {benchmark}, an evaluation set"):
        build_from_runs([run], out, settings)
    assert not out.exists()
    # Allowed only when asked for explicitly, for a smoke test.
    smoke = settings.model_copy(update={"decontaminate": False})
    assert build_from_runs([run], out, smoke).kept == 1


def test_a_run_folder_mixing_in_an_evaluation_set_is_refused_whole(
    scratch_dir: Path, settings: DistillSettings
) -> None:
    write_run(scratch_dir / "run" / "gsm8k_train", [answer("gsm8k_train/0", QUESTIONS[0])])
    write_run(scratch_dir / "run" / "ifeval", [answer("ifeval/0", QUESTIONS[1])], benchmark="ifeval")
    out = scratch_dir / "data"
    with pytest.raises(ConfigError, match="ifeval, an evaluation set"):
        build_from_runs([scratch_dir / "run"], out, settings)
    assert not out.exists()
    # One benchmark's own folder is accepted.
    assert build_from_runs([scratch_dir / "run" / "gsm8k_train"], out, settings).kept == 1


def test_thinking_runs_are_refused_unless_allowed(scratch_dir: Path, settings: DistillSettings) -> None:
    flagged = write_run(scratch_dir / "flagged", [answer("gsm8k_train/0", QUESTIONS[0])], thinking=True)
    hidden = write_run(
        scratch_dir / "hidden",
        [answer("gsm8k_train/0", QUESTIONS[0], reasoning="Let me think.")],
        thinking=None,  # the summary does not say; the records do
    )
    for run in (flagged, hidden):
        with pytest.raises(ConfigError, match="thinking run"):
            build_from_runs([run], scratch_dir / "data", settings)
    allowed = settings.model_copy(update={"allow_thinking": True})
    manifest = build_from_runs([flagged, hidden], scratch_dir / "data", allowed)
    assert (manifest.kept, manifest.dropped) == (1, {"duplicate": 1})


def test_a_dataset_is_replaced_only_with_force(scratch_dir: Path, settings: DistillSettings) -> None:
    run = write_run(scratch_dir / "run", [answer("gsm8k_train/0", QUESTIONS[0])])
    out = scratch_dir / "data"
    build_from_runs([run], out, settings)
    with pytest.raises(ConfigError, match="already holds a dataset"):
        build_from_runs([run], out, settings)
    build_from_runs([run], out, settings.model_copy(update={"seed": 7}), force=True)
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["settings"]["seed"] == 7

    (scratch_dir / "a-file").write_text("not a folder", encoding="utf-8")
    with pytest.raises(ConfigError, match="is a file"):
        build_from_runs([run], scratch_dir / "a-file", settings)


def test_no_survivors_is_an_error_and_writes_nothing(scratch_dir: Path, settings: DistillSettings) -> None:
    run = write_run(scratch_dir / "run", [answer("gsm8k_train/0", QUESTIONS[0], correct=False)])
    with pytest.raises(ConfigError, match=r"No answer survived the checks \(wrong 1\)"):
        build_from_runs([run], scratch_dir / "data", settings)
    assert not (scratch_dir / "data").exists()
    with pytest.raises(ConfigError, match="No benchmark results"):
        build_from_runs([scratch_dir / "data"], scratch_dir / "data2", settings)


# ---------------------------------------------------------------------------
# The gold arm
# ---------------------------------------------------------------------------

GSM8K_TRAIN = [
    {"question": QUESTIONS[0], "answer": "Ann has 2+2 = <<2+2=4>>4 apples.\n#### 4"},
    {"question": QUESTIONS[1], "answer": "10-2 = <<10-2=8>>8 are left.\n#### 8"},
    {"question": JULIE, "answer": "12*7 = <<12*7=84>>84 and 84+20 = <<84+20=104>>104\n#### 104"},
    {"question": QUESTIONS[3], "answer": "3 x 6 = <<3*6=19>>19 eggs.\n#### 19"},  # GSM8K has a few such errors
    {"question": QUESTIONS[2], "answer": "malformed: no final line"},
]
GSM8K_TEST = [
    {"question": JULIE, "answer": "#### 104"},
    {"question": BAKER, "answer": "#### 62"},
]


@pytest.fixture
def gsm8k_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def load_hf_rows(path: str, config: str, split: str) -> list[dict[str, str]]:
        assert (path, config) == ("openai/gsm8k", "main")
        return {"train": GSM8K_TRAIN, "test": GSM8K_TEST}[split]

    monkeypatch.setattr(gsm8k, "load_hf_rows", load_hf_rows)


@pytest.mark.usefixtures("gsm8k_rows")
def test_the_gold_arm_asks_the_benchmark_prompts_and_runs_the_same_checks(scratch_dir: Path) -> None:
    out = scratch_dir / "gold"
    # The default evaluation set is GSM8K's test split (patched in above).
    manifest = build_gold(out, DistillSettings(val_fraction=0.0))

    assert manifest.kind == "gold"
    assert list(manifest.dropped.items()) == [("empty", 1), ("contaminated", 1), ("arithmetic", 1)]
    assert manifest.examples["contaminated"] == [{"id": "gsm8k_train/2", "matches": "gsm8k/0"}]
    assert manifest.read == 5
    # Human solutions have no token counts, so no length limit applies.
    assert (manifest.settings.max_tokens, manifest.mean_answer_tokens) == (None, None)
    assert manifest.sources == [
        {"dataset": "openai/gsm8k", "config": "main", "split": "train", "n": 5, "match": None}
    ]
    rows = read_rows(out / "train.jsonl")
    assert [row["id"] for row in rows] == ["gsm8k_train/0", "gsm8k_train/1"]
    assert rows[0] == {
        "messages": [
            {"role": "user", "content": prompt(QUESTIONS[0])},
            {"role": "assistant", "content": "Ann has 2+2 = 4 apples.\nAnswer: 4"},
        ],
        "id": "gsm8k_train/0",
        "source": "gold:openai/gsm8k",
    }


@pytest.mark.usefixtures("gsm8k_rows")
def test_the_gold_arm_can_match_a_teacher_dataset_question_for_question(scratch_dir: Path) -> None:
    teacher = scratch_dir / "teacher"
    teacher.mkdir()
    (teacher / "train.jsonl").write_text(json.dumps({"id": "gsm8k_train/1"}) + "\n", encoding="utf-8")
    (teacher / "val.jsonl").write_text(json.dumps({"id": "gsm8k_train/0"}) + "\n", encoding="utf-8")

    manifest = build_gold(scratch_dir / "gold", DistillSettings(decontaminate=False), match=teacher)
    assert manifest.dropped == {"empty": 1, "not_matched": 2}
    assert [row["id"] for row in read_rows(scratch_dir / "gold" / "train.jsonl")] == ["gsm8k_train/1"]
    assert [row["id"] for row in read_rows(scratch_dir / "gold" / "val.jsonl")] == ["gsm8k_train/0"]
    assert manifest.sources[0]["match"] == str(teacher)

    with pytest.raises(ConfigError, match="not found"):
        build_gold(scratch_dir / "gold2", match=scratch_dir / "nowhere")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_the_cli_builds_a_dataset_and_says_what_to_run_next(scratch_dir: Path, eval_set: Path) -> None:
    run = write_run(scratch_dir / "run", [
        answer("gsm8k_train/0", QUESTIONS[0]),
        answer("gsm8k_train/1", QUESTIONS[1], correct=False),
    ])  # fmt: skip
    runner = CliRunner()
    args = ["distill", "build", str(run), "--eval-set", str(eval_set)]
    result = runner.invoke(app, [*args, "-o", str(scratch_dir / "data")], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    text = " ".join(ANSI.sub("", result.output).split())
    assert "kept 1 of 2 answers (50.0%): 1 train rows, 0 val rows" in text
    assert re.search(r"\bwrong\W+1\W+graded wrong", text)
    assert "configs/train/distill-gsm8k.yaml" in text
    assert "not decontaminated" not in text
    # No question was held out, and the trainer refuses an empty eval file.
    assert "set data.eval_file: null and data.eval_split: 0" in text
    assert "val.jsonl is empty" in text
    assert "data.eval_file at" not in text

    both = write_run(scratch_dir / "both", [
        answer("gsm8k_train/0", QUESTIONS[0]),
        answer("gsm8k_train/1", QUESTIONS[1]),
    ])  # fmt: skip
    held = ["distill", "build", str(both), "--eval-set", str(eval_set), "--val-fraction", "0.5"]
    result = runner.invoke(app, [*held, "-o", str(scratch_dir / "held")], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    text = " ".join(ANSI.sub("", result.output).split())
    assert "1 train rows, 1 val rows" in text
    assert "data.eval_file at" in text
    assert str(scratch_dir / "held" / "val.jsonl") in "".join(text.split())  # rich may fold a long path
    assert "data.eval_split: 0" not in text

    smoke = [*args, "-o", str(scratch_dir / "smoke"), "--no-decontaminate"]
    result = runner.invoke(app, smoke, env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "not decontaminated" in ANSI.sub("", result.output)


def test_the_cli_refuses_evaluation_answers_with_a_hint(
    scratch_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = write_run(scratch_dir / "run", [answer("gsm8k/0", QUESTIONS[0])], benchmark="gsm8k")
    monkeypatch.setattr(sys, "argv", ["kumaru", "distill", "build", str(run), "-o", str(scratch_dir / "data")])
    monkeypatch.setenv("COLUMNS", "200")
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
    out = " ".join(ANSI.sub("", capsys.readouterr().out).split())
    assert "holds answers to gsm8k, an evaluation set" in out
    assert "hint: Run the teacher on a training split (e.g. gsm8k_train)" in out


# ---------------------------------------------------------------------------
# Configs and registry
# ---------------------------------------------------------------------------


def _flat(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat |= _flat(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def test_the_arms_and_the_pilot_differ_only_where_they_must() -> None:
    teacher, gold, pilot = (
        _flat(load_train_config(TRAIN_CONFIGS / name).model_dump())
        for name in ("distill-gsm8k.yaml", "distill-gsm8k-gold.yaml", "distill-gsm8k-0.5b.yaml")
    )
    # A comparison is fair only if the data is the one thing that changed.
    assert {k for k in teacher if teacher[k] != gold[k]} == {
        "output_dir", "run_name", "data.train_file", "data.eval_file",
    }  # fmt: skip
    # The pilot may also split the batch differently, to fit in memory, but not
    # change it: the loss is averaged over the answer tokens of the whole batch.
    split = {"optim.per_device_batch_size", "optim.gradient_accumulation_steps"}
    assert {k for k in teacher if teacher[k] != pilot[k]} - split == {
        "base_model", "output_dir", "run_name",
    }  # fmt: skip
    teacher_batch, pilot_batch = (
        arm["optim.per_device_batch_size"] * arm["optim.gradient_accumulation_steps"]
        for arm in (teacher, pilot)
    )
    assert pilot_batch == teacher_batch
    # The adapter kept is the checkpoint with the lowest eval loss, and the
    # trainer can only return to an evaluation it also saved.
    assert teacher["save_steps"] == teacher["eval_steps"]


def test_the_student_profile_scores_the_model_the_configs_train() -> None:
    student = load_config("student", use_env=False)
    teacher_arm = load_train_config(TRAIN_CONFIGS / "distill-gsm8k.yaml")
    assert student.backend.model == teacher_arm.base_model
    assert student.backend.max_batch_size > 1
    assert student.generation.temperature == 0.0  # greedy: the only mode that batches
    longest = DistillSettings().max_tokens or 0
    assert student.generation.max_tokens >= longest
    # GSM8K's longest training prompt is 276 tokens with the chat template.
    assert teacher_arm.data.max_seq_length >= 276 + longest


def test_only_training_splits_declare_themselves_training_data() -> None:
    declared = {name for name in BENCHMARKS.names() if BENCHMARKS.get(name).training_data}
    assert declared == {"gsm8k_train"}
