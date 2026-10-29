"""Protect question sampling and every grader: the code that decides right vs wrong."""

import json
from collections import Counter
from pathlib import Path

import pytest

from kumaru.core.errors import ConfigError
from kumaru.core.registry import BENCHMARKS
from kumaru.core.types import Role
from kumaru.eval.benchmarks import codeexec, gsm8k
from kumaru.eval.benchmarks.base import BenchItem, select_items
from kumaru.eval.benchmarks.codeexec import Outcome
from kumaru.eval.benchmarks.golden import DEFAULT_PATH, GRADERS, GoldenSet, grade_answer
from kumaru.eval.benchmarks.gsm8k import GSM8K, gold_answer
from kumaru.eval.benchmarks.humaneval import GUARD, HumanEval, assemble_program, run_program
from kumaru.eval.benchmarks.mmlu_pro import MMLUPro, clean_options


def make_items(sizes: dict[str, int]) -> list[BenchItem]:
    """Items interleaved across groups, so "order preserved" is a real check."""
    items = []
    for index in range(max(sizes.values())):
        for group, size in sizes.items():
            if index < size:
                items.append(BenchItem(id=f"{group}{index}", prompt="q", group=group))
    return items


def write_jsonl(path: Path, rows: list) -> Path:
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestSelectItems:
    def test_same_seed_same_questions_different_seed_different_questions(self) -> None:
        items = make_items({"": 200})
        first = [i.id for i in select_items(items, 20, seed=1)]
        assert first == [i.id for i in select_items(items, 20, seed=1)]
        assert first != [i.id for i in select_items(items, 20, seed=2)]

    def test_no_limit_or_a_large_limit_returns_everything_in_order(self) -> None:
        items = make_items({"a": 3, "b": 2})
        assert select_items(items, None, seed=0) == items
        assert select_items(items, 99, seed=0) == items

    def test_non_positive_limit_returns_nothing(self) -> None:
        assert select_items(make_items({"a": 3}), 0, seed=0) == []

    def test_groups_get_proportional_shares(self) -> None:
        chosen = select_items(make_items({"a": 60, "b": 30, "c": 10}), 10, seed=5)
        assert Counter(i.group for i in chosen) == {"a": 6, "b": 3, "c": 1}

    def test_largest_remainder_ties_go_to_the_alphabetically_first_group(self) -> None:
        # Quotas 2.5 / 1.5 / 1.0: one seat left over, a and b tie on remainder.
        chosen = select_items(make_items({"b": 3, "a": 5, "c": 2}), 5, seed=5)
        assert Counter(i.group for i in chosen) == {"a": 3, "b": 1, "c": 1}

    def test_selection_keeps_the_original_order(self) -> None:
        items = make_items({"a": 40, "b": 40})
        position = {item.id: index for index, item in enumerate(items)}
        chosen = select_items(items, 15, seed=9)
        assert len(chosen) == 15
        assert [position[i.id] for i in chosen] == sorted(position[i.id] for i in chosen)


class TestGoldenGraders:
    def test_exact_ignores_case_quotes_whitespace_and_a_final_period(self) -> None:
        assert grade_answer("exact", "Paris", '  "Paris."  ').correct
        assert grade_answer("exact", ["4", "four"], "FOUR").correct
        assert not grade_answer("exact", "Paris", "Paris, France").correct

    def test_contains_requires_every_string_and_reports_the_missing_ones(self) -> None:
        grade = grade_answer("contains", ["alpha", "beta"], "ALPHA and gamma")
        assert not grade.correct
        assert grade.detail["missing"] == ["beta"]

    def test_contains_any_needs_one_match(self) -> None:
        grade = grade_answer("contains_any", ["x-ray", "Gamma"], "alpha and gamma")
        assert grade.correct
        assert grade.detail["found"] == ["Gamma"]

    def test_not_contains_reports_forbidden_strings(self) -> None:
        grade = grade_answer("not_contains", ["as an ai"], "As an AI, I cannot.")
        assert not grade.correct
        assert grade.detail["forbidden"] == ["as an ai"]

    def test_regex_is_case_sensitive_unless_the_pattern_says_otherwise(self) -> None:
        grade = grade_answer("regex", r"^def \w+", "  def foo(): pass")
        assert grade.correct
        assert grade.extracted == "def foo"
        assert not grade_answer("regex", "hello", "Hello").correct
        assert grade_answer("regex", "(?i)hello", "Hello").correct

    def test_regex_with_a_list_needs_any_one_pattern(self) -> None:
        grade = grade_answer("regex", [r"^no\b", r"(?i)^yes\b"], "YES, both are.")
        assert (grade.correct, grade.extracted) == (True, "YES")
        assert not grade_answer("regex", [r"^no\b", r"^yes\b"], "Maybe.").correct

    def test_numeric_uses_the_extracted_final_number_and_tolerance(self) -> None:
        grade = grade_answer("numeric", 26, "12 + 14 = 26\nAnswer: 26")
        assert (grade.correct, grade.extracted, grade.detail["method"]) == (True, "26", "answer_tag")
        assert not grade_answer("numeric", 3.14159, "Answer: 3.14").correct
        assert grade_answer("numeric", 3.14159, "Answer: 3.14", tolerance=0.01).correct
        assert not grade_answer("numeric", 1450000, "Answer: 1,450,001").correct  # exact by default

    def test_choice_compares_letters_case_insensitively(self) -> None:
        grade = grade_answer("choice", " b ", "The answer is (B)")
        assert grade.correct
        assert grade.extracted == "B"

    def test_unknown_grader_raises_config_error(self) -> None:
        with pytest.raises(ConfigError):
            grade_answer("vibes", "x", "x")


class TestGoldenSetLoader:
    def test_rows_become_items_with_ids_groups_and_defaults(self, scratch_dir: Path) -> None:
        path = write_jsonl(
            scratch_dir / "support.jsonl",
            [
                "// a comment line",
                "",
                {"id": "pens", "prompt": "How many?", "grader": "numeric", "answer": 26,
                 "tags": ["math", "easy"], "tolerance": 0.5, "system": "Be terse.", "max_tokens": 64},
                {"prompt": "Say hi", "answer": "hi"},
            ],
        )  # fmt: skip
        bench = GoldenSet(path)
        items = bench.load()
        assert bench.name == "golden-support"
        assert [i.id for i in items] == ["golden-support/pens", "golden-support/4"]
        first, second = items
        assert (first.group, first.max_tokens, first.meta["tolerance"]) == ("math", 64, 0.5)
        assert (second.group, second.meta["grader"], second.meta["tolerance"]) == (None, "contains", 0.0)
        assert [m.role for m in bench.messages(first)] == [Role.SYSTEM, Role.USER]
        assert bench.grade(first, "Answer: 26.3").correct

    @pytest.mark.parametrize(
        ("rows", "message"),
        [
            (["{not json"], "bad.jsonl:1"),
            ([{"prompt": "  ", "answer": 1}], "non-empty 'prompt'"),
            ([["a", "list"]], "non-empty 'prompt'"),
            ([{"prompt": "p", "answer": 1, "grader": "vibes"}], "unknown grader"),
            ([{"prompt": "p"}], "missing 'answer'"),
            ([{"id": "x", "prompt": "p", "answer": 1}, {"id": "x", "prompt": "q", "answer": 2}], "duplicate id"),
            (["// only comments"], "no questions"),
            # A misspelled key must not silently fall back to a default.
            ([{"prompt": "p", "answer": 1, "grade": "numeric"}], "unknown key 'grade'"),
            ([{"prompt": "p", "answer": 1, "tolerance": None}], "'tolerance' must be a number"),
            ([{"prompt": "p", "answer": 1, "tolerance": -0.1}], "'tolerance' must be a number"),
            ([{"prompt": "p", "answer": 1, "max_tokens": 0}], "'max_tokens' must be a whole number"),
            ([{"prompt": "p", "answer": 1, "tags": "math"}], "'tags' must be a list"),
            ([{"prompt": "p", "answer": 1, "system": 7}], "'system' must be text"),
            ([{"prompt": "p", "answer": ["ok", "("], "grader": "regex"}], "invalid regex '\\('"),
            # Each of these would grade every reply the same way, right or wrong.
            ([{"prompt": "p", "answer": ""}], "'answer' must not be empty"),
            ([{"prompt": "p", "answer": ["ok", " "]}], "'answer' must not be empty"),
            ([{"prompt": "p", "answer": [], "grader": "regex"}], "'answer' must not be empty"),
            ([{"prompt": "p", "answer": "twenty", "grader": "numeric"}], "needs a number"),
            ([{"prompt": "p", "answer": "(C)", "grader": "choice"}], "needs one letter A-J"),
        ],
    )
    def test_malformed_files_raise_config_error_naming_the_problem(
        self, scratch_dir: Path, rows: list, message: str
    ) -> None:
        path = write_jsonl(scratch_dir / "bad.jsonl", rows)
        with pytest.raises(ConfigError, match=message):
            GoldenSet(path).load()

    def test_missing_file_raises_config_error(self, scratch_dir: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            GoldenSet(scratch_dir / "absent.jsonl").load()

    def test_a_file_that_is_not_utf8_raises_config_error(self, scratch_dir: Path) -> None:
        path = scratch_dir / "utf16.jsonl"  # what Windows PowerShell 5's `>` writes
        path.write_text('{"prompt": "p", "answer": 1}\n', encoding="utf-16")
        with pytest.raises(ConfigError, match="Cannot read golden set"):
            GoldenSet(path).load()

    def test_a_byte_order_mark_is_accepted(self, scratch_dir: Path) -> None:
        path = scratch_dir / "bom.jsonl"
        path.write_text('{"prompt": "p", "answer": 1}\n', encoding="utf-8-sig")
        assert [i.prompt for i in GoldenSet(path).load()] == ["p"]

    def test_shipped_golden_set_loads_with_unique_ids_and_known_graders(self) -> None:
        bench = GoldenSet()
        items = bench.load()
        assert bench.name == "golden"
        assert bench.path == DEFAULT_PATH
        # Not pinned to a count: the set is meant to grow with every bug you fix.
        assert items and len(items) == len({i.id for i in items})
        assert all(i.meta["grader"] in GRADERS for i in items)


class TestShippedFormatQuestions:
    """A format question must fail a reply that breaks a rule its prompt states."""

    @pytest.mark.parametrize(
        ("question", "reply", "correct"),
        [
            ("banana-caps", "BANANA", True),
            ("banana-caps", " BANANA\n", True),
            ("banana-caps", "BANANA.", False),  # "and nothing else"
            ("banana-caps", "Banana", False),
            ("three-colors", "Red\nGreen\nBlue", True),
            ("three-colors", "- blue\n- RED\n- green", True),  # any order, case or bullet
            ("three-colors", "red\nred\nred", False),  # "the three primary colours"
            ("three-colors", "Red\nGreen\nGreen", False),
            ("three-colors", "Red, Green, Blue", False),  # "one per line"
            ("three-colors", "Red\nGreen\nBlue\nYellow", False),
            ("three-colors", "Red light\nGreen\nBlue", False),  # "with no other text"
            ("json-ada", '{"name": "Ada", "age": 36}', True),
            ("json-ada", '{\n  "age": 36,\n  "name": "Ada"\n}', True),  # either key order
            ("json-ada", '```json\n{"name": "Ada", "age": 36}\n```', True),
            ("json-ada", '{"name":"Ada","age":36,"admin":true}', False),  # "exactly two keys"
            ("json-ada", '{"admin": true, "name": "Ada", "age": 36}', False),
            ("json-ada", '{"name": "Ada", "age": 360}', False),
            ("json-ada", '{"name": "Ada"}', False),
            ("json-ada", 'Here it is: {"name": "Ada", "age": 36}', False),  # "only the JSON"
        ],
    )
    def test_grader_accepts_exactly_what_the_prompt_asks_for(
        self, question: str, reply: str, correct: bool
    ) -> None:
        bench = GoldenSet()
        item = next(i for i in bench.load() if i.id == f"golden/{question}")
        assert bench.grade(item, reply).correct == correct


class TestGsm8kAndMmluPro:
    def test_gsm8k_gold_answer_is_the_number_after_the_marker(self) -> None:
        assert gold_answer("She has 3 + 4 = 7 pens.\n#### 1,234") == "1234"

    def test_gsm8k_grade_accepts_formatted_numbers(self) -> None:
        item = BenchItem(id="gsm8k/0", prompt="p", answer="1234")
        grade = GSM8K().grade(item, "So...\nAnswer: 1,234")
        assert (grade.correct, grade.detail["method"]) == (True, "answer_tag")

    def test_gsm8k_train_asks_the_training_split_under_its_own_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked = []

        def rows(path: str, name: str | None, split: str) -> list[dict]:
            asked.append((path, name, split))
            return [{"question": " Sam has 3 pens and buys 4. ", "answer": "3 + 4 = 7\n#### 7"}]

        monkeypatch.setattr(gsm8k, "load_hf_rows", rows)
        [item] = BENCHMARKS.get("gsm8k-train")().load()

        assert asked == [("openai/gsm8k", "main", "train")]
        # Its own ids, so teacher data can never pass for a test-set run.
        assert (item.id, item.answer) == ("gsm8k_train/0", "7")
        assert item.prompt == gsm8k.PROMPT.format(question="Sam has 3 pens and buys 4.")

    @pytest.mark.parametrize(
        ("options", "index", "expected"),
        [
            (["a", "b", "N/A"], 1, ["a", "b"]),
            (["a", "N/A", "b", "c"], 0, ["a", "b", "c"]),
            # Dropping the padding would move the answer from C to B: keep it.
            (["a", "N/A", "b"], 2, ["a", "N/A", "b"]),
        ],
    )
    def test_clean_options_drops_padding_only_when_the_answer_stays_put(
        self, options: list[str], index: int, expected: list[str]
    ) -> None:
        assert clean_options(options, index) == expected

    def test_mmlu_pro_grade_only_accepts_letters_that_exist(self) -> None:
        item = BenchItem(id="mmlu_pro/1", prompt="p", answer="C", meta={"letters": "ABCD"})
        assert MMLUPro().grade(item, "The answer is (C)").correct
        grade = MMLUPro().grade(item, "The answer is (E)")
        assert (grade.correct, grade.extracted) == (False, None)


PROMPT = 'def add(a, b):\n    """Return a + b."""\n'
TEST = "def check(candidate):\n    assert candidate(2, 3) == 5\n"


class TestHumanEval:
    def test_completion_defining_the_function_follows_the_prompt(self) -> None:
        program = assemble_program(PROMPT, "```python\ndef add(a, b):\n    return a + b\n```", TEST, "add")
        assert program.startswith(GUARD)
        assert PROMPT + "\n\ndef add(a, b):\n    return a + b\n" in program
        assert program.endswith("check(add)\n")

    def test_bare_body_continues_the_prompt(self) -> None:
        program = assemble_program(PROMPT, "    return a + b\n", TEST, "add")
        assert PROMPT + "    return a + b\n" in program

    def test_unindented_body_is_shifted_under_the_signature(self) -> None:
        program = assemble_program(PROMPT, "total = a + b\nreturn total\n", TEST, "add")
        assert PROMPT + "    total = a + b\n    return total\n" in program

    def test_correct_code_passes(self) -> None:
        program = assemble_program(PROMPT, "    return a + b\n", TEST, "add")
        assert run_program(program, timeout_s=30) == ("passed", "")

    def test_wrong_code_fails_with_the_traceback_tail(self) -> None:
        program = assemble_program(PROMPT, "    return a - b\n", TEST, "add")
        status, stderr = run_program(program, timeout_s=30)
        assert status == "failed"
        assert "AssertionError" in stderr

    def test_endless_code_times_out(self) -> None:
        program = assemble_program(PROMPT, "    while True:\n        pass\n", TEST, "add")
        assert run_program(program, timeout_s=1) == ("timeout", "")

    @pytest.mark.parametrize("early_exit", ["import sys\nsys.exit(0)", "import os\nos._exit(0)", "raise SystemExit"])
    def test_a_clean_exit_before_the_tests_is_not_a_pass(self, early_exit: str) -> None:
        # Wrong code that exits with status 0 before check() runs used to pass.
        completion = f"def add(a, b):\n    return a - b\n\n{early_exit}\n"
        program = assemble_program(PROMPT, completion, TEST, "add")
        status, detail = run_program(program, timeout_s=30)
        assert status == "failed"
        assert "before the tests finished" in detail

    def test_in_the_sandbox_it_runs_the_images_python_and_says_what_stopped_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = []

        def killed(program, *, sandbox="none"):
            seen.append((program.steps[0].argv[0], sandbox))
            return Outcome("failed", "run", 137, "", "", 1.0, "killed: out of memory?")

        monkeypatch.setattr(codeexec, "run", killed)
        # Not mistaken for an early exit, the other outcome with a note.
        assert run_program("x = [0] * 10**10\n", timeout_s=30, sandbox="docker") == (
            "failed",
            "killed: out of memory?",
        )
        assert seen == [("python3", "docker")]

    def test_guard_disables_destructive_calls(self) -> None:
        status, stderr = run_program(GUARD + "import os\nos.remove('anything')\n", timeout_s=30)
        assert status == "failed"
        assert "TypeError" in stderr

    def test_secrets_in_the_environment_are_withheld(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KUMARU_TEST_SECRET", "s3cret")
        program = "import os\nassert 'KUMARU_TEST_SECRET' not in os.environ\n"
        assert run_program(program, timeout_s=30) == ("passed", "")

    def test_grade_reports_the_status(self) -> None:
        item = BenchItem(
            id="HumanEval/0", prompt="p", meta={"prompt": PROMPT, "test": TEST, "entry_point": "add"}
        )
        grade = HumanEval().grade(item, "```python\ndef add(a, b):\n    return a + b\n```")
        assert (grade.correct, grade.extracted, grade.detail) == (True, "passed", {"status": "passed"})
