"""Protect answer extraction: a misread answer silently corrupts every benchmark score."""

import random
import re
import time

import pytest

from kumaru.eval.benchmarks.extract import (
    defines_python,
    extract_choice,
    extract_code,
    extract_final_number,
    fences,
    last_boxed,
    numbers_equal,
    open_fence,
    parse_number,
)


class TestExtractFinalNumber:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Work...\nAnswer: 42", ("42", "answer_tag")),
            ("**Answer:** $1,234", ("1,234", "answer_tag")),
            ("Final answer: -3.5", ("-3.5", "answer_tag")),
            ("Answer:\n\n12", ("12", "answer_tag")),
            ("So \\boxed{18} it is.", ("18", "boxed")),
            ("\\boxed{\\text{7}}", ("7", "boxed")),
            ("So the answer is 72 dollars.", ("72", "answer_is")),
            ("We get 3, then 4.", ("4", "last_number")),
        ],
    )
    def test_each_format_is_found_and_labelled(self, text: str, expected: tuple) -> None:
        assert extract_final_number(text) == expected

    def test_last_answer_tag_wins_over_earlier_drafts(self) -> None:
        assert extract_final_number("Answer: 5\nWait, recheck.\nAnswer: 7") == ("7", "answer_tag")

    def test_answer_tag_beats_boxed_and_boxed_beats_trailing_numbers(self) -> None:
        assert extract_final_number("\\boxed{5}\nAnswer: 6") == ("6", "answer_tag")
        assert extract_final_number("\\boxed{5} (checked 3 times)") == ("5", "boxed")

    def test_latex_thousands_separators_are_normalized(self) -> None:
        value, method = extract_final_number("Answer: 1{,}080")
        assert (value, method) == ("1,080", "answer_tag")
        assert parse_number(value) == 1080.0
        assert extract_final_number("Answer: \\$1\\,500")[0] == "1,500"

    def test_answer_tag_only_looks_three_lines_ahead(self) -> None:
        assert extract_final_number("Answer:\n\n\n\n12") == ("12", "last_number")

    @pytest.mark.parametrize("text", ["", None, "no digits here"])
    def test_nothing_numeric_returns_none(self, text: str | None) -> None:
        assert extract_final_number(text) == (None, "none")


class TestLastBoxed:
    def test_nested_braces_are_kept(self) -> None:
        assert last_boxed("x \\boxed{a{b}c} y") == "a{b}c"

    def test_the_last_box_wins(self) -> None:
        assert last_boxed("\\boxed{1} then \\boxed{2}") == "2"

    @pytest.mark.parametrize("text", ["no box", "\\boxed{unclosed", "\\boxed without brace"])
    def test_missing_or_unbalanced_box_returns_none(self, text: str) -> None:
        assert last_boxed(text) is None


class TestNumbers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("1,234", 1234.0), ("$5", 5.0), (" 2.5 ", 2.5), (3, 3.0), (2.5, 2.5), ("abc", None), (None, None)],
    )
    def test_parse_number(self, raw: object, expected: float | None) -> None:
        assert parse_number(raw) == expected

    def test_numbers_equal_tolerates_formatting_and_float_noise(self) -> None:
        assert numbers_equal("1,000", 1000)
        assert numbers_equal("$1,450,000.00", 1450000)
        assert numbers_equal("0.3", 0.1 + 0.2)  # 0.30000000000000004
        assert numbers_equal("0", "0.0")

    def test_numbers_equal_rejects_different_or_unparseable_values(self) -> None:
        assert not numbers_equal("3", "4")
        assert not numbers_equal("x", "3")
        assert not numbers_equal(None, None)

    def test_numbers_equal_is_exact_near_large_values_unless_asked(self) -> None:
        # scripts/check_graders.py caught this: a 1e-6 relative tolerance graded
        # 1,450,001 correct for GSM8K's 1,450,000.
        assert not numbers_equal("1,450,001", 1450000)
        assert not numbers_equal("0.1", "0.10000001")
        assert numbers_equal("37.83", 37.8, rel_tol=0.001)


class TestExtractChoice:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Thinking... The answer is (C).", ("C", "answer_is")),
            ("the answer is: **B**", ("B", "answer_is")),
            ("Answer: D", ("D", "answer_tag")),
            ("**Answer:** (A)", ("A", "answer_tag")),
            ("\\boxed{\\text{C}}", ("C", "boxed")),
            ("Option B fits best.", ("B", "last_letter")),
        ],
    )
    def test_each_format_is_found_and_labelled(self, text: str, expected: tuple) -> None:
        assert extract_choice(text) == expected

    def test_last_statement_wins_within_a_method(self) -> None:
        text = "The answer is (A). Wait, actually the answer is (B)."
        assert extract_choice(text) == ("B", "answer_is")

    def test_answer_is_outranks_a_later_answer_tag(self) -> None:
        # The MMLU-Pro reference script's order, kept for comparability.
        assert extract_choice("the answer is (B)\nAnswer: A") == ("B", "answer_is")

    def test_a_word_starting_with_a_letter_is_not_an_answer(self) -> None:
        assert extract_choice("The answer is Apple") == (None, "none")

    def test_letters_outside_the_option_range_are_ignored(self) -> None:
        assert extract_choice("The answer is (E)", letters="ABCD") == (None, "none")
        assert extract_choice("B is right, I think", letters="ABCD") == ("B", "last_letter")

    @pytest.mark.parametrize("text", ["", None])
    def test_empty_reply_returns_none(self, text: str | None) -> None:
        assert extract_choice(text) == (None, "none")


class TestExtractCode:
    def test_python_fence_is_extracted(self) -> None:
        text = "Here:\n```python\ndef f():\n    return 1\n```\nDone."
        assert extract_code(text) == "def f():\n    return 1\n"

    def test_block_defining_the_entry_point_beats_a_longer_block(self) -> None:
        helper = "```python\ndef helper():\n    pass\n" + "# padding\n" * 20 + "```"
        target = "```python\ndef target(x):\n    return x\n```"
        assert extract_code(f"{helper}\n{target}", "target") == "def target(x):\n    return x\n"

    def test_async_definition_counts_as_defining(self) -> None:
        text = "```python\nx = 1\n" + "# pad\n" * 9 + "```\n```python\nasync def go():\n    pass\n```"
        assert extract_code(text, "go") == "async def go():\n    pass\n"

    def test_python_blocks_beat_longer_blocks_in_other_languages(self) -> None:
        text = "```bash\npip install a-very-long-package-name and-another\n```\n```python\nx = 1\n```"
        assert extract_code(text) == "x = 1\n"

    def test_without_python_blocks_the_longest_block_is_used(self) -> None:
        assert extract_code("```js\nconsole.log(1)\n```") == "console.log(1)\n"

    def test_without_entry_point_the_longest_python_block_wins(self) -> None:
        text = "```python\nx = 1\n```\n```python\ny = 2\nz = 3\n```"
        assert extract_code(text) == "y = 2\nz = 3\n"

    def test_unterminated_fence_keeps_the_code_after_it(self) -> None:
        assert extract_code("```python\ndef f():\n    return") == "def f():\n    return"

    def test_unfenced_reply_is_returned_whole(self) -> None:
        assert extract_code("def f():\n    return 2") == "def f():\n    return 2"

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("def f(x):\n    return x", True),
            ("class A:\n    def f(self):\n        pass", True),
            ("\n\n\tasync  def f():\n\t\tpass", True),
            ("async\ndef f():\n    pass", True),
            ("def ff(x):\n    return f(x)", False),
            ("x = f(1)\n# def f(", False),
        ],
    )
    def test_a_definition_is_found_at_any_indent(self, code: str, expected: bool) -> None:
        assert defines_python(code, "f") is expected

    @pytest.mark.parametrize(
        ("head", "unit", "count", "tail"),
        [
            # Without a closing fence, the regexes this replaced took quadratic
            # time: each ``` read its line to the end again, and the fence's tag
            # and the rest of its line overlapped. 10.5 s each.
            ("", "`", 100_000, ""),
            ("```", "a", 100_000, ""),
            ("", "```", 33_333, ""),
            # ^\s*def read on across all the blank lines from each one: 8 s.
            ("```python\n", "\n", 20_000, "```"),
        ],
        ids=["backquotes", "open fence", "fences", "blank lines"],
    )
    def test_a_long_reply_is_read_in_linear_time(self, head: str, unit: str, count: int, tail: str) -> None:
        text = head + unit * count + tail
        start = time.process_time()
        extract_code(text, "f")
        assert time.process_time() - start < 0.5


# The regexes `fences` and `open_fence` replaced: the same answers, in quadratic time.
OLD_FENCE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)```", re.DOTALL)
OLD_OPEN_FENCE = re.compile(r"```[ \t]*[\w+-]*[^\n]*\n(.*)$", re.DOTALL)


def test_the_fence_scans_find_what_the_regexes_they_replaced_found() -> None:
    rng = random.Random(9)
    pieces = ["`", "```", "\n", " ", "\t", "a", "python", "+", "-", "x y"]
    for _ in range(3000):
        text = "".join(rng.choices(pieces, k=rng.randint(0, 14)))
        assert fences(text) == OLD_FENCE.findall(text), text
        old = OLD_OPEN_FENCE.search(text)
        assert open_fence(text) == (old.group(1) if old else None), text
