"""Protect IFEval's instruction checks, the strict/loose readings, and its four metrics."""

import pytest

from kumaru.eval.benchmarks.base import BenchItem
from kumaru.eval.benchmarks.ifeval import IFEval
from kumaru.eval.benchmarks.ifeval_checks import (
    CHECKS,
    check_instructions,
    constrained_response,
    end_checker,
    json_format,
    keywords_existence,
    keywords_forbidden_words,
    keywords_frequency,
    keywords_letter_frequency,
    loose_variants,
    multiple_sections,
    no_comma,
    nth_paragraph_first_word,
    number_bullet_lists,
    number_highlighted_sections,
    number_paragraphs,
    number_placeholders,
    postscript,
    quotation,
    repeat_prompt,
    title,
    two_responses,
)
from kumaru.eval.benchmarks.runner import ItemRecord


@pytest.fixture
def nltk_ready() -> None:
    """Skip (not fail) where the optional `bench` extra or its data is absent."""
    nltk = pytest.importorskip("nltk")
    pytest.importorskip("regex")
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        pytest.skip("NLTK punkt_tab data missing: python -m nltk.downloader punkt_tab")


def test_all_25_instruction_types_are_registered() -> None:
    assert len(CHECKS) == 25


class TestKeywordChecks:
    def test_existence_is_case_insensitive_and_needs_every_keyword(self) -> None:
        assert keywords_existence("I like Python and RUST.", keywords=["python", "rust"])
        assert not keywords_existence("I like Python.", keywords=["python", "rust"])

    def test_frequency_relations(self) -> None:
        text = "The cat and the hat."
        assert keywords_frequency(text, keyword="the", frequency=2, relation="at least")
        assert not keywords_frequency(text, keyword="the", frequency=2, relation="less than")
        with pytest.raises(ValueError):
            keywords_frequency(text, keyword="the", frequency=2, relation="exactly")

    def test_forbidden_words_match_whole_words_only(self) -> None:
        assert keywords_forbidden_words("Scattered thoughts.", forbidden_words=["cat"])
        assert not keywords_forbidden_words("A CAT sat.", forbidden_words=["cat"])

    def test_letter_frequency_counts_the_requested_character_even_if_not_a_letter(self) -> None:
        # The one deliberate difference from the reference, which substitutes a
        # random letter here and so flips a coin.
        assert keywords_letter_frequency("banana", letter="A", let_frequency=3, let_relation="at least")
        assert keywords_letter_frequency("Wow!!! Great!", letter="!", let_frequency=4, let_relation="at least")
        assert not keywords_letter_frequency("Wow!", letter="!", let_frequency=2, let_relation="at least")


class TestStructureChecks:
    def test_paragraphs_are_split_on_the_markdown_divider(self) -> None:
        assert number_paragraphs("One\n***\nTwo\n***\nThree", num_paragraphs=3)
        assert number_paragraphs("***\nOne\n***\nTwo", num_paragraphs=2)  # leading divider ignored
        assert not number_paragraphs("One\n***\n***\nTwo", num_paragraphs=2)  # empty middle fails

    def test_nth_paragraph_first_word_ignores_case_and_trailing_punctuation(self) -> None:
        text = 'Hello there.\n\n"Second," she said.\n\nThird.'
        assert nth_paragraph_first_word(text, num_paragraphs=3, nth_paragraph=2, first_word="second")
        assert not nth_paragraph_first_word(text, num_paragraphs=3, nth_paragraph=3, first_word="second")
        assert not nth_paragraph_first_word(text, num_paragraphs=2, nth_paragraph=2, first_word="second")

    def test_placeholders_postscripts_and_constrained_answers(self) -> None:
        assert number_placeholders("Dear [name], see [address].", num_placeholders=2)
        assert not number_placeholders("Dear [name].", num_placeholders=2)
        assert postscript("Body.\nP.S. Remember the milk.", postscript_marker="P.S.")
        assert postscript("Body.\nP.P.S extra", postscript_marker="P.P.S")
        assert not postscript("Body only.", postscript_marker="P.S.")
        assert constrained_response("Hmm. My answer is maybe.")
        assert not constrained_response("Yes.")

    def test_bullets_must_match_exactly_and_bold_lines_are_not_bullets(self) -> None:
        text = "* one\n* two\n- three\n**bold** line"
        assert number_bullet_lists(text, num_bullets=3)
        assert not number_bullet_lists(text, num_bullets=2)

    def test_highlights_count_single_and_double_asterisks_but_not_empty_ones(self) -> None:
        assert number_highlighted_sections("*one* and **two**", num_highlights=2)
        assert not number_highlighted_sections("** ** and *one*", num_highlights=2)

    def test_sections_json_and_titles(self) -> None:
        assert multiple_sections("SECTION 1\nIntro\nSECTION 2\nBody", section_spliter="SECTION", num_sections=2)
        assert not multiple_sections("SECTION 1\nIntro", section_spliter="SECTION", num_sections=2)
        assert json_format('```json\n{"a": 1}\n```')
        assert not json_format("not json")
        assert title("<<My Title>>\nText")
        assert not title("<< >> no title")

    def test_combination_and_start_end_checks(self) -> None:
        assert two_responses("Answer one\n******\nAnswer two")
        assert not two_responses("Same\n******\nSame")
        assert not two_responses("A\n******\n******\nB")
        assert repeat_prompt("Write a poem. Here it is.", prompt_to_repeat="write a poem.")
        assert end_checker('Done. Any other questions?"', end_phrase="Any other questions?")
        assert quotation('"Hello"')
        assert not quotation('"')
        assert no_comma("a b")
        assert not no_comma("a, b")


class TestStrictAndLoose:
    def test_loose_variants_drop_first_and_last_lines_and_asterisks(self) -> None:
        variants = loose_variants("*first*\nmiddle\nlast")
        assert len(variants) == 8
        assert variants[:4] == ["*first*\nmiddle\nlast", "middle\nlast", "*first*\nmiddle", "middle"]
        assert variants[4] == "first\nmiddle\nlast"

    def test_a_preamble_fails_strict_but_passes_loose(self) -> None:
        response = 'Sure, here it is:\n"The quoted reply"'
        strict, loose = check_instructions(
            ["startend:quotation", "punctuation:no_comma"], [{}, {}], response
        )
        assert strict == [False, False]
        assert loose == [True, True]

    def test_markdown_asterisks_are_forgiven_only_by_loose(self) -> None:
        strict, loose = check_instructions(["startend:quotation"], [{}], '**"Quoted"**')
        assert (strict, loose) == ([False], [True])

    def test_an_empty_reply_follows_nothing(self) -> None:
        # no_comma would pass on "", but an empty reply must never score.
        assert check_instructions(["punctuation:no_comma"], [{}], "  ") == ([False], [False])


def ifeval_item(key: int, instructions: list[str]) -> BenchItem:
    return BenchItem(
        id=f"ifeval/{key}",
        prompt="p",
        meta={"instructions": instructions, "kwargs": [{} for _ in instructions]},
    )


class TestIFEvalBenchmark:
    def test_grade_needs_every_instruction_and_scores_the_fraction(self) -> None:
        item = ifeval_item(1, ["punctuation:no_comma", "startend:quotation"])
        assert IFEval().grade(item, '"no commas here"').correct
        grade = IFEval().grade(item, '"one, two"')
        assert (grade.correct, grade.score) == (False, 0.5)
        assert grade.detail["strict"] == [False, True]

    def test_aggregate_reports_the_four_official_metrics_and_counts_failures(self) -> None:
        items = {
            "ifeval/1": ifeval_item(1, ["punctuation:no_comma", "startend:quotation"]),
            "ifeval/2": ifeval_item(2, ["punctuation:no_comma", "startend:quotation"]),
            "ifeval/3": ifeval_item(3, ["punctuation:no_comma"]),
        }
        records = [
            ItemRecord(id="ifeval/1", detail={"strict": [True, True], "loose": [True, True]}),
            ItemRecord(id="ifeval/2", detail={"strict": [False, True], "loose": [True, True]}),
            ItemRecord(id="ifeval/3", error="TimeoutError: no reply"),
        ]
        metrics = IFEval().aggregate(records, items)
        assert metrics == pytest.approx(
            {
                "prompt_level_strict": 1 / 3,
                "instruction_level_strict": 3 / 5,
                "prompt_level_loose": 2 / 3,
                "instruction_level_loose": 4 / 5,
                "strict/punctuation": 1 / 3,
                "strict/startend": 1.0,
            }
        )


@pytest.mark.usefixtures("nltk_ready")
class TestCountingWithNltk:
    def test_sentences_use_punkt_so_abbreviations_do_not_split(self) -> None:
        from kumaru.eval.benchmarks.ifeval_checks import count_sentences, number_sentences

        assert count_sentences("Dr. Smith arrived. He sat down.") == 2
        assert number_sentences("One. Two. Three.", num_sentences=3, relation="at least")

    def test_words_are_unicode_word_runs(self) -> None:
        from kumaru.eval.benchmarks.ifeval_checks import count_words, number_words

        assert count_words("Hello, world! It's 2024.") == 5
        assert number_words("one two three", num_words=4, relation="less than")

    def test_capital_words_are_all_caps_tokens(self) -> None:
        from kumaru.eval.benchmarks.ifeval_checks import capital_word_frequency, count_capital_words

        assert count_capital_words("I met NASA staff today.") == 2
        assert capital_word_frequency("I met NASA staff.", capital_frequency=3, capital_relation="less than")


class TestLanguageChecks:
    def test_case_checks_fail_fast_on_mixed_case(self) -> None:
        from kumaru.eval.benchmarks.ifeval_checks import english_capital, english_lowercase

        # Rejected by the case test alone, before any language detection.
        assert not english_capital("Not All Caps")
        assert not english_lowercase("Has Capitals")

    def test_language_detection_distinguishes_english_from_french(self) -> None:
        pytest.importorskip("langdetect")
        from kumaru.eval.benchmarks.ifeval_checks import english_lowercase, response_language

        english = "The weather today is sunny and warm, and everyone is enjoying the afternoon in the park."
        french = "Le temps aujourd'hui est ensoleillé et chaud, et tout le monde profite de l'après-midi au parc."
        assert response_language(english, language="en")
        assert not response_language(french, language="en")
        assert response_language(french, language="fr")
        assert english_lowercase(english.lower())
