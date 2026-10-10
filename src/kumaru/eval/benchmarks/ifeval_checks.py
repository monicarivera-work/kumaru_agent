"""IFEval's 25 verifiable instructions, each a small pure function.

IFEval asks for things a program can check — "answer in under 300 words",
"wrap the whole reply in double quotes", "include the keywords X and Y", "end
with 'Is there anything else I can help with?'". No judge model and no opinion:
an instruction is either followed or not.

These functions reimplement the published semantics of the reference
implementation (google-research `instruction_following_eval`), including its
quirks, so scores stay comparable with published numbers. Counting uses the
same tools as the reference — NLTK's pretrained Punkt model for sentences, its
word tokenizer for capitalised words, `\\w+` runs for words — because counts
decide verdicts at the thresholds, and hand-rolled regexes drift from NLTK on
lists, contractions and non-Latin scripts.

Checked against the reference on GPT-4's 541 published responses: all
instruction verdicts agree except two, both from one deliberate difference. When
`keywords:letter_frequency` names a non-letter ("#", "!"), the reference swaps
in a *random* letter, so its verdict is a coin flip; we count the character the
prompt asked for.

Strict vs loose (from the paper): *strict* checks the reply as written;
*loose* also accepts it after removing markdown asterisks and/or the first or
last line, forgiving "Sure, here it is:" preambles and sign-offs.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable
from typing import Any

from kumaru.core.errors import DependencyMissingError, KumaruError


def _compare(count: int, relation: str, target: Any) -> bool:
    if relation == "less than":
        return count < int(target)
    if relation == "at least":
        return count >= int(target)
    raise ValueError(f"unknown IFEval relation: {relation!r}")


def detect_language(text: str) -> str | None:
    """ISO 639-1 code of `text`, or None when it cannot be determined."""
    try:
        from langdetect import DetectorFactory, detect
        from langdetect.lang_detect_exception import LangDetectException
    except ImportError as exc:
        raise DependencyMissingError("langdetect", extra="bench") from exc
    DetectorFactory.seed = 0  # langdetect is randomized unless seeded
    try:
        return detect(text)
    except LangDetectException:
        return None


def _nltk() -> Any:
    try:
        import nltk
    except ImportError as exc:
        raise DependencyMissingError("nltk", extra="bench") from exc
    return nltk


def ensure_nltk_data() -> None:
    """Download NLTK's Punkt sentence model (a few MB) on first use."""
    nltk = _nltk()
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        if not nltk.download("punkt_tab", quiet=True):
            raise KumaruError(
                "Could not download NLTK's 'punkt_tab' sentence model.",
                hint="Connect to the internet once, or run: python -m nltk.downloader punkt_tab",
            ) from None


@functools.lru_cache(maxsize=1)
def _sentence_tokenizer() -> Any:
    ensure_nltk_data()
    return _nltk().tokenize.PunktTokenizer("english")


def count_sentences(text: str) -> int:
    return len(_sentence_tokenizer().tokenize(text))


def count_words(text: str) -> int:
    # The `regex` engine (an NLTK dependency, and what NLTK itself now uses)
    # keeps combining marks inside words; `re` would split Hindi or Thai words
    # at every vowel sign.
    try:
        import regex
    except ImportError as exc:
        raise DependencyMissingError("regex", extra="bench") from exc
    return len(regex.findall(r"\w+", text))


def count_capital_words(text: str) -> int:
    """Tokens written entirely in capitals ("I", "NASA", and the "'M" of "I'M")."""
    _sentence_tokenizer()  # word_tokenize needs the Punkt data too
    return sum(1 for token in _nltk().word_tokenize(text) if token.isupper())


# -- keywords -----------------------------------------------------------------

def keywords_existence(response: str, *, keywords: list[str], **_: Any) -> bool:
    return all(re.search(re.escape(k), response, re.IGNORECASE) for k in keywords)


def keywords_frequency(response: str, *, keyword: str, frequency: int, relation: str, **_: Any) -> bool:
    count = len(re.findall(re.escape(keyword), response, re.IGNORECASE))
    return _compare(count, relation, frequency)


def keywords_forbidden_words(response: str, *, forbidden_words: list[str], **_: Any) -> bool:
    return not any(
        re.search(rf"\b{re.escape(word)}\b", response, re.IGNORECASE) for word in forbidden_words
    )


def keywords_letter_frequency(response: str, *, letter: str, let_frequency: int, let_relation: str, **_: Any) -> bool:
    # Deliberate difference: the reference checks a random letter when `letter`
    # is not a-z (see the module docstring); we count what the prompt asked for.
    return _compare(response.lower().count(letter.lower()), let_relation, let_frequency)


# -- language -----------------------------------------------------------------

def response_language(response: str, *, language: str, **_: Any) -> bool:
    detected = detect_language(response)
    # The reference counts "could not detect" as followed.
    return detected is None or detected == language


# -- length constraints -------------------------------------------------------

def number_sentences(response: str, *, num_sentences: int, relation: str, **_: Any) -> bool:
    return _compare(count_sentences(response), relation, num_sentences)


def number_paragraphs(response: str, *, num_paragraphs: int, **_: Any) -> bool:
    # Paragraphs are separated by the markdown divider ***.
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index in (0, len(paragraphs) - 1):
                count -= 1
            else:
                return False
    return count == int(num_paragraphs)


def number_words(response: str, *, num_words: int, relation: str, **_: Any) -> bool:
    return _compare(count_words(response), relation, num_words)


def nth_paragraph_first_word(
    response: str, *, num_paragraphs: int, nth_paragraph: int, first_word: str, **_: Any
) -> bool:
    # Paragraphs are separated by blank lines here (not ***).
    paragraphs = re.split(r"\n\n", response)
    count = sum(1 for paragraph in paragraphs if paragraph.strip())
    nth = int(nth_paragraph)
    if nth > count:
        return False
    # Reference quirk kept on purpose: indexes the raw split, empties included.
    paragraph = paragraphs[nth - 1].strip()
    if not paragraph:
        return False
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    found = ""
    for char in word:
        if char in {".", ",", "?", "!", "'", '"'}:
            break
        found += char.lower()
    return count == int(num_paragraphs) and found == first_word.lower()


# -- detectable content -------------------------------------------------------

def number_placeholders(response: str, *, num_placeholders: int, **_: Any) -> bool:
    return len(re.findall(r"\[.*?\]", response)) >= int(num_placeholders)


def postscript(response: str, *, postscript_marker: str, **_: Any) -> bool:
    text = response.lower()
    if postscript_marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif postscript_marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + re.escape(postscript_marker.lower()) + r".*$"
    return bool(re.findall(pattern, text, flags=re.MULTILINE))


# -- detectable format --------------------------------------------------------

def number_bullet_lists(response: str, *, num_bullets: int, **_: Any) -> bool:
    stars = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
    return len(stars) + len(dashes) == int(num_bullets)


def constrained_response(response: str, **_: Any) -> bool:
    text = response.strip()
    return any(option in text for option in ("My answer is yes.", "My answer is no.", "My answer is maybe."))


def number_highlighted_sections(response: str, *, num_highlights: int, **_: Any) -> bool:
    singles = [h for h in re.findall(r"\*[^\n\*]*\*", response) if h.strip("*").strip()]
    doubles = [
        h for h in re.findall(r"\*\*[^\n\*]*\*\*", response)
        if h.removeprefix("**").removesuffix("**").strip()
    ]
    return len(singles) + len(doubles) >= int(num_highlights)


def multiple_sections(response: str, *, section_spliter: str, num_sections: int, **_: Any) -> bool:
    sections = re.split(r"\s?" + re.escape(section_spliter) + r"\s?\d+\s?", response)
    return len(sections) - 1 >= int(num_sections)


def json_format(response: str, **_: Any) -> bool:
    text = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def title(response: str, **_: Any) -> bool:
    return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", response))


# -- combination ----------------------------------------------------------------

def two_responses(response: str, **_: Any) -> bool:
    parts = response.split("******")
    valid = []
    for index, part in enumerate(parts):
        if not part.strip():
            if index not in (0, len(parts) - 1):
                return False
        else:
            valid.append(part)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def repeat_prompt(response: str, *, prompt_to_repeat: str, **_: Any) -> bool:
    return response.strip().lower().startswith(prompt_to_repeat.strip().lower())


# -- start / end ------------------------------------------------------------------

def end_checker(response: str, *, end_phrase: str, **_: Any) -> bool:
    return response.strip().strip('"').lower().endswith(end_phrase.strip().lower())


def quotation(response: str, **_: Any) -> bool:
    text = response.strip()
    return len(text) > 1 and text[0] == '"' and text[-1] == '"'


# -- change case ------------------------------------------------------------------

def capital_word_frequency(response: str, *, capital_frequency: int, capital_relation: str, **_: Any) -> bool:
    return _compare(count_capital_words(response), capital_relation, capital_frequency)


def english_capital(response: str, **_: Any) -> bool:
    if not response.isupper():
        return False
    detected = detect_language(response)
    return detected is None or detected == "en"


def english_lowercase(response: str, **_: Any) -> bool:
    if not response.islower():
        return False
    detected = detect_language(response)
    return detected is None or detected == "en"


# -- punctuation --------------------------------------------------------------------

def no_comma(response: str, **_: Any) -> bool:
    return "," not in response


CHECKS: dict[str, Callable[..., bool]] = {
    "keywords:existence": keywords_existence,
    "keywords:frequency": keywords_frequency,
    "keywords:forbidden_words": keywords_forbidden_words,
    "keywords:letter_frequency": keywords_letter_frequency,
    "language:response_language": response_language,
    "length_constraints:number_sentences": number_sentences,
    "length_constraints:number_paragraphs": number_paragraphs,
    "length_constraints:number_words": number_words,
    "length_constraints:nth_paragraph_first_word": nth_paragraph_first_word,
    "detectable_content:number_placeholders": number_placeholders,
    "detectable_content:postscript": postscript,
    "detectable_format:number_bullet_lists": number_bullet_lists,
    "detectable_format:constrained_response": constrained_response,
    "detectable_format:number_highlighted_sections": number_highlighted_sections,
    "detectable_format:multiple_sections": multiple_sections,
    "detectable_format:json_format": json_format,
    "detectable_format:title": title,
    "combination:two_responses": two_responses,
    "combination:repeat_prompt": repeat_prompt,
    "startend:end_checker": end_checker,
    "startend:quotation": quotation,
    "change_case:capital_word_frequency": capital_word_frequency,
    "change_case:english_capital": english_capital,
    "change_case:english_lowercase": english_lowercase,
    "punctuation:no_comma": no_comma,
}


def loose_variants(response: str) -> list[str]:
    """The eight relaxed readings of a reply used by IFEval's loose metric."""
    lines = response.split("\n")
    without_first = "\n".join(lines[1:]).strip()
    without_last = "\n".join(lines[:-1]).strip()
    without_both = "\n".join(lines[1:-1]).strip()
    variants = [response, without_first, without_last, without_both]
    return variants + [v.replace("*", "") for v in variants]


def check_instructions(
    instruction_ids: list[str], kwargs_list: list[dict[str, Any]], response: str
) -> tuple[list[bool], list[bool]]:
    """Return `(strict, loose)` follow flags, one per instruction."""
    strict: list[bool] = []
    loose: list[bool] = []
    variants = loose_variants(response)
    for instruction_id, kwargs in zip(instruction_ids, kwargs_list, strict=True):
        check = CHECKS[instruction_id]
        strict.append(bool(response.strip()) and check(response, **kwargs))
        loose.append(any(v.strip() and check(v, **kwargs) for v in variants))
    return strict, loose
