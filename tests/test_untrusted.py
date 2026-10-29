"""Text from files and programs reaches the model with its chat control tokens escaped.

llama-server recognises control tokens anywhere in the prompt, so a file that
holds ``<|im_end|>`` and ``<|im_start|>system`` could open a real system turn.
These tests check that escaping catches every kind of token, leaves everything
else alone, leaves nothing to catch when applied twice, and that
`restore_control_tokens` gives back exactly what was escaped, which is what
lets `edit_file` take text the model copied from a tool result.
"""

import ast
import random
from pathlib import Path

import pytest

from kumaru.agent import untrusted
from kumaru.agent.untrusted import (
    CONTROL_TOKEN,
    MAX_CONTROL_TOKEN_CHARS,
    neutralize_control_tokens,
    restore_control_tokens,
)


@pytest.mark.parametrize(
    ("token", "escaped"),
    [
        ("<|im_start|>", "&lt;|im_start|&gt;"),
        ("<|end▁of▁sentence|>", "&lt;|end▁of▁sentence|&gt;"),
        (
            "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
            "&lt;\uff5cbegin\u2581of\u2581sentence\uff5c&gt;",
        ),
        ("<|a[b]|>", "&lt;|a&#91;b&#93;|&gt;"),
        ("<s>", "&lt;s&gt;"),
        ("</s>", "&lt;/s&gt;"),
        ("<think>", "&lt;think&gt;"),
        ("</tool_call>", "&lt;/tool_call&gt;"),
        ("<start_of_turn>", "&lt;start_of_turn&gt;"),
        ("<<SYS>>", "&lt;&lt;SYS&gt;&gt;"),
        ("<</SYS>>", "&lt;&lt;/SYS&gt;&gt;"),
        ("[INST]", "&#91;INST&#93;"),
        ("[/INST]", "&#91;/INST&#93;"),
        ("[TOOL_CALLS]", "&#91;TOOL_CALLS&#93;"),
    ],
)
def test_every_kind_of_control_token_is_escaped_and_comes_back(token: str, escaped: str) -> None:
    text = f"before {token} after"
    assert neutralize_control_tokens(text) == f"before {escaped} after"
    assert restore_control_tokens(f"before {escaped} after") == text


@pytest.mark.parametrize(
    "text",
    [
        "if a < b and c > d: pass",
        "<div class='x'>[0]</div>",
        "<S> <ss> <think > [inst] [ INST ]",
        "<|two words|> <||> <|a|b|>",
        "<|" + "a" * 65 + "|>",
        "&lt;div&gt; &amp; &#91;0&#93;",
    ],
)
def test_other_text_is_left_alone(text: str) -> None:
    assert neutralize_control_tokens(text) == text
    assert restore_control_tokens(text) == text


def test_the_longest_token_is_max_control_token_chars_long() -> None:
    longest = "<|" + "a" * 64 + "|>"
    assert CONTROL_TOKEN.fullmatch(longest) and len(longest) == MAX_CONTROL_TOKEN_CHARS
    assert CONTROL_TOKEN.fullmatch("<\uff5c" + "a" * 64 + "\uff5c>")
    assert not CONTROL_TOKEN.search("<|" + "a" * 65 + "|>")
    # The longest come back too, even a name of 64 brackets: 320 characters escaped.
    for token in [longest, "<|" + "[]" * 32 + "|>", "<\uff5c" + "]" * 64 + "\uff5c>"]:
        assert restore_control_tokens(neutralize_control_tokens(token)) == token
    # Restoring undoes only what escaping can produce.
    too_long = "&lt;|" + "a" * 65 + "|&gt;"
    assert restore_control_tokens(too_long) == too_long


# Pieces that combine into tokens, near misses, and tokens inside tokens.
PIECES = [
    "<", ">", "|", "[", "]", "/", "<|", "|>", "\uff5c", "<\uff5c", "\uff5c>", " ", "\n", "x",
    "s", "im_start", "SYS", "INST", "THINK", "think", "tool_call", "<s>", "<|im_end|>", "[INST]",
]  # fmt: skip
ENTITY_PIECES = ["&", "&lt;", "&gt;", "&#91;", "&#93;", "lt;", "#91;"]
# What is left of a text once brackets and the characters of their entities go.
SKELETON = str.maketrans(dict.fromkeys("<>[]&lgt;#913", ""))


def fuzz(pieces: list[str], seed: int, count: int = 4000) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choices(pieces, k=rng.randint(1, 24))) for _ in range(count)]


def test_no_control_token_survives_and_escaping_twice_changes_nothing() -> None:
    for text in fuzz(PIECES + ENTITY_PIECES, seed=8):
        once = neutralize_control_tokens(text)
        assert CONTROL_TOKEN.search(once) is None, text
        assert neutralize_control_tokens(once) == once, text
        # Only brackets change, into entities: without them the texts are the same.
        assert once.translate(SKELETON) == text.translate(SKELETON), text


def test_restore_gives_back_exactly_what_was_escaped() -> None:
    # Text that already holds "&lt;s&gt;" looks the same as escaped text, which is
    # why edit_file tries old_text as it is first.
    for text in fuzz(PIECES, seed=7):
        assert restore_control_tokens(neutralize_control_tokens(text)) == text, text


def test_the_module_imports_nothing_from_kumaru() -> None:
    # Any module may import it, so it must not import any of them.
    tree = ast.parse(Path(untrusted.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {"__future__", "re"}
