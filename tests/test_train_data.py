"""Protect pure training-data preparation without downloading tokenizers or models."""

import json
from pathlib import Path

import pytest

from kumaru.core.config import REPO_ROOT
from kumaru.core.errors import ConfigError
from kumaru.train.data import (
    IGNORE_INDEX,
    SFTCollator,
    encode_conversation,
    load_chat_jsonl,
    split_rows,
)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self,
        payload: list[dict],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ):
        text = "".join(f"<{m['role']}>" + m["content"] + f"</{m['role']}>" for m in payload)
        if add_generation_prompt:
            text += "<assistant>"
        ids = [ord(char) for char in text]
        return ids if tokenize else text

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_load_chat_jsonl_real_sample_has_31_rows_with_assistant_turns() -> None:
    rows = load_chat_jsonl(REPO_ROOT / "data" / "samples" / "chat_train.jsonl")
    assert len(rows) == 31
    assert all(any(m["role"] == "assistant" for m in row["messages"]) for row in rows)


def test_load_chat_jsonl_converts_alpaca_format(scratch_dir: Path) -> None:
    path = write_jsonl(
        scratch_dir / "alpaca.jsonl",
        [{"instruction": "Do it", "input": "with care", "output": "Done"}],
    )
    rows = load_chat_jsonl(path)
    assert rows == [
        {
            "messages": [
                {"role": "user", "content": "Do it\n\nwith care"},
                {"role": "assistant", "content": "Done"},
            ]
        }
    ]


def test_load_chat_jsonl_malformed_json_reports_line_number(scratch_dir: Path) -> None:
    path = scratch_dir / "bad.jsonl"
    path.write_text(
        '{"messages": [{"role": "assistant", "content": "ok"}]}\n{"bad"',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as excinfo:
        load_chat_jsonl(path)
    assert ":2" in str(excinfo.value)


def test_load_chat_jsonl_skips_rows_with_no_assistant_turn(scratch_dir: Path) -> None:
    path = write_jsonl(
        scratch_dir / "skip.jsonl",
        [
            {"messages": [{"role": "user", "content": "Only user"}]},
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
        ],
    )
    rows = load_chat_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["messages"][0]["content"] == "Q"


def test_split_rows_is_deterministic_for_fixed_seed() -> None:
    rows = [{"id": i, "messages": [{"role": "assistant", "content": str(i)}]} for i in range(30)]
    assert split_rows(rows, 0.2, seed=123) == split_rows(rows, 0.2, seed=123)
    train, eval_rows = split_rows(rows, 0.2, seed=123)
    assert len(train) == 24
    assert len(eval_rows) == 6


def test_encode_conversation_masks_non_assistant_positions(fake_tokenizer: FakeTokenizer) -> None:
    messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    encoded = encode_conversation(messages, fake_tokenizer, max_seq_length=100)
    assert encoded is not None
    labels = encoded["labels"]
    input_ids = encoded["input_ids"]
    first_supervised = next(i for i, label in enumerate(labels) if label != IGNORE_INDEX)
    assert fake_tokenizer.decode(input_ids[:first_supervised]).endswith("<assistant>")
    assert labels[first_supervised:] == input_ids[first_supervised:]
    assert all(label == IGNORE_INDEX for label in labels[:first_supervised])


def test_encode_conversation_returns_none_for_oversized_examples(fake_tokenizer: FakeTokenizer) -> None:
    messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A" * 100}]
    assert encode_conversation(messages, fake_tokenizer, max_seq_length=10) is None


def test_sft_collator_pads_shapes_labels_and_multiple() -> None:
    collator = SFTCollator(pad_token_id=9, pad_to_multiple_of=4)
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "labels": [IGNORE_INDEX, 2, 3]},
            {"input_ids": [4, 5, 6, 7, 8], "labels": [4, 5, 6, 7, 8]},
        ]
    )
    assert tuple(batch["input_ids"].shape) == (2, 8)
    assert tuple(batch["labels"].shape) == (2, 8)
    assert tuple(batch["attention_mask"].shape) == (2, 8)
    assert batch["input_ids"][0, -1].item() == 9
    assert batch["labels"][0, -1].item() == IGNORE_INDEX
    assert batch["labels"][0, -1].item() != 9
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
