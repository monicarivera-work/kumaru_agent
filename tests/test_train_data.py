"""Protect pure training-data preparation without downloading tokenizers or models."""

import json
import tracemalloc
from pathlib import Path

import pytest

# kumaru.train.data imports numpy at the top; the dev extra installs it, so only an install without dev skips this file.
pytest.importorskip("numpy")

import numpy as np

from kumaru.core.config import REPO_ROOT
from kumaru.core.errors import ConfigError
from kumaru.train.data import (
    IGNORE_INDEX,
    PackedTokenDataset,
    SFTCollator,
    encode_conversation,
    load_chat_jsonl,
    pack_corpus_to_memmap,
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
    pytest.importorskip("torch")
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


# Pretraining: token packing. The fake tokenizer maps each character to its code point,
# so the astral characters here produce ids above 65,535 and prove the stream is uint32.
PACK_TEXT = "Kumaru packs text: \u00d1and\u00fa, caf\u00e9, \u6771\u4eac, \U0001f600, \U0001d11e.\n" * 40


def reference_pack(text_file: Path, out_path: Path, tokenizer: FakeTokenizer, chunk_chars: int) -> None:
    """The previous packer, which held every chunk in RAM and wrote one memmap at the end."""
    chunks = []
    with text_file.open("r", encoding="utf-8", errors="replace") as fh:
        while chunk := fh.read(chunk_chars):
            ids = tokenizer(chunk, add_special_tokens=False)["input_ids"]
            chunks.append(np.asarray(ids, dtype=np.uint32))
    flat = np.concatenate(chunks)
    memmap = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(flat.size,))
    memmap[:] = flat
    memmap.flush()
    del memmap


class WatchingTokenizer(FakeTokenizer):
    """Snapshots the output folder each time the packer asks for the next chunk."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.ids: list[int] = []
        self.snapshots: list[tuple[bytes, bytes, bool]] = []

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        on_disk = b"".join(p.read_bytes() for p in sorted(self.out_dir.iterdir()))
        expected = np.asarray(self.ids, dtype=np.uint32).tobytes()
        self.snapshots.append((on_disk, expected, (self.out_dir / "tokens.bin").exists()))
        result = super().__call__(text, add_special_tokens)
        self.ids.extend(result["input_ids"])
        return result


class CrashingTokenizer(FakeTokenizer):
    def __init__(self, crash_on_call: int) -> None:
        self.calls = 0
        self.crash_on_call = crash_on_call

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        self.calls += 1
        if self.calls == self.crash_on_call:
            raise RuntimeError("tokenizer crashed")
        return super().__call__(text, add_special_tokens)


def test_pack_corpus_matches_previous_format_and_replaces_old_output(
    scratch_dir: Path, fake_tokenizer: FakeTokenizer
) -> None:
    corpus = scratch_dir / "corpus.txt"
    corpus.write_bytes(PACK_TEXT.encode("utf-8"))
    reference = scratch_dir / "reference.bin"
    reference_pack(corpus, reference, fake_tokenizer, chunk_chars=7)
    out_dir = scratch_dir / "run"
    out_dir.mkdir()
    (out_dir / "tokens.bin").write_bytes(b"\xff" * 8 * len(PACK_TEXT))  # longer than the result

    dst = pack_corpus_to_memmap(corpus, out_dir / "tokens.bin", fake_tokenizer, chunk_chars=7)

    assert dst == out_dir / "tokens.bin"
    assert dst.read_bytes() == reference.read_bytes()
    assert [p.name for p in out_dir.iterdir()] == ["tokens.bin"]
    expected = np.asarray([ord(char) for char in PACK_TEXT], dtype=np.uint32)
    assert expected.max() > 65_535
    dataset = PackedTokenDataset(dst, seq_length=16)
    assert len(dataset) == (expected.size - 1) // 16
    assert np.array_equal(dataset.tokens, expected)
    del dataset  # release the memmap so Windows can delete the scratch folder


def test_pack_corpus_streams_each_chunk_to_disk_before_reading_the_next(scratch_dir: Path) -> None:
    corpus = scratch_dir / "corpus.txt"
    corpus.write_bytes(b"x" * 50_000)
    out_dir = scratch_dir / "run"
    out_dir.mkdir()
    tokenizer = WatchingTokenizer(out_dir)

    dst = pack_corpus_to_memmap(corpus, out_dir / "tokens.bin", tokenizer, chunk_chars=4_096)

    assert len(tokenizer.snapshots) == 13  # ceil(50,000 / 4,096)
    for on_disk, expected, published in tokenizer.snapshots:
        assert on_disk == expected  # every earlier chunk is already on disk, in order
        assert not published  # tokens.bin appears only once packing is complete
    assert dst.stat().st_size == 4 * 50_000


def test_pack_corpus_failure_keeps_previous_output_and_no_temp_file(scratch_dir: Path) -> None:
    corpus = scratch_dir / "corpus.txt"
    corpus.write_bytes(b"y" * 10_000)
    out_dir = scratch_dir / "run"
    out_dir.mkdir()
    dst = out_dir / "tokens.bin"
    dst.write_bytes(b"previous")

    with pytest.raises(RuntimeError, match="tokenizer crashed"):
        pack_corpus_to_memmap(corpus, dst, CrashingTokenizer(crash_on_call=3), chunk_chars=1_000)

    assert [p.name for p in out_dir.iterdir()] == ["tokens.bin"]
    assert dst.read_bytes() == b"previous"


def test_pack_corpus_with_zero_tokens_raises_and_writes_nothing(
    scratch_dir: Path, fake_tokenizer: FakeTokenizer
) -> None:
    corpus = scratch_dir / "empty.txt"
    corpus.write_bytes(b"")
    out_dir = scratch_dir / "run"

    with pytest.raises(ConfigError, match="produced zero tokens"):
        pack_corpus_to_memmap(corpus, out_dir / "tokens.bin", fake_tokenizer)

    assert list(out_dir.iterdir()) == []


def test_pack_corpus_memory_stays_bounded_by_one_chunk(scratch_dir: Path, fake_tokenizer: FakeTokenizer) -> None:
    corpus = scratch_dir / "corpus.txt"
    corpus.write_bytes(b"z" * 2_000_000)  # 2M tokens: 8 MB of uint32 ids

    tracemalloc.start()
    try:
        pack_corpus_to_memmap(corpus, scratch_dir / "run" / "tokens.bin", fake_tokenizer, chunk_chars=20_000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Holding the stream in RAM would peak at 16 MB (the ids plus a concatenated copy).
    assert peak < 2_000_000
