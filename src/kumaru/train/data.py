"""Training data: loading, chat formatting, loss masking, and token packing.

This file contains the two ideas that determine whether fine-tuning works at
all. Both are easy to get wrong and produce a model that trains smoothly to a
low loss while being useless.

**1. Loss masking (`train_on_completions_only`).**
An SFT example contains the system prompt, the user's question, and the
assistant's answer. If you compute loss over all of it, the model spends most of
its gradient budget learning to *predict questions* — a task you will never ask
it to perform. Setting the label to -100 (PyTorch's ignore index) on every
non-assistant token concentrates the entire learning signal on the behaviour you
actually want. This routinely makes the difference between a fine-tune that
helps and one that quietly degrades the model.

**2. Chat-template fidelity.**
The model was trained with exact control tokens (`<|im_start|>assistant\\n`).
Training with even slightly different formatting teaches it a *second*,
conflicting convention and degrades instruction-following. We never hand-write
the format: we call `tokenizer.apply_chat_template`, which reads it from the
checkpoint, so it is correct for whatever model you point at.

Packing (for pretraining) is a different concern: concatenate the whole corpus
into one long token stream and slice fixed-length windows. No padding, so every
token in every batch contributes to the loss — typically a 2-4x throughput win
over padding variable-length documents.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kumaru.core.errors import ConfigError
from kumaru.core.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100  # PyTorch cross-entropy skips targets equal to this


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_chat_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file of chat conversations, validating as we go.

    Expected per line::

        {"messages": [{"role": "user", "content": "..."},
                      {"role": "assistant", "content": "..."}]}

    Also accepts the common ``{"instruction", "input", "output"}`` (Alpaca)
    shape and converts it, because a large share of public datasets use it.

    Validation is strict and reports line numbers: a malformed example that
    slips through silently costs you an entire training run to discover.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Training data not found: {p}")

    rows: list[dict[str, Any]] = []
    skipped = 0

    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{p}:{lineno} is not valid JSON: {exc}") from exc

            if "instruction" in obj and "output" in obj:
                user = obj["instruction"]
                if obj.get("input"):
                    user = f"{user}\n\n{obj['input']}"
                obj = {
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": obj["output"]},
                    ]
                }

            messages = obj.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ConfigError(f"{p}:{lineno} has no non-empty 'messages' list")

            for m in messages:
                if not isinstance(m, dict) or "role" not in m or "content" not in m:
                    raise ConfigError(f"{p}:{lineno} has a message missing 'role' or 'content'")

            # An example with no assistant turn contributes zero gradient under
            # completion-only masking, so drop it rather than waste a batch slot.
            if not any(m["role"] == "assistant" and m["content"].strip() for m in messages):
                skipped += 1
                continue

            rows.append({"messages": messages})

    if not rows:
        raise ConfigError(f"{p} contains no usable examples")
    if skipped:
        logger.warning("skipped %d example(s) with no assistant turn", skipped)

    logger.info("loaded %d example(s) from %s", len(rows), p)
    return rows


def split_rows(
    rows: list[dict[str, Any]], eval_split: float, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically hold out an evaluation slice.

    Seeded so the split is identical across runs — otherwise eval loss is not
    comparable between two experiments and you cannot tell which change helped.
    """
    if eval_split <= 0 or len(rows) < 20:
        return rows, []
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    n_eval = max(1, int(len(rows) * eval_split))
    eval_idx = set(idx[:n_eval].tolist())
    return (
        [r for i, r in enumerate(rows) if i not in eval_idx],
        [r for i, r in enumerate(rows) if i in eval_idx],
    )


# ---------------------------------------------------------------------------
# Chat encoding with loss masking
# ---------------------------------------------------------------------------


def _template_ids(tokenizer: Any, payload: list[dict[str, Any]], *, add_generation_prompt: bool = False) -> list[int]:
    """Call `apply_chat_template` and always get back a flat list of token ids.

    Necessary because the return type has changed across transformers releases:
    older versions return `list[int]`, newer ones default to `return_dict=True`
    and hand back a `BatchEncoding`, and some paths return a nested batch. Left
    unnormalized, the newer shape silently iterates dictionary *keys* — the
    encoder appears to work while producing nonsense.
    """
    out = tokenizer.apply_chat_template(
        payload,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )

    # BatchEncoding / dict -> pull out the ids
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    # torch tensor or numpy array -> list
    if hasattr(out, "tolist"):
        out = out.tolist()
    # batch of one conversation -> unwrap
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]

    return [int(t) for t in out]


def encode_conversation(
    messages: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_length: int,
    completions_only: bool = True,
) -> dict[str, list[int]] | None:
    """Tokenize one conversation into `input_ids` and masked `labels`.

    Returns ``None`` when the example is longer than `max_seq_length` — dropped
    rather than truncated, because a truncated answer teaches the model to stop
    mid-sentence.

    The technique: render the conversation twice per assistant turn — once up to
    the generation prompt, once including the answer — and take the token
    difference. That isolates exactly the answer tokens without ever assuming
    what the template's control tokens look like.
    """
    payload = [{"role": m["role"], "content": m["content"]} for m in messages]

    full_ids = _template_ids(tokenizer, payload)
    if len(full_ids) > max_seq_length:
        return None

    if not completions_only:
        return {"input_ids": full_ids, "labels": list(full_ids)}

    labels = [IGNORE_INDEX] * len(full_ids)

    for i, msg in enumerate(payload):
        if msg["role"] != "assistant":
            continue
        # Prefix ends immediately before the answer text (after the assistant header).
        prefix = _template_ids(tokenizer, payload[:i], add_generation_prompt=True)
        # `through` ends after the answer and its end-of-turn token. Supervising
        # that terminator is what teaches the model to stop.
        through = _template_ids(tokenizer, payload[: i + 1])

        start, end = len(prefix), min(len(through), len(full_ids))
        if start >= end or full_ids[:start] != prefix[:start]:
            # Template is not strictly prefix-stable; fall back to supervising
            # the whole example rather than silently training on nothing.
            logger.debug("chat template not prefix-stable; supervising full sequence")
            return {"input_ids": full_ids, "labels": list(full_ids)}

        labels[start:end] = full_ids[start:end]

    if all(label == IGNORE_INDEX for label in labels):
        return None

    return {"input_ids": full_ids, "labels": labels}


def build_sft_dataset(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_length: int,
    completions_only: bool = True,
) -> list[dict[str, list[int]]]:
    """Encode every conversation, reporting how much data survived."""
    encoded: list[dict[str, list[int]]] = []
    dropped = 0

    for row in rows:
        item = encode_conversation(
            row["messages"],
            tokenizer,
            max_seq_length=max_seq_length,
            completions_only=completions_only,
        )
        if item is None:
            dropped += 1
        else:
            encoded.append(item)

    if not encoded:
        raise ConfigError(
            "Every example was dropped during encoding.",
            hint=f"Raise data.max_seq_length above {max_seq_length}, or shorten your examples.",
        )

    if dropped:
        pct = 100 * dropped / len(rows)
        level = logger.warning if pct > 10 else logger.info
        level("dropped %d/%d example(s) (%.1f%%) longer than %d tokens",
              dropped, len(rows), pct, max_seq_length)

    supervised = sum(sum(1 for x in e["labels"] if x != IGNORE_INDEX) for e in encoded)
    total = sum(len(e["input_ids"]) for e in encoded)
    logger.info(
        "encoded %d example(s): %d tokens, %d supervised (%.1f%%)",
        len(encoded), total, supervised, 100 * supervised / max(1, total),
    )
    return encoded


class SFTCollator:
    """Pad a batch to its longest sequence and build the attention mask.

    Padding to the batch max rather than a global max is "dynamic padding": with
    sorted-ish data it cuts wasted compute substantially, because most batches
    are far shorter than the longest example in the dataset.

    Labels are padded with -100, never with the pad token. Padding labels with a
    real token id would train the model to emit padding.
    """

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        # Tensor cores want dimensions that are multiples of 8 (fp16/bf16);
        # rounding up here is free and measurably faster.
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        longest = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of > 1:
            longest = -(-longest // self.pad_to_multiple_of) * self.pad_to_multiple_of

        input_ids, labels, attention = [], [], []
        for f in features:
            n_pad = longest - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * n_pad)
            labels.append(f["labels"] + [IGNORE_INDEX] * n_pad)
            attention.append([1] * len(f["input_ids"]) + [0] * n_pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Pretraining: token packing
# ---------------------------------------------------------------------------


def pack_corpus_to_memmap(
    text_file: str | Path,
    out_path: str | Path,
    tokenizer: Any,
    *,
    chunk_chars: int = 1_000_000,
) -> Path:
    """Tokenize a raw text corpus into a flat `uint32` memory-mapped array.

    Why a memmap rather than a list of tensors: pretraining corpora are larger
    than RAM. A memmap lets the OS page in only the slices you touch, so the
    same code trains on a 10 MB toy file or a 200 GB corpus with no change and
    no loading delay at startup.

    `uint32` rather than `uint16` because modern vocabularies exceed 65,536
    tokens (Qwen2.5 is ~152k). Using uint16 would silently corrupt token ids —
    a bug that shows up only as mysteriously bad model quality.
    """
    src, dst = Path(text_file), Path(out_path)
    if not src.exists():
        raise ConfigError(f"Corpus not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    all_ids: list[np.ndarray] = []
    total = 0

    with src.open("r", encoding="utf-8", errors="replace") as fh:
        while chunk := fh.read(chunk_chars):
            ids = tokenizer(chunk, add_special_tokens=False)["input_ids"]
            arr = np.asarray(ids, dtype=np.uint32)
            all_ids.append(arr)
            total += arr.size
            logger.info("tokenized %s tokens…", f"{total:,}")

    if total == 0:
        raise ConfigError(f"{src} produced zero tokens")

    flat = np.concatenate(all_ids)
    memmap = np.memmap(dst, dtype=np.uint32, mode="w+", shape=(flat.size,))
    memmap[:] = flat
    memmap.flush()

    logger.info("wrote %s tokens to %s (%.1f MB)", f"{total:,}", dst, dst.stat().st_size / 1e6)
    return dst


class PackedTokenDataset:
    """Fixed-length windows over a packed token stream.

    Language modelling predicts token *t+1* from tokens *<= t*, so inputs and
    targets are the same slice offset by one. Every position in every window
    produces a training signal — there is no padding and no wasted compute.
    """

    def __init__(self, memmap_path: str | Path, seq_length: int) -> None:
        self.path = Path(memmap_path)
        self.seq_length = seq_length
        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        # -1 because each window needs one extra token to form its target.
        self.n_windows = max(0, (len(self.tokens) - 1) // seq_length)
        if self.n_windows == 0:
            raise ConfigError(
                f"{self.path} holds {len(self.tokens)} tokens, too few for "
                f"seq_length={seq_length}."
            )

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        start = idx * self.seq_length
        window = np.asarray(self.tokens[start : start + self.seq_length + 1], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(window[:-1]),
            "labels": torch.from_numpy(window[1:]),
        }

    def iter_batches(self, batch_size: int, shuffle: bool = True, seed: int = 0) -> Iterator[dict]:
        """Yield batches forever, reshuffling on each pass over the data.

        Infinite rather than epoch-bounded because pretraining is measured in
        *steps*, not epochs — the loop asks for as many batches as the schedule
        needs and never has to handle exhaustion. It also means a corpus smaller
        than one batch still trains (by sampling with replacement) instead of
        raising `StopIteration` on the first step.
        """
        import torch

        rng = np.random.default_rng(seed)
        pass_index = 0

        while True:
            order = np.arange(self.n_windows)
            if shuffle:
                rng.shuffle(order)

            if self.n_windows < batch_size:
                # Corpus smaller than one batch: draw with replacement so the
                # loop still runs. Expect immediate overfitting — which is the
                # correct behaviour to observe on a toy corpus.
                order = rng.integers(0, self.n_windows, size=batch_size)

            for i in range(0, len(order) - batch_size + 1, batch_size):
                items = [self[int(j)] for j in order[i : i + batch_size]]
                yield {
                    "input_ids": torch.stack([it["input_ids"] for it in items]),
                    "labels": torch.stack([it["labels"] for it in items]),
                }

            pass_index += 1
            rng = np.random.default_rng(seed + pass_index)
