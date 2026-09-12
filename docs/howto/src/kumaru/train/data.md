# `src\kumaru\train\data.py`

> Training-data utilities for loading chat JSONL, masking SFT loss, dynamically padding batches, and packing raw text into token memmaps.

**Read this when:** you are preparing SFT data, investigating why loss decreases but quality regresses, adding a new dataset format, or sizing a pretraining corpus.

---

## What it does

This file owns Kumaru's training data path. For supervised fine-tuning, it validates JSONL conversations, converts Alpaca-style rows, applies the model's own chat template, masks non-assistant tokens with `-100`, and builds padded batches.

For pretraining, it tokenizes raw text into a `uint32` memory-mapped token stream and exposes fixed-length next-token prediction windows. That avoids padding and lets the same code handle tiny and large corpora.

## Why it exists

Two quiet data bugs can ruin an LLM fine-tune while the loss still looks healthy. If loss is computed on user prompts, the model learns to predict questions instead of answers. If chat control tokens are hand-written incorrectly, training teaches a conflicting format.

`data.py` prevents both by using `tokenizer.apply_chat_template()` and by supervising assistant turns only. It also drops overlong examples instead of truncating answers mid-sentence.

## Mental model

1. **SFT examples are conversations; loss belongs on assistant output.** Labels are `-100` everywhere PyTorch should ignore.
2. **The tokenizer owns the chat format.** Kumaru never hard-codes model-specific control tokens.
3. **Pretraining is packed next-token prediction.** A long token array becomes fixed windows where inputs predict labels shifted by one.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `IGNORE_INDEX` | `int = -100` | Label value ignored by PyTorch cross entropy. |
| `load_chat_jsonl` | `load_chat_jsonl(path: str | Path) -> list[dict[str, Any]]` | Reads chat JSONL or Alpaca rows and validates them. |
| `split_rows` | `split_rows(rows: list[dict[str, Any]], eval_split: float, seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]` | Deterministically creates a train/eval split. |
| `encode_conversation` | `encode_conversation(messages: Sequence[dict[str, Any]], tokenizer: Any, *, max_seq_length: int, completions_only: bool = True) -> dict[str, list[int]] | None` | Tokenizes one conversation and builds masked labels. |
| `build_sft_dataset` | `build_sft_dataset(rows: list[dict[str, Any]], tokenizer: Any, *, max_seq_length: int, completions_only: bool = True) -> list[dict[str, list[int]]]` | Encodes all conversations and reports dropped/supervised token counts. |
| `SFTCollator` | `SFTCollator(pad_token_id: int, pad_to_multiple_of: int = 8)` | Dynamic-padding collator for Hugging Face `Trainer`. |
| `SFTCollator.__call__` | `__call__(features: list[dict[str, list[int]]]) -> dict[str, Any]` | Builds `input_ids`, `labels`, and `attention_mask` tensors. |
| `pack_corpus_to_memmap` | `pack_corpus_to_memmap(text_file: str | Path, out_path: str | Path, tokenizer: Any, *, chunk_chars: int = 1_000_000) -> Path` | Tokenizes raw text to a flat `uint32` memmap. |
| `PackedTokenDataset` | `PackedTokenDataset(memmap_path: str | Path, seq_length: int)` | Dataset of fixed-length packed token windows. |
| `PackedTokenDataset.__len__` | `__len__() -> int` | Returns number of training windows. |
| `PackedTokenDataset.__getitem__` | `__getitem__(idx: int) -> dict[str, Any]` | Returns one window with shifted `input_ids` and `labels`. |
| `PackedTokenDataset.iter_batches` | `iter_batches(batch_size: int, shuffle: bool = True, seed: int = 0) -> Iterator[dict]` | Yields stacked tensor batches for the custom pretraining loop. |

## How to use it

Expected SFT JSONL format:

```jsonl
{"messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"What is Kumaru?"},{"role":"assistant","content":"Kumaru is a local-first LLM agent."}]}
{"messages":[{"role":"user","content":"Return JSON."},{"role":"assistant","content":"{\"ok\": true}"}]}
```

Alpaca-style rows are accepted and converted:

```jsonl
{"instruction":"Summarize this","input":"Kumaru trains adapters.","output":"Kumaru can train LoRA adapters."}
```

Validate the real training path and write a masking preview:

```powershell
kumaru train configs\train\lora-small.yaml --dry-run
Get-Content runs\lora-small\example_preview.txt
```

Encode SFT data directly in Python:

```python
from transformers import AutoTokenizer
from kumaru.train.data import load_chat_jsonl, split_rows, build_sft_dataset, SFTCollator

rows = load_chat_jsonl("data/samples/chat_train.jsonl")
train_rows, eval_rows = split_rows(rows, eval_split=0.1, seed=42)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
items = build_sft_dataset(train_rows, tok, max_seq_length=1024, completions_only=True)
batch = SFTCollator(tok.pad_token_id)(items[:2])
print(batch["input_ids"].shape)
```

Pack a raw corpus for pretraining:

```python
from transformers import AutoTokenizer
from kumaru.train.data import pack_corpus_to_memmap, PackedTokenDataset

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
path = pack_corpus_to_memmap("data/samples/corpus.txt", "runs/nano/tokens.bin", tok)
dataset = PackedTokenDataset(path, seq_length=512)
for batch in dataset.iter_batches(batch_size=16, seed=42):
    print(batch["input_ids"].shape, batch["labels"].shape)
    break
```

Run through the CLI with checked-in configs:

```powershell
kumaru train configs\train\lora-qwen.yaml --dry-run
kumaru train configs\train\pretrain-nano.yaml --dry-run
```

## Tuning / operational notes

For LoRA, data choices often matter more than rank. Keep `data.train_on_completions_only: true`; turning it off makes user and system tokens consume gradient budget. Use `data.max_seq_length: 2048` for the 7B QLoRA starter and `1024` for the 0.5B smoke run.

Overlong SFT examples are dropped, not truncated. If more than 10% are dropped, the logger warns. Dropping protects answer endings and stop tokens; truncating mid-answer teaches incomplete completions.

For pretraining, packing is a throughput feature. Concatenating documents and slicing fixed windows removes padding, so every token in every batch contributes loss. The code uses `uint32` because Qwen tokenizers have vocabularies larger than 65,536; `uint16` would corrupt token ids.

On a 24 GB RTX 3090:

- `lora-qwen.yaml`: about 9-11 GB VRAM with 4-bit base, batch 1, accumulation 16, checkpointing; about 20-40 minutes per epoch for about 1,000 examples of about 1k tokens.
- `lora-small.yaml`: about 2 GB VRAM, a few minutes, good for catching data and masking problems.
- `pretrain-nano.yaml`: sample corpus overfits quickly; a properly sized 35M model wants about 700M tokens.

If you hit OOM or data collapse, change these in order:

1. Lower `data.max_seq_length` only after checking whether examples are genuinely too long.
2. Keep `optim.per_device_batch_size` small and raise accumulation instead.
3. Use QLoRA (`load_in_4bit: true`) for 7B.
4. Turn on gradient checkpointing in the training config.
5. Shorten or split long conversations rather than letting all rows drop.

## How to extend it

To support a new dataset format, extend `load_chat_jsonl()` by converting the new shape into the canonical `{"messages": [...]}` form before validation. For example, a prompt/completion dataset:

```python
if "prompt" in obj and "completion" in obj:
    obj = {
        "messages": [
            {"role": "user", "content": obj["prompt"]},
            {"role": "assistant", "content": obj["completion"]},
        ]
    }
```

Do not bypass `encode_conversation()`. New formats should still use the tokenizer's chat template and completion-only labels.

To add a pretraining document separator, append or tokenize a separator between chunks in `pack_corpus_to_memmap()`. Keep the output dtype `np.uint32` unless every supported tokenizer is guaranteed below 65,536 ids.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `has no non-empty 'messages' list` | JSONL row is not in chat or supported Alpaca shape | Convert rows to `messages`, or add a converter in `load_chat_jsonl()`. |
| `Every example was dropped during encoding` | Rendered conversations exceed `max_seq_length` or have no supervised assistant tokens | Increase `data.max_seq_length`, shorten rows, and inspect roles. |
| Loss drops but model predicts prompts | `train_on_completions_only` is false or labels include user text | Keep completion-only masking; inspect `example_preview.txt`. |
| Adapter learns not to stop | Assistant end-of-turn token not supervised due to broken template behavior | Use an instruct tokenizer with a valid chat template. |
| Pretraining raises `too few for seq_length` | Tokenized corpus has fewer than `seq_length + 1` tokens | Use a larger `text_file` or smaller `model.max_seq_length`. |
| Quality is mysteriously bad after packing | Token ids corrupted by an incorrect dtype in custom code | Keep `uint32` memmaps for modern tokenizers. |

## Related files

- `src\kumaru\train\finetune_lora.py` — calls `load_chat_jsonl`, `split_rows`, `build_sft_dataset`, and `SFTCollator`.
- `src\kumaru\train\pretrain_nano.py` — calls `pack_corpus_to_memmap` and `PackedTokenDataset`.
- `src\kumaru\train\config.py` — defines `data.*` fields that control this module.
- `configs\train\lora-qwen.yaml` — uses completion-only masking and 2048-token examples.
- `configs\train\pretrain-nano.yaml` — uses `text_file` and Qwen tokenizer for packing.
