# `data/samples/corpus.txt`

> Tiny UTF-8 raw-text corpus used by the nano pretraining profile to exercise token packing and from-scratch training.

**Read this when:** you are replacing the demonstration corpus, validating pretraining data, or debugging `PackedTokenDataset` behavior.

---

## What it does

`data/samples/corpus.txt` is plain UTF-8 text. `configs/train/pretrain-nano.yaml` points `data.text_file` at it. On the first run with a given `output_dir`, the packer reads the file 1,000,000 characters at a time, tokenizes each chunk, and appends the token ids to `runs/nano/tokens.bin`, a flat `uint32` file. Training reads that file as a memory map and cuts it into windows of `model.max_seq_length` tokens. Later runs with the same `output_dir` reuse `tokens.bin` and do not read the text again.

The file is deliberately too small to produce a useful model: about 1,700 tokens with the profile's Qwen tokenizer, which is six 256-token windows. It exists so `kumaru train configs/train/pretrain-nano.yaml` can run end to end immediately.

## When to use this profile

Use this file for smoke tests and to observe overfitting. Replace it for any real pretraining experiment. The config comments estimate that its ~101M-parameter model wants roughly 2B tokens (~8 GB of text) for serious Chinchilla-style training; this sample is only a few kilobytes.

## Expected data format

The pretraining corpus is not JSONL. It is one raw text file:

| Requirement | Details |
|---|---|
| Encoding | UTF-8; loader opens with `encoding="utf-8", errors="replace"`. |
| Size | Any size. Packing streams: each 1,000,000-character chunk is tokenized and written to `tokens.bin.tmp` before the next is read, so RAM stays at about 0.2 GB with the Qwen tokenizer however large the file is. The finished file is renamed to `tokens.bin`. On disk it takes 4 bytes per token (about 8 GB for 2B tokens). The corpus needs at least `model.max_seq_length + 1` tokens. |
| Structure | Free-form text. Paragraphs, headings, and newlines are preserved as text. |
| Tokenization | `data.tokenizer` from the training config tokenizes chunks with `add_special_tokens=False`. |
| Packing | All chunks form one token stream. At load time `PackedTokenDataset` cuts it into windows of `model.max_seq_length + 1` tokens; consecutive windows share one token, and any tail too short for a window is unused. |
| Labels | For each window, labels are inputs shifted by one token. |

## Every field explained

For this data file, the "fields" are file-level properties rather than YAML keys:

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| Text bytes | UTF-8 text | sample explanatory prose | Raw language-model training stream. | Use representative, legally usable, deduplicated text. |
| Newlines/paragraphs | text formatting | preserved | Gives the tokenizer natural boundaries and document structure. | Preserve meaningful formatting; avoid generated separators unless wanted. |
| Corpus size | file length | tiny sample | Number of tokens available for windows. | Use enough text for at least many windows; serious runs need orders of magnitude more. |
| Character quality | text content | clean technical prose | Model learns spelling, grammar, topics, and artifacts. | Remove boilerplate, corrupt encodings, repeated pages, and logs unless those are target data. |

The YAML fields that consume this file live in `configs/train/pretrain-nano.yaml`: `data.text_file` (or `data.train_file` when `text_file` is unset), `data.tokenizer`, and `model.max_seq_length`, plus `output_dir`, which decides where `tokens.bin` is written and reused. Pretraining ignores `data.max_seq_length`, which only applies to LoRA fine-tuning.

## How to build your own corpus

1. Gather text you are allowed to use.
2. Convert it to UTF-8.
3. Deduplicate aggressively; repeated text teaches memorization.
4. Remove navigation, boilerplate, binary dumps, broken markup, and secrets.
5. Keep document separators only if you want the model to learn them.
6. Save as a normal text file in the repository or a known absolute path.
7. Point `data.text_file` at it and set a new `output_dir`.

Example config change (only the changed keys):

```yaml
output_dir: runs/nano-my-corpus   # a new folder, so this corpus is packed fresh
data:
  train_file: data/my_corpus.txt
  text_file: data/my_corpus.txt
  tokenizer: Qwen/Qwen2.5-0.5B-Instruct
model:
  # ...other model keys unchanged...
  max_seq_length: 512             # the packed window length
```

## How to use it

Validate packing and config:

```powershell
kumaru train configs/train/pretrain-nano.yaml --dry-run
```

Run the toy pretraining loop:

```powershell
kumaru train configs/train/pretrain-nano.yaml
```

For a larger corpus, edit `data.text_file` and `output_dir`, then dry-run again. The dry run packs the corpus and stops. Packing is the slow part on a big file: it ran at about 1.4 MB of text per second in local testing, about 1.5 hours for 8 GB, and it happens once per `output_dir`. Pack this way before any `torchrun` launch, whose other ranks stop waiting for rank 0 after PyTorch's process-group timeout:

```powershell
kumaru train configs/train/pretrain-nano.yaml --dry-run
```

## Tuning it

1. Increase corpus size before increasing model size.
2. If the corpus is too small for one window, lower `model.max_seq_length`; pretraining ignores `data.max_seq_length`. No repack is needed, because `tokens.bin` is cut into windows when training starts.
3. If OOM occurs, lower `optim.per_device_batch_size` first.
4. If the model memorizes quickly, the corpus is too small or too repetitive.
5. Keep tokenizer choice stable across experiments if comparing losses.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Training data not found: ... (data.text_file)` | `data.text_file` points to a missing path; `kumaru train` checks it before loading the tokenizer. | Correct the path in the training config. Relative paths resolve from the directory you run in. |
| `Corpus not found` | The same missing path, reached through the `torchrun` entry point, which skips that check. | Correct the path; dry-run with `kumaru train` first. |
| `produced zero tokens` | Empty file or tokenizer issue. | Add text and verify tokenizer loads. |
| `too few for seq_length` | Not enough packed tokens for one training window. | Add more text or lower `model.max_seq_length`. |
| Old text still trains after editing `data.text_file` | `tokens.bin` in that `output_dir` was reused. | Set a new `output_dir`, or delete `tokens.bin` to repack. |
| `tokens.bin.tmp` left in `output_dir` | Packing was killed before it finished. | Harmless: the next run packs again and overwrites it. Delete it to reclaim the disk space. |
| Loss drops to near zero | Corpus is tiny and being memorized. | Use a much larger held-out corpus. |
| Bad model learns garbage | Corpus contains boilerplate, corruption, duplicates, or secrets. | Clean and deduplicate before training. |

## Related files

- `configs/train/pretrain-nano.yaml` — pretraining config that consumes this file.
- `src/kumaru/train/data.py` — `pack_corpus_to_memmap()` and `PackedTokenDataset`.
- `src/kumaru/train/config.py` — `DataConfig` and `NanoModelConfig` fields.

