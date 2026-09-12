# `data/samples/corpus.txt`

> Tiny UTF-8 raw-text corpus used by the nano pretraining profile to exercise token packing and from-scratch training.

**Read this when:** you are replacing the demonstration corpus, validating pretraining data, or debugging `PackedTokenDataset` behavior.

---

## What it does

`data/samples/corpus.txt` is plain UTF-8 text. `configs/train/pretrain-nano.yaml` points `data.text_file` at it. The pretraining loader tokenizes the entire text into a flat `uint32` memory-mapped token array, then trains on fixed-length windows.

The file is deliberately too small to produce a useful model. It exists so `kumaru train configs/train/pretrain-nano.yaml` can run end to end immediately.

## When to use this profile

Use this file for smoke tests and to observe overfitting. Replace it for any real pretraining experiment. The config comments estimate that a 35M-parameter model wants roughly 700M tokens for serious Chinchilla-style training; this sample is only a few kilobytes.

## Expected data format

The pretraining corpus is not JSONL. It is one raw text file:

| Requirement | Details |
|---|---|
| Encoding | UTF-8; loader opens with `encoding="utf-8", errors="replace"`. |
| Size | Any size that produces enough tokens; packing streams chunks and writes a memmap. |
| Structure | Free-form text. Paragraphs, headings, and newlines are preserved as text. |
| Tokenization | `data.tokenizer` from the training config tokenizes chunks with `add_special_tokens=False`. |
| Packing | Tokens are concatenated and sliced into fixed windows of `seq_length + 1`. |
| Labels | For each window, labels are inputs shifted by one token. |

## Every field explained

For this data file, the "fields" are file-level properties rather than YAML keys:

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| Text bytes | UTF-8 text | sample explanatory prose | Raw language-model training stream. | Use representative, legally usable, deduplicated text. |
| Newlines/paragraphs | text formatting | preserved | Gives the tokenizer natural boundaries and document structure. | Preserve meaningful formatting; avoid generated separators unless wanted. |
| Corpus size | file length | tiny sample | Number of tokens available for windows. | Use enough text for at least many windows; serious runs need orders of magnitude more. |
| Character quality | text content | clean technical prose | Model learns spelling, grammar, topics, and artifacts. | Remove boilerplate, corrupt encodings, repeated pages, and logs unless those are target data. |

The YAML fields that consume this file live in `configs/train/pretrain-nano.yaml`: `data.text_file`, `data.tokenizer`, `data.max_seq_length`, and `model.max_seq_length`.

## How to build your own corpus

1. Gather text you are allowed to use.
2. Convert it to UTF-8.
3. Deduplicate aggressively; repeated text teaches memorization.
4. Remove navigation, boilerplate, binary dumps, broken markup, and secrets.
5. Keep document separators only if you want the model to learn them.
6. Save as a normal text file in the repository or a known absolute path.
7. Point `data.text_file` at it.

Example config change:

```yaml
data:
  train_file: data/my_corpus.txt
  text_file: data/my_corpus.txt
  tokenizer: Qwen/Qwen2.5-0.5B-Instruct
  max_seq_length: 512
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

For a larger corpus, edit `data.text_file`, then dry-run again:

```powershell
kumaru train configs/train/pretrain-nano.yaml --dry-run
```

## Tuning it

1. Increase corpus size before increasing model size.
2. If `seq_length` is too large for the corpus, lower `data.max_seq_length` and `model.max_seq_length` together.
3. If OOM occurs, lower `optim.per_device_batch_size` first.
4. If the model memorizes quickly, the corpus is too small or too repetitive.
5. Keep tokenizer choice stable across experiments if comparing losses.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Corpus not found` | `data.text_file` points to a missing path. | Correct the path in the training config. |
| `produced zero tokens` | Empty file or tokenizer issue. | Add text and verify tokenizer loads. |
| `too few for seq_length` | Not enough packed tokens for one training window. | Add more text or lower sequence length. |
| Loss drops to near zero | Corpus is tiny and being memorized. | Use a much larger held-out corpus. |
| Bad model learns garbage | Corpus contains boilerplate, corruption, duplicates, or secrets. | Clean and deduplicate before training. |

## Related files

- `configs/train/pretrain-nano.yaml` — pretraining config that consumes this file.
- `src/kumaru/train/data.py` — `pack_corpus_to_memmap()` and `PackedTokenDataset`.
- `src/kumaru/train/config.py` — `DataConfig` and `NanoModelConfig` fields.

