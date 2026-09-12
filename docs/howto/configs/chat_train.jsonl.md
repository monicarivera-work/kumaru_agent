# `data/samples/chat_train.jsonl`

> Sample supervised fine-tuning dataset in JSON Lines chat format.

**Read this when:** you are building an SFT dataset, debugging `kumaru train ... --dry-run`, or deciding whether an example should train the assistant response only.

---

## What it does

`data/samples/chat_train.jsonl` is a small set of chat examples for LoRA training configs. Each nonblank line is one complete JSON object. The sample lines use:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

`src/kumaru/train/data.py` reads this file with `load_chat_jsonl()`, validates every line, skips examples with no non-empty assistant message, and reports line numbers for malformed JSON or missing fields.

## When to use this profile

Use the sample file only to prove the training path works. Build your own dataset for any real adapter. Good starting sizes:

| Goal | Examples |
|---|---:|
| Style, tone, or output format | 100-1,000 high-quality examples |
| New task behavior or domain conventions | 1,000-10,000 examples |
| Under 50 examples | Usually memorization; use very low rank and inspect eval loss |

## Expected data format

### Chat format

Each line must be valid JSON and contain a non-empty `messages` list:

```json
{"messages":[
  {"role":"system","content":"Optional system instruction."},
  {"role":"user","content":"User request."},
  {"role":"assistant","content":"Ideal assistant answer."}
]}
```

Rules enforced by the loader:

| Requirement | Why it matters |
|---|---|
| One JSON object per line | JSONL streams and reports errors by line number. |
| Top-level `messages` is a non-empty list | Training examples are conversations. |
| Each message is an object with `role` and `content` | `encode_conversation()` passes these through the tokenizer chat template. |
| At least one assistant message has nonblank content | Completion-only masking needs assistant tokens to supervise. |
| Blank lines are allowed | Loader strips and skips them. |

Roles follow the shared message model: `system`, `user`, `assistant`, and `tool` exist in core types, but SFT examples normally use `system`, `user`, and `assistant`.

### Alpaca alternative

`load_chat_jsonl()` also accepts the common Alpaca shape:

```json
{"instruction":"Write a SQL query.","input":"Table: users(id, email)","output":"SELECT email FROM users;"}
```

It converts that to:

```json
{"messages":[
  {"role":"user","content":"Write a SQL query.\n\nTable: users(id, email)"},
  {"role":"assistant","content":"SELECT email FROM users;"}
]}
```

If `input` is empty or missing, the user content is just `instruction`.

## Every field explained

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `messages` | list[object] | required | Conversation turns for one training example. | Include the minimum turns needed to define the desired answer. |
| `messages[].role` | string | required | Speaker for a turn. | Use `system` for policy/context, `user` for prompt, `assistant` for the target answer. |
| `messages[].content` | string | required | Text for that turn. | Make assistant content exactly what you want the model to learn. |
| `instruction` | string | accepted alternative | Alpaca user instruction. | Use only when importing Alpaca-style data. |
| `input` | string | optional alternative | Additional Alpaca input appended after a blank line. | Include source text, table schema, or context. |
| `output` | string | accepted alternative | Alpaca assistant answer. | Must be non-empty to contribute gradient. |

## How to build your own dataset

1. Decide the behavior you want: format, tone, task process, or domain response.
2. Write examples whose assistant answer is production-quality.
3. Keep formatting consistent. The model learns inconsistent punctuation, headings, and refusal style.
4. Hold out a validation set with `data.eval_file`, or use `data.eval_split`.
5. Avoid examples with answers that stop mid-thought; long examples over `data.max_seq_length` are dropped, not truncated.
6. Keep private or regulated data out unless your local storage and training process are approved for it.

Example line:

```json
{"messages":[{"role":"user","content":"Summarize this incident in one sentence: database failover completed after 7 minutes."},{"role":"assistant","content":"Database failover completed successfully after a 7-minute interruption."}]}
```

## How to use it

Validate with a training config:

```powershell
kumaru train configs/train/lora-small.yaml --dry-run
```

Then train the small smoke adapter:

```powershell
kumaru train configs/train/lora-small.yaml
```

Use your own file by editing the config:

```yaml
data:
  train_file: data/my_chat_train.jsonl
  eval_split: 0.1
```

Then validate:

```powershell
kumaru train configs/train/lora-qwen.yaml --dry-run
```

## Tuning it

1. Fix data quality before changing LoRA hyperparameters.
2. If examples are dropped, shorten them or raise `data.max_seq_length`.
3. If the model learns to copy questions, ensure `train_on_completions_only: true`.
4. If eval loss rises, reduce repeated/near-duplicate rows and lower epochs.
5. If behavior is inconsistent, normalize answer structure before adding more rows.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `is not valid JSON` | A line is not a complete JSON object. | Validate JSONL line-by-line; no trailing commas. |
| `has no non-empty 'messages' list` | Missing or empty `messages`. | Add a proper `messages` array or use Alpaca fields. |
| `message missing 'role' or 'content'` | A turn object is incomplete. | Add both fields to every message. |
| Example is silently skipped with warning | No non-empty assistant turn. | Add an assistant answer. |
| Every example dropped during encoding | Tokenized examples exceed `max_seq_length`. | Shorten examples or raise sequence length. |
| Fine-tune regresses general ability | Data is too small, too repetitive, or rank/LR/epochs too high. | Improve data and tune `lora.r`, LR, and epochs. |

## Related files

- `src/kumaru/train/data.py` — authoritative loader, Alpaca conversion, and loss masking.
- `configs/train/lora-small.yaml` — fast validation profile.
- `configs/train/lora-qwen.yaml` — 7B QLoRA profile.
- `src/kumaru/train/config.py` — fields such as `data.train_file`, `eval_split`, and `max_seq_length`.

