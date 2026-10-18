# `configs/student.yaml`

> Benchmark profile for the distillation student, Qwen2.5-1.5B-Instruct. It decodes greedily, 32 answers per batch, so all 1,319 GSM8K questions take minutes instead of hours.

**Read this when:** you are scoring the student before or after distillation, or benchmarking any small model on the `transformers` backend.

---

## What it does

This profile loads `Qwen/Qwen2.5-1.5B-Instruct` with the `transformers` backend on CUDA in bfloat16 and decodes greedily, with no repetition penalty. Greedy requests that arrive together are decoded as one batch of up to 32 (`backend.max_batch_size`), and `kumaru bench run` sends 32 at a time to fill it.

The same profile scores the model before and after distillation. Only `--adapter` changes between the runs, so any difference between them comes from the adapter.

## When to use this profile

- Scoring the base student, and each trained adapter, on `gsm8k` or any other benchmark.
- The 0.5B pilot: add `-m Qwen/Qwen2.5-0.5B-Instruct`.
- Not for chat. Chat streams its replies, and streamed requests are never batched; greedy decoding with no penalty can also loop on open-ended questions. To chat with a student, use a chat profile: `kumaru chat -c local-gpu -m Qwen/Qwen2.5-1.5B-Instruct --adapter <dir>`.

Everything not listed below comes from [`configs/default.yaml`](default.yaml.md).

## Every field explained

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `backend.name` | string | `transformers` | Loads the PyTorch weights in-process. | Keep. Only this backend loads a LoRA adapter (`--adapter`) or batches greedy requests. |
| `backend.model` | string | `Qwen/Qwen2.5-1.5B-Instruct` | The student. | Must be the `base_model` the adapter was trained on; a test keeps it equal to the distill configs. `-m` overrides it for the 0.5B pilot. |
| `backend.device` | string | `cuda` | Where the model runs. | Keep. |
| `backend.dtype` | string | `bfloat16` | Weight and compute precision. | `float32` makes batched answers match one-at-a-time ones far more often, at twice the memory (below). |
| `backend.max_context_length` | int | `4096` | Prompt plus answer cap. | A GSM8K row needs at most ~1,300 tokens: a prompt of up to ~280 plus an answer of up to 1,024. |
| `backend.max_batch_size` | int | `32` | Greedy requests decoded together; 1 turns batching off. | The KV cache costs 28 KB per token per row, so 32 rows × ~1,200 tokens ≈ 1.1 GB. Lower it if memory is short. |
| `generation.max_tokens` | int | `1024` | The reply cap when a command sets none, as in chat. | `kumaru bench run` sets its own, 8,192 unless `--max-tokens` says otherwise, so pass `--max-tokens 1024` there. |
| `generation.temperature` | float | `0.0` | Greedy decoding. | Keep: deterministic, and the only mode that batches. |
| `generation.top_p` | float | `1.0` | Nucleus sampling cutoff. | No effect while greedy. |
| `generation.repetition_penalty` | float | `1.0` | No penalty. | Keep: it scores the model as trained. A penalty also discourages tokens already in the question, such as the numbers a math answer must copy. `default.yaml` uses 1.05, and the checkpoint's own setting is 1.1. |

**VRAM, estimated:** the weights take 1.5B × 2 bytes ≈ 3.1 GB, and the KV cache ≈ 1.1 GB. About 5 GB in all, so it fits beside other work.

**Batching, measured** on Qwen2.5-0.5B with 16 GSM8K questions:

| Precision | Speed-up at batch 8 | Answers identical to one-at-a-time | Accuracy |
|---|---|---|---|
| bfloat16 | 4.6× | 8 of 16 | unchanged |
| float32 | 6.3× | 16 of 16 | unchanged |

In bfloat16, a different batch shape can round a matrix product differently, and where two tokens are nearly tied the answer takes another path. Accuracy did not move in these tests. float32 rounds too, but 65,000 times more finely, so a flip is rare; only `max_batch_size: 1` matches one-at-a-time decoding exactly.

## How to use it

```powershell
kumaru bench run gsm8k -c student --full --max-tokens 1024 --label student-base -o runs\distill\eval-base
kumaru bench run gsm8k -c student --full --max-tokens 1024 --adapter runs\distill\qwen1.5b-teacher\adapter --label student-teacher -o runs\distill\eval-teacher
kumaru bench compare runs\distill\eval-base runs\distill\eval-teacher -e 5
```

- **The first run downloads the model**, about 3 GB, from Hugging Face.
- **`--label` names the run** in the comparison table. Without it, every run of this profile is called `student-default`, adapter or not.
- **`--full`** scores all 1,319 questions. With the default 100, only a difference of about 10 points is reliably detectable; with 1,319, about 3.

Closer to one-at-a-time decoding, at twice the memory:

```powershell
$env:KUMARU_BACKEND__DTYPE = "float32"
```

Inspect the resolved settings:

```powershell
kumaru config -c student
```

## Tuning it

1. **Out of memory:** lower the batch with `-j 8` (questions in flight), or set `$env:KUMARU_BACKEND__MAX_BATCH_SIZE = "8"`.
2. **Another student model:** change `backend.model` and the `base_model` of the distill configs together. A test fails if they differ.
3. **Comparing runs:** keep every flag equal except `--adapter`, `--label` and `-o`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Slow, with a high **cut off** count | No `--max-tokens`: a looping answer holds its whole batch for up to 8,192 tokens. | Pass `--max-tokens 1024`; read the cut-off answers with `kumaru bench report <dir>`. |
| Size-mismatch error while loading the adapter | The adapter was trained on another base model (for example the 0.5B pilot). | Pass the matching `-m`; the base is `base_model_name_or_path` in the adapter's `adapter_config.json`. |
| CUDA out of memory | Another program (a game, OBS) holds GPU memory. | Lower the batch (Tuning, step 1). |
| Two runs of one model differ by a few answers | bfloat16 batching, near-tied tokens. | Use float32, or treat it as noise: `bench compare` measures it. |
| Results differ from a run made with v0.2.0 | Greedy decoding now uses the profile's `repetition_penalty` instead of the checkpoint's 1.1, and stops on every end token. | Re-run the baseline into a new folder. See [`transformers_backend.md`](../src/kumaru/backends/transformers_backend.md#changed-since-v020). |
| `The llamacpp_server backend cannot attach a LoRA adapter` | `-b llamacpp_server` or a GGUF profile was used. | Use this profile. |

## Related files

- `configs/train/distill-gsm8k.yaml`, `distill-gsm8k-gold.yaml`, `distill-gsm8k-0.5b.yaml` — the recipes whose adapters this profile scores ([`distill-gsm8k.yaml.md`](distill-gsm8k.yaml.md)).
- `src/kumaru/backends/transformers_backend.py` — greedy batching ([`transformers_backend.md`](../src/kumaru/backends/transformers_backend.md)).
- `src/kumaru/eval/benchmarks/runner.py` — `default_concurrency` sends `max_batch_size` questions at once ([`runner.md`](../src/kumaru/eval/benchmarks/runner.md)).
- `src/kumaru/train/distill.py` — the experiment, end to end ([`distill.md`](../src/kumaru/train/distill.md)).
- `configs/default.yaml` — every field this profile leaves out.
