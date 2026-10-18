# `src\kumaru\eval\benchmarks\runner.py`

> Runs benchmarks against one loaded model, writes every graded answer to disk as it arrives, resumes interrupted runs, and renders reports and paired A/B comparisons. Behind `kumaru bench run|report|compare`.

**Read this when:** you are running a benchmark, resuming one, reading a results directory, comparing two runs, or wondering why a run used a given label, concurrency or token budget.

---

## What it does

- **`run_benchmarks` / `BenchRunner`**:
  - resolves benchmark names or golden-set paths;
  - samples the questions and loads the model once;
  - asks each question with chat's decoding settings and a per-question seed, `-j` at a time;
  - grades each answer the moment it arrives and appends it to `items.jsonl`.

  It also writes `summary.json` per benchmark and `run.json` per run.
- **Resume.** Point `-o/--out` at an existing run. Only questions with no valid answer are asked, and a run with different settings is refused before anything is written.
- **`compare_runs`.** Pairs two runs question by question. It gives a bootstrap CI for B − A, an exact McNemar p and a Holm correction across benchmarks, and shows which questions flipped.
- **`render_summaries`, `render_comparisons`, `render_flips`.** The tables the CLI prints.

## Why it exists

Three properties make a benchmark number usable, and this module guarantees them:

1. **Reproducible.** The same `--seed` and `-n` give the same questions, and each question gets the same sampling seed in every run (`item_seed`). Settings, resolved config, backend, hardware and git commit are recorded next to the results.
2. **Auditable.** A score says *that* something changed; the records say *what*. Every answer is kept with its prompt, reasoning, extracted answer, verdict, tokens and latency, so a surprising number can be read question by question.
3. **Cheap to interrupt.** A local 35B run takes a long time. A crash, Ctrl+C or a dead server loses only the questions in flight, and re-running the same command with `-o <run dir>` continues where it stopped.

## Mental model

### 1. What one run does

1. **Resolve.** Each argument is either a registry name or a golden-set file. A path ending in `.jsonl`, or an existing file that isn't a registered name, becomes `GoldenSet(path)`.
   - The same benchmark given twice runs once, so `mmlu_pro mmlu-pro` runs once.
   - Two *different* benchmarks with one name are refused, because their results would share one folder. Golden sets `a\set.jsonl` and `b\set.jsonl` are both named `golden-set`, so `kumaru bench run a\set.jsonl b\set.jsonl` stops with `Two benchmarks share the name 'golden-set' (the second is b\set.jsonl), so their results would share one folder.` and the hint `Rename one of the files: a golden set is named after its file.` This happens before the model loads, so no run directory is created.
   - A benchmark that executes code is refused without `--allow-code-exec`.
   - Each benchmark's `preflight()` runs.
2. **Load and sample** every benchmark's questions (`select_items`, see [`base.md`](base.md#what--n--limit-picks-select_items)). This happens before the model loads, so a typo or missing package fails in seconds.
3. **Load the backend once** (`loading <model>...`). Resolve decoding with `ChatAgent.resolve_params(thinking=, thinking_budget=, temperature=)`. This is exactly how `kumaru chat` decodes: `generation` < `thinking_mode` (only when thinking is on) < flags.
4. **Name the run** (`default_label`, below). The directory is `--out`, or else `<repo>\runs\bench\<YYYYmmdd-HHMMSS>-<label>`. The default is anchored at the repo root, not the current directory.
5. **Check every benchmark against existing results** (the fingerprint, below). This happens before anything is written, so a refused resume leaves the old run untouched.
6. Create the directory, print `results: <dir>` and write `run.json`.
7. **Run the benchmarks one after another.** Within a benchmark, up to `concurrency` questions are in flight, and each is graded and appended as soon as it returns. `summary.json` is written before the first question and again when the benchmark ends or stops.

### 2. The run directory

```
runs\bench\20261010-114917-qwen25-7b-gguf-default\
    run.json                         how the run was made (first session)
    run.resumed-<YYYYmmdd-HHMMSS>.json   one per resumed session, same shape
    golden\items.jsonl               one JSON record per answered question, append-only
    golden\summary.json              BenchSummary: accuracy, CI, metrics, provenance
    gsm8k\items.jsonl
    gsm8k\summary.json
    ...
```

- The sub-directory is `safe_name(benchmark name)`, for example `golden`, `golden-support` or `mmlu_pro`.
- `*.tmp` files exist only for an instant. `summary.json` and `run.json` are written to a `.tmp` and then `os.replace`d, so a hard kill never leaves half a summary behind.

**`items.jsonl`** holds one `ItemRecord` per line. A real line (golden, Qwen2.5-7B):

```json
{"id":"golden/mult","key":"967daaf02648d506","group":"math","prompt":"What is 17 × 24? Finish with \"Answer: <number>\".","correct":true,"score":1.0,"extracted":"408","expected":408,"output":"Answer: 408","reasoning":null,"finish_reason":"stop","prompt_tokens":47,"completion_tokens":7,"latency_ms":189.6,"seed":599856693,"error":null,"detail":{"method":"answer_tag"}}
```

| Field | Meaning |
|---|---|
| `id`, `group`, `prompt` | The question. `prompt` is the user message only; the system message is not stored. |
| `key` | Hash of the fingerprint, the benchmark's messages, `max_tokens` and seed. With `--use-agent-prompt` the messages are hashed before the agent prompt is added (see [Resume](#6-resume-fingerprint-vs-record-key)). A resume reuses the record only if this matches. |
| `correct`, `score` | The verdict. `score` is partial credit, or `float(correct)`. |
| `extracted`, `expected`, `detail` | What the grader read, the gold answer, and grader details such as `method`, `status`, or IFEval's `strict`/`loose` lists. |
| `output`, `reasoning` | The answer text that was graded, and the thinking text. Reasoning is kept but never graded. |
| `finish_reason` | `"stop"`, `"length"` (cut off by the budget; graded as it stands, usually wrong) or the backend's value. |
| `prompt_tokens`, `completion_tokens`, `latency_ms`, `seed` | Cost and the per-question seed. |
| `error` | Set when the request or the grader failed. The answer counts as wrong, and a resume retries it. |

**`summary.json`** holds a `BenchSummary`. The fields that are easy to misread:

| Field | Meaning |
|---|---|
| `n`, `n_planned`, `complete` | Answered (including failed) / sampled questions. `complete` means `n == n_planned`. |
| `accuracy`, `ci95` | `n_correct / n` with a Wilson 95% interval. Errors count as wrong. |
| `metrics`, `by_group`, `extraction` | `aggregate()` output (e.g. IFEval's four scores), accuracy per group, and the counts behind the report's `answers found by:` or `tests:` line. `extraction` counts `error` for a failed question, else `detail["method"]`, else `detail["status"]` (HumanEval's `passed`/`failed`/`timeout`). Whole-reply graders (`contains`, `regex`, IFEval) add nothing; the report shows the remainder as `whole reply`. |
| `n_errors`, `n_truncated` | Failed questions, and answers that hit the token budget (`finish_reason == "length"`). |
| `mean_completion_tokens`, `median_latency_ms` | Over error-free answers only. |
| `wall_seconds`, `items_per_minute` | Accumulated over all sessions of the run. |
| `params` | Resolved decoding for this benchmark. `max_tokens` is the largest per-question budget actually sent (8,192 with thinking off). |
| `fingerprint`, `items_sha256`, `item_ids` | The resume check, a digest of `[id, system, prompt]` of the sampled questions, and the sample itself. |
| `started_at`, `git_commit`, `kumaru_version` | First session's start (UTC). Commit and version of the latest session. |

> `summary.json` is **not** updated after every answer. As the module docstring says, it is written when a benchmark starts and again when it stops: finished, failed or stopped (including Ctrl+C and the five-errors stop). `items.jsonl` is the source of truth.
>
> - **During a run, or after a hard kill,** `bench report` shows the state at the start of that session: `0/100 (partial)` for a fresh run. After a hard kill it stays stale until a resume rewrites it from `items.jsonl`.
> - **`bench compare` reads `items.jsonl`** (restricted to `item_ids`), so it sees every answer saved so far.

### 3. Labels: `default_label`

`<base>-<mode>`, passed through `safe_name`:

- `base` is the profile's file stem when `-c` is given: `-c qwen36` and `-c configs\qwen36.yaml` both give `qwen36`. Otherwise it is `model_name(model id)`: the file or repo name, taken after the last `\` or `/`, cut at `:` and stripped of a `.gguf`, `.safetensors` or `.bin` extension:

  | Model id | `model_name` |
  |---|---|
  | `Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` | `Qwen2.5-7B-Instruct-GGUF` |
  | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS` | `Qwen3.6-35B-A3B-GGUF` |
  | `D:\kumaru\models\Qwen3-8B-Q4_K_M.gguf` | `Qwen3-8B-Q4_K_M` |
  | `Qwen/Qwen2.5-7B-Instruct` | `Qwen2.5-7B-Instruct` (only weight-file extensions are stripped) |
- `mode` comes from the resolved thinking setting:

| Resolved thinking | `mode` | Example label |
|---|---|---|
| on, no budget | `think` | `qwen36-think` |
| on, budget B | `think<B>` | `qwen36-think8192`, `qwen36-think0` |
| off (explicitly `false`) | `nothink` | `qwen36-nothink` (qwen36's profile default) |
| `None` (the model/template decides) | `default` | `qwen25-7b-gguf-default`, `test-default` |

`--label` replaces the whole label. The directory name always gets the timestamp prefix unless you pass `-o`.

### 4. The answer budget: `_max_tokens`

Every benchmark's default answer budget is `Benchmark.answer_tokens = 8192`. The reason: a cut-off answer is graded as it stands and usually fails, and tighter caps were measured cutting 3–16% of answers mid-reasoning (see [`base.md`](base.md#the-answer-budget-one-default-8192-tokens)). Per question:

| Situation | Budget sent as `max_tokens` |
|---|---|
| `--max-tokens N` | `N`, regardless of thinking. It is also part of the fingerprint. |
| Thinking off or default | `item.max_tokens` (golden rows can set it) or **8,192** |
| Thinking on, budget `B` (`--thinking-budget`, or the profile's) | that + `B`, e.g. 8,192 + 8,192 = 16,384 |
| Thinking on, no budget | the larger of that and the profile's resolved `max_tokens` (qwen36's `thinking_mode`: 32,768) |
| Always | capped at `MAX_OUTPUT_TOKENS = 131_072` |

- **`--thinking-budget` alone does not turn thinking on.** If the profile has `generation.thinking: false` (qwen36), add `--think`.
- **`n_truncated`, the "cut off" column, is how you spot a binding cap.** If it is not zero, the report prints `Some answers hit the token limit (cut off) and were graded as they stood, usually wrong; raise --max-tokens or --thinking-budget.` Nothing forces a cut answer to be wrong: `_ask` grades the text as it stands. It usually fails, because it lacks the final `Answer:` line, the closing code fence or the required ending. But not always: a fallback rule (`last_number`, `last_letter`), a `contains` grader or an IFEval check can pass on the fragment. On Qwen3.6's first run (1,024–2,048-token caps), 12 of the 16 cut MMLU-Pro answers were graded wrong; the other 4, and the one cut golden answer, passed on text found mid-reasoning.
- `run.json`'s `params.max_tokens` is the profile's chat value, not the budget the benchmark sent. Read the budget from each `summary.json`'s `params.max_tokens`.

### 5. Concurrency: `default_concurrency`

Without `-j`:

- **`transformers`:** for a greedy run (temperature 0), `backend.max_batch_size`, so that a batch fills. Otherwise 1: the backend batches only greedy requests ([`transformers_backend.md`](../../backends/transformers_backend.md)).
- **Any other backend except `llamacpp_server`:** 1. These serve one request at a time, so sending more only queues them, and a queued request can hit the read timeout.
- **`llamacpp_server`:** `slots` is the server's reported `total_slots`, or `llama_server.parallel` (default 4).
  - With `kv_unified: false`, the result is `slots`.
  - With `kv_unified: true` (the default), all slots share one context window. The result is then `max(1, min(slots, context_length // worst_case))`, where `worst_case` is the largest prompt + answer budget among the questions still to ask, + 256 (`PROMPT_MARGIN_TOKENS`).
  - For example, with 8,192-token budgets and a 32,768 context (qwen25-7b-gguf), 32,768 // (8,192 + prompt + 256) is 3 for prompts under about 2,400 tokens. With 131,072 (qwen36) the result stays at the 4 slots.

The progress line shows the value in use: `acc 92.0%  j=3`.

### 6. Resume: fingerprint vs record key

- **Fingerprint** (per benchmark, checked against `summary.json`). It hashes:
  - the benchmark name, backend, `backend.model`, `model_path`, `adapter_path`;
  - the resolved decoding parameters except `seed` and `max_tokens` (so thinking, thinking budget, temperature, top-p/k, min-p, penalties);
  - `--max-tokens`, `--seed`;
  - with `--use-agent-prompt`, the agent settings that shape the prepended prompt: `system_prompt`, `enabled_tools`, `max_history_tokens` and `memory_policy`. Without the flag this entry is empty, so runs without it keep the fingerprints and keys of older runs, which still resume. A run saved with the flag before these settings joined the fingerprint is refused; start it afresh.

  A mismatch refuses the run: `<run>\<bench> holds results from different settings (model, decoding parameters, seed or prompt mode).`

  `-n`, `-j`, `--context`, `--label`, `backend.max_batch_size` and the dataset contents are **not** in the fingerprint. A bf16 run with batching and one without can therefore share a directory, although a few greedy answers may differ through rounding ([`transformers_backend.md`](../../backends/transformers_backend.md)).
- **Settings, not code.** The fingerprint does not cover Kumaru's code. After changing a grader, a resume keeps every saved verdict from the old grader. After changing a prompt template, every key changes, so a resume re-asks every question into the old directory. In both cases start a new run directory instead of resuming an old one.
- **Record key** (per question). It hashes the fingerprint, the benchmark's messages, that question's `max_tokens` and its seed. A saved record is reused only if `error is None` and the key matches. Otherwise the question is asked again, and the new line is appended. `read_records` keeps the **last** record per id.
  - With `--use-agent-prompt`, the key hashes the bare question, **before** `ChatAgent.prepare` adds the agent prompt. That prompt states today's date, so hashing it would make a run resumed the next day ask every question again. The settings that shape the prompt are in the fingerprint instead.
- **`items.jsonl` is append-only.**
  - A hard kill can leave a torn last line without a newline. `_ends_mid_line` detects it, and the runner writes a `\n` before the first new record so the new record isn't glued onto the fragment and lost with it.
  - Unparseable lines are skipped on read, so that question is simply asked again.
- **What a resume does:**
  - It keeps the original `run.json` and adds `run.resumed-<stamp>.json`.
  - It keeps `started_at` and adds to `wall_seconds`.
  - It retries failed questions.
  - Changing `-n` is allowed. Only the overlap with the new sample is reused (a larger `-n` is not a superset of a smaller one), and the summary then describes the new sample. Records outside it stay in the file but are ignored by `report` and `compare`.
- A run made under the old per-benchmark budgets resumes without complaint, since the fingerprint excludes per-question `max_tokens`. But every key differs, so every question is re-asked. Start a fresh directory instead.

### 7. Provenance: `run.json`

| Key | Content |
|---|---|
| `label`, `benchmarks`, `started_at` | Run name, benchmark names (after dedupe) and UTC start. |
| `kumaru_version`, `git_commit` | `git rev-parse --short HEAD`, plus `+dirty` if tracked files are modified (5 s timeout; `null` without git). |
| `command` | `sys.argv`, verbatim. |
| `settings` | The `BenchSettings` (limit, seed, concurrency, thinking, budgets, flags, label, out_dir). |
| `params` | The resolved base decoding parameters, before per-benchmark overrides and budgets. |
| `backend` | `BackendInfo`, including `extra` (for llama-server: engine build, slots, model file, log file, base URL, pid). |
| `environment` | Python, platform, machine, and `gpus` from `nvidia-smi` (name, total/used memory, driver) when available. |
| `config` | The full resolved `KumaruConfig`, with secrets redacted. |

**Redaction (`_SECRET_KEY`, `_redact`).** A value is replaced by `"<redacted>"` when all three hold:

- its key has a name segment `api_key`/`apikey`/`api-key`, `token`, `secret`, `password`, `passwd`, `authorization` or `credential(s)`, delimited by the start or end of the name or by `_ . -`;
- the value is truthy;
- the value is not a number.

So `server.api_key`, `hf_token`, `client_secret`, `x-api-key` and `Authorization` are hidden, while `max_tokens`, `max_history_tokens` and `tokenizer` are not, and numbers never are.

Runs made before this fix (e.g. `runs\bench\20261010-104148-qwen25-7b-gguf-default\run.json`) show `"max_tokens": "<redacted>"` and `"max_history_tokens": "<redacted>"`. That was over-redaction by an earlier version; the real values are in each `summary.json`'s `params`.

`command` and the backend `extra` are not redacted. Don't put secrets on the command line.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `run_benchmarks` | `async run_benchmarks(cfg: KumaruConfig, specs: Sequence[str], settings: BenchSettings, *, profile: str \| None = None, console: Console \| None = None, show_progress: bool = True) -> tuple[Path, list[BenchSummary]]` | Run and return `(results_dir, summaries)`. `profile` only feeds the label. |
| `BenchRunner` | `BenchRunner(cfg, settings, *, profile=None, console=None, show_progress=True)`; `async run(specs) -> list[BenchSummary]`; `.out_dir` | The class behind `run_benchmarks`. |
| `BenchSettings` | `limit=100, seed=1234, concurrency=None (≥1), thinking=None, thinking_budget=None (≥0), max_tokens=None (1..131072), temperature=None (0..2), use_agent_prompt=False, allow_code_exec=False, label=None, out_dir=None` | How to run. `None` defers to the profile or computed default. Stored in `run.json`. |
| `ItemRecord` / `BenchSummary` / `Comparison` | pydantic models | The on-disk shapes above, and one comparison row. |
| `compare_runs` | `compare_runs(baseline: Path, candidate: Path, *, resamples: int = 10_000) -> list[Comparison]` | Paired comparison on every shared benchmark, Holm-adjusted. |
| `verdict` | `verdict(c: Comparison) -> str` | The verdict column (rules below). |
| `render_summaries` | `render_summaries(console, summaries, *, details: bool = False) -> None` | The run table. `details=True` is `bench report`. |
| `render_comparisons` | `render_comparisons(console, comparisons) -> None` | The compare table and footer. |
| `render_flips` | `render_flips(console, baseline: Path, candidate: Path, limit: int) -> None` | Up to `limit` "only B right" / "only A right" questions per benchmark. |
| `find_results` | `find_results(path: Path) -> dict[str, Path]` | Benchmark name → directory, for a run directory or one benchmark's directory. |
| `load_results` | `load_results(bench_dir: Path) -> tuple[BenchSummary, dict[str, ItemRecord]]` | Summary plus records, restricted to `summary.item_ids`. |
| `load_summaries` | `load_summaries(path: Path) -> list[BenchSummary]` | All summaries of a run. |
| `read_records` | `read_records(path: Path) -> dict[str, ItemRecord]` | Parse `items.jsonl`; last record per id wins; blank and torn lines skipped. |
| `read_summary` | `read_summary(path: Path) -> BenchSummary` | Parse one `summary.json`, or raise `ConfigError`. |
| `summarize` | `summarize(bench, items, records, header, *, wall_seconds: float) -> BenchSummary` | Records → summary. |
| `resolve_benchmark` | `resolve_benchmark(spec: str) -> Benchmark` | Name or golden-file path → instance. |
| `list_benchmarks` | `list_benchmarks() -> list[dict[str, Any]]` | Rows for `bench list`: name, description, source, code_exec. |
| `default_label` | `default_label(model_id: str, profile: str \| None, params: GenerationParams) -> str` | See above. |
| `model_name` | `model_name(model_id: str) -> str` | The file or repo name in a model path or id (the table above). Used by `default_label` when no profile is given. |
| `default_concurrency` | `default_concurrency(cfg, backend, per_request_tokens: int, params: GenerationParams \| None = None) -> int` | See above. `params` decides whether a `transformers` run is greedy. |
| `safe_name` | `safe_name(text: str) -> str` | Runs of characters outside `[A-Za-z0-9._-]` become `-`; leading and trailing `-`/`.` are stripped; empty becomes `run`. |
| `item_seed` | `item_seed(seed: int, item_id: str) -> int` | `crc32(f"{seed}:{item_id}") & 0x7FFFFFFF`. Pinned by a test: old runs pair on it. |
| `record_key` | `record_key(fingerprint: str, messages: Sequence[Message], params: GenerationParams) -> str` | The 16-hex-character resume key. |
| `items_digest` | `items_digest(items: Sequence[BenchItem]) -> str` | `items_sha256`. |
| `git_commit`, `environment` | `() -> str \| None`, `() -> dict` | Provenance. |
| Constants | `DEFAULT_OUT = REPO_ROOT / "runs" / "bench"`, `MAX_CONSECUTIVE_ERRORS = 5`, `TEST_OUTCOMES = frozenset({"passed", "failed", "timeout"})`, `PROMPT_MARGIN_TOKENS = 256`, `MAX_OUTPUT_TOKENS = 131_072` | `TEST_OUTCOMES` are the `extraction` keys that turn the report line into `tests:`. |

## How to use it

### Run

```powershell
kumaru bench run golden gsm8k mmlu_pro ifeval -c qwen36                 # thinking off (qwen36's default), -n 100
kumaru bench run golden gsm8k -c qwen36 --think --thinking-budget 8192  # label qwen36-think8192
kumaru bench run gsm8k -c qwen36 --full -t 0                            # all 1,319 problems, greedy
kumaru bench run humaneval -c qwen36 --allow-code-exec                  # read humaneval.md first
kumaru bench run golden -c test                                         # offline plumbing check (echo)
```

Flags, verified against `kumaru bench run --help`:

| Flag | Default | Effect |
|---|---|---|
| `BENCHMARKS...` | | Names from `kumaru bench list`, or golden-set `.jsonl` paths. |
| `-c/--config`, `-b/--backend`, `-m/--model`, `--adapter`, `--context` | | Same as `kumaru chat`. `--context` ≥ 512. |
| `-n/--limit` | 100 | Questions per benchmark (a fixed, stratified sample). |
| `--full` | off | Every question; overrides `--limit`. |
| `--seed` | 1234 | Picks the questions and each question's sampling seed. |
| `--think/--no-think` | profile | Force reasoning on or off. |
| `--thinking-budget` | profile | Max thinking tokens per question (≥ 0). Does not enable thinking by itself. |
| `--max-tokens` | computed | 1..131072. Output cap per question; replaces the computed budget. The help text gives the default: "Output cap per question (default: 8192; thinking adds --thinking-budget, or uses the profile's max_tokens without one)." |
| `-t/--temperature` | profile | 0..2. `0` = greedy. By default the profile's preset applies: qwen36 uses 0.7, or 1.0 with thinking. |
| `-j/--concurrency` | computed | Questions in flight. |
| `--label` | `<profile or model>-<thinking mode>` | Run name, e.g. `qwen36-think4096`. See [Labels](#3-labels-default_label). |
| `-o/--out` | `runs\bench\<YYYYmmdd-HHMMSS>-<label>` | The run's folder. Point at an earlier run to resume it. |
| `--allow-code-exec` | off | Required for `humaneval`. |
| `--use-agent-prompt` | off | Prepend Kumaru's system prompt via `ChatAgent.prepare`, as chat does. Off by default: public numbers are measured on the raw benchmark prompt. |

When the run finishes it prints the summary table, then `saved to <dir>`, `details: kumaru bench report <dir>` and `compare: kumaru bench compare <baseline-run> <dir>`.

**Sampling.** Each question carries a fixed seed, so two runs at the same temperature draw from the same random stream. They still aren't guaranteed to be identical: llama-server batches concurrent requests, and batch composition can change floating-point results. Two runs of one model differ by a few questions. `compare` (below) quantifies exactly that noise.

### Resume an interrupted run

```powershell
kumaru bench run golden gsm8k mmlu_pro ifeval -c qwen36 -o runs\bench\20261010-120042-qwen36-nothink
```

Use the same flags as the original run. Different decoding flags are refused before anything is written. Ctrl+C prints `Stopped. Finished answers are saved; resume with --out <dir>`.

### Report

```powershell
$env:COLUMNS = '120'
kumaru bench report runs\bench\20261010-114917-qwen25-7b-gguf-default
kumaru bench report runs\bench\20261010-114917-qwen25-7b-gguf-default\gsm8k   # one benchmark
```

Real output, trimmed. It illustrates the format; results belong in [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md):

```
qwen25-7b-gguf-default  Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M · default
┌───────────┬─────┬────────────┬───────────┬────────┬─────────┬───────────────┬─────────────┐
│ benchmark │   n │ accuracy % │    95% CI │ errors │ cut off │ tokens/answer │ answers/min │
├───────────┼─────┼────────────┼───────────┼────────┼─────────┼───────────────┼─────────────┤
│ golden    │  27 │       96.3 │ 81.7-99.3 │      0 │       0 │            86 │       125.2 │
│ gsm8k     │ 100 │       92.0 │ 85.0-95.9 │      0 │       0 │           244 │        47.8 │
│ humaneval │ 100 │       81.0 │ 72.2-87.5 │      0 │       0 │           206 │        55.1 │
│ ifeval    │ 100 │       72.0 │ 62.5-79.9 │      0 │       0 │           321 │        34.4 │
│ mmlu_pro  │ 100 │       59.0 │ 49.2-68.1 │      0 │       0 │           416 │        25.3 │
└───────────┴─────┴────────────┴───────────┴────────┴─────────┴───────────────┴─────────────┘
golden  answers found by: answer_tag 14, whole reply 13
    by group: code 100.0 (n=3), dates 100.0 (n=2), format 75.0 (n=4), honesty 100.0 (n=2), knowledge 100.0 (n=4), ...
gsm8k  answers found by: answer_tag 100
humaneval  tests: passed 81, failed 19
ifeval  prompt_level_strict 72.0  instruction_level_strict 79.5  prompt_level_loose 76.0  instruction_level_loose 82.7
strict/change_case 50.0  strict/combination 71.4  strict/detectable_content 90.9  strict/detectable_format 92.3  ...
mmlu_pro  answers found by: answer_is 100
    by group: biology 100.0 (n=6), business 42.9 (n=7), chemistry 66.7 (n=9), computer science 66.7 (n=3), ...
```

- **Title:** label, model, and thinking mode (`think`, `no-think` or `default`, with `(budget N)`), all taken from the first summary.
- **`n`:** reads `n/N (partial)` for an incomplete benchmark.
- **`errors`:** failed requests or graders.
- **`cut off`:** `n_truncated`.
- **`tokens/answer`:** the mean over error-free answers.
- **The lines under the table:** `metrics`, then `answers found by:` (or `tests:`), then `by group:`. Each is one line; long ones wrap on the console, which is why IFEval's `strict/<category>` scores start a new line above. Metric keys containing `/` (per-category scores) and `by group:` appear only in `report`, not after `run`. A non-zero `cut off` anywhere adds the token-limit warning.
- **How each answer was read.** `answers found by:` counts extraction methods. A grader that judges the whole reply (`contains`, `regex` and so on) records no method, so those answers are counted as `whole reply`: golden's `answer_tag 14, whole reply 13` covers all 27. A code benchmark prints its test outcomes instead: `tests: passed 81, failed 19`, plus `timeout` or `error` when there are any. IFEval prints no such line unless questions failed (then `answers found by: error 2, whole reply 98`). See [`extract.md`](extract.md).

### Compare two runs

```powershell
$env:COLUMNS = '150'
kumaru bench compare runs\bench\<A> runs\bench\<B>            # A = baseline, B = candidate
kumaru bench compare runs\bench\<A> runs\bench\<B> -e 3       # plus up to 3 flipped questions per direction
```

Real output: the same model and profile run twice, once under the old per-benchmark budgets and once under 8,192. It is a good picture of the noise floor:

```
A = qwen25-7b-gguf-default   B = qwen25-7b-gguf-default
┌───────────┬─────┬──────┬──────┬──────────────────────┬─────────────────┬──────────┬─────────────────────┬───────────────┐
│ benchmark │   n │  A % │  B % │       B - A (95% CI) │ only A / only B │ p (Holm) │ verdict             │ tokens A -> B │
├───────────┼─────┼──────┼──────┼──────────────────────┼─────────────────┼──────────┼─────────────────────┼───────────────┤
│ golden    │  27 │ 96.3 │ 96.3 │  +0.0 (+0.0 to +0.0) │           0 / 0 │        1 │ no clear difference │      94 -> 86 │
│ gsm8k     │ 100 │ 96.0 │ 92.0 │  -4.0 (-9.0 to +0.0) │           5 / 1 │    0.875 │ no clear difference │    249 -> 244 │
│ humaneval │ 100 │ 80.0 │ 81.0 │  +1.0 (-3.0 to +6.0) │           2 / 3 │        1 │ no clear difference │    208 -> 206 │
│ ifeval    │ 100 │ 65.0 │ 72.0 │ +7.0 (+0.0 to +14.0) │          3 / 10 │    0.461 │ no clear difference │    296 -> 321 │
│ mmlu_pro  │ 100 │ 60.0 │ 59.0 │  -1.0 (-9.0 to +7.0) │           8 / 7 │        1 │ no clear difference │    407 -> 416 │
└───────────┴─────┴──────┴──────┴──────────────────────┴─────────────────┴──────────┴─────────────────────┴───────────────┘
Only questions the two runs answered differently decide the test (only A / only B). p is an exact McNemar test, Holm-corrected across benchmarks.

gsm8k: only B right (1)
  gsm8k/991  Solve the following math problem. Think step by step, then write the final answer on its own last l…
    expected 15  ·  A: 30  ·  B: 15

gsm8k: only A right (5)
  gsm8k/409  Solve the following math problem. Think step by step, then write the final answer on its own last l…
    expected 168000  ·  A: 168000  ·  B: 42000
```

The same model moved by 4–7 points with nothing significant. That is what the statistics in [`stats.md`](stats.md) predict at n = 100.

How it works:

- **Pairing.** Benchmarks present in both runs are compared. Within each, questions are paired by id among the records both runs have. `n` shows `(+k unpaired)` for questions only one run has, such as a different sample or a partial run.
- **Partial runs compare fine.** `compare` reads `items.jsonl`, so you can compare against a run that is still going.
- **What is compared.** The boolean `correct`; for IFEval that is prompt-level strict. `B - A (95% CI)` is the paired bootstrap. `only A / only B` are the discordant counts that decide the McNemar test. `p (Holm)` is corrected across the benchmarks in this comparison. `tokens A -> B` is the mean completion tokens over error-free paired answers.
- **`verdict`:**
  - `no paired questions` when n = 0;
  - `B better` / `A better` when `p_holm < 0.05`;
  - `B better? (not after correction)` when only the raw p < 0.05;
  - `no clear difference` otherwise.
- **Flips (`-e N`).** For each benchmark, "only B right (k)" comes first, then "only A right (k)". Each entry shows up to N questions with id, the first 100 characters of the prompt, and `expected · A: · B:` (the extracted answer or the error, `-` when none). IFEval and HumanEval store no expected answer, so read their `detail` in `items.jsonl`.

### Read the records yourself

```powershell
Get-Content runs\bench\<run>\golden\items.jsonl | ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { -not $_.correct } | Select-Object id, extracted, expected, finish_reason
```

```python
from pathlib import Path
from kumaru.eval.benchmarks.runner import load_results

summary, records = load_results(Path(r"runs\bench\<run>\gsm8k"))
cut = [r.id for r in records.values() if r.finish_reason == "length"]
```

### From Python

```python
import asyncio
from kumaru.core.config import load_config
from kumaru.eval.benchmarks.runner import BenchSettings, run_benchmarks, compare_runs, render_comparisons
from rich.console import Console

cfg = load_config("test")                     # echo backend: offline, no model
settings = BenchSettings(limit=10)            # out_dir=Path(...) to write somewhere other than runs\bench
out_dir, summaries = asyncio.run(run_benchmarks(cfg, ["golden"], settings, profile="test"))
print(out_dir.name, summaries[0].label, summaries[0].params["max_tokens"])   # <stamp>-test-default test-default 8192
render_comparisons(Console(), compare_runs(out_dir, out_dir))                # a run against itself: "no clear difference"
```

## How to extend it

- **New benchmarks need no runner change.** Register them (see [`__init__.md`](__init__.md#add-a-benchmark-end-to-end)). `generation_overrides`, `aggregate`, `blocking_grader`, `needs_code_execution` and `BenchItem.max_tokens` are the hooks the runner honours.
- **New summary fields:** add them to `BenchSummary` with a default, so old `summary.json` files still validate, and fill them in `summarize`.
- **Anything that changes what is sent** must be in the record key (it is, through the messages and params) or in `_fingerprint`. If it should forbid mixing sessions, it must go in `_fingerprint`. The agent prompt shows why both exist: it states today's date, so the key hashes the bare question and the fingerprint holds the settings that shape the prompt.
- **Changing `item_seed` breaks pairing with every existing run.** `test_item_seeds_are_pinned_because_old_runs_pair_on_them` exists to stop that.
- **Tests:** `tests\test_bench_runner.py` runs the whole runner offline on the echo backend with a 4-question golden file. It derives the expected budget from `GoldenSet.answer_tokens` (`BUDGET`), so budget changes don't need test edits. Run it alone:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests\test_bench_runner.py -q -p no:cacheprovider
  ```

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `<run>\<bench> holds results from different settings (model, decoding parameters, seed or prompt mode).` | Resuming with other model, thinking, temperature, `--max-tokens`, `--seed` or `--use-agent-prompt` settings. With the flag, an edited persona (`system_prompt`, `enabled_tools`, `max_history_tokens`, `memory_policy`) counts too, as does a run saved before those settings joined the fingerprint. | Repeat the original flags (see `run.json` `command`), or omit `-o` for a fresh run (the hint says the same). |
| `'humaneval' grades answers by running model-written code on this machine.` | Missing consent. | `--allow-code-exec`, after reading [`humaneval.md`](humaneval.md). |
| `Stopped <bench>: 5 questions failed in a row. Last error: ...` | Server down, out of memory, or wrong URL. Grader crashes count too; as the hint says, an error starting `grader failed` is the benchmark's own bug. | Backend: `kumaru chat -q hi` with the same flags, then re-run with `--out <dir>` to resume; failed questions are retried. Grader: fix it, then start a new run directory (the fingerprint covers settings, not code). |
| `cut off` > 0 and the token-limit warning | The budget binds. Thinking runs are the usual case. | Thinking on: raise `--thinking-budget`. Off: `--max-tokens` above 8,192. Use a new directory if you change `--max-tokens` (it is in the fingerprint). |
| `bench report` shows `0/100 (partial)` for a running or killed run | `summary.json` is written only when a benchmark starts and when it stops; after a hard kill it stays stale. | Use `bench compare` (reads `items.jsonl`), or resume: a resume rewrites `summary.json` from `items.jsonl` as it starts. |
| Resuming re-asks every question | The record keys changed: other messages (a prompt-template edit) or other budgets (e.g. a run from before the 8,192 default). A `--use-agent-prompt` run resumed on a later day is not affected: its keys hash the bare question. | Expected. Prefer a fresh directory so old and new answers don't mix. |
| Many `unpaired` in a comparison | Different `-n`/`--seed`, or a partial run. | Same `-n` and `--seed`; finish both runs. |
| `The two runs share no benchmark.` | Disjoint benchmark sets. | The hint lists each run's benchmarks. |
| `No benchmark results in <path>` | Not a run directory, or a benchmark directory without `summary.json`. | Point at `runs\bench\<run>` or `runs\bench\<run>\<bench>`. |
| `Two benchmarks share the name 'golden-set' (the second is ...), so their results would share one folder.` | Two golden files with the same stem, e.g. `a\set.jsonl` and `b\set.jsonl`, are both named `golden-<stem>`. | Rename one of the files, as the hint says. |
| Throughput lower than expected (`j=1`) | Not llama-server, or a unified KV cache too small for `prompt + budget`. | `-j N` explicitly, or a larger `--context`. |

## Related files

- `src\kumaru\cli.py` — `kumaru bench run|report|compare|list` (the flags above).
- `src\kumaru\eval\benchmarks\base.py` — the benchmark contract and `select_items` ([`base.md`](base.md)).
- `src\kumaru\eval\benchmarks\stats.py` — Wilson, McNemar, bootstrap, Holm ([`stats.md`](stats.md)).
- `src\kumaru\agent\chat.py` — `ChatAgent.resolve_params` and `prepare`, shared with chat.
- `src\kumaru\backends\llamacpp_server.py` — reports `total_slots` and `n_ctx` used by `default_concurrency`.
- `tests\test_bench_runner.py` — pins files on disk, resume, refusal, failure handling, comparison, labels, seeds, concurrency, redaction and torn lines. Among them: `test_two_golden_files_with_the_same_name_are_refused`, `test_agent_prompt_runs_resume_the_next_day_without_asking_again`, `test_a_grader_failing_every_question_stops_the_run_and_says_so` and `test_summary_lines_say_how_each_answer_was_read`.
- [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md) — results and grader validation.
