# The Kumaru Programming Guide

> How to actually use this thing, day to day, the way you'd use any other serious tool.

This is the document to read after [`docs/QUICKSTART.md`](docs/QUICKSTART.md). It is about *operating* the system, not just starting it.

---

## Part 1 — The mental model

Six layers. Each one only knows about the layer directly beneath it.

```
  You / your code
        ↓
  Server        HTTP, streaming, admission control      server/
        ↓
  Agent         prompt, memory budget, tools            agent/
        ↓
  Backend       one interface, five implementations     backends/
        ↓
  Model         weights + optional LoRA adapter         train/ produces these
        ↓
  Hardware      GPU, VRAM, memory bandwidth
```

**The one rule worth internalizing:** dependencies point *downward only*. `core` imports nothing from the layers above it. A backend never imports the server. This is why you can replace any single layer without touching the others, and it is the entire reason the system stays modifiable as it grows.

When you want to change something, that rule tells you where to go:

| You want to… | Edit |
|---|---|
| Change how it answers | `agent.system_prompt` in your config |
| Change which model | `backend.model` in your config |
| Change how it decodes | the `generation` block |
| Make it reason before answering | the **Think** button, or `generation.thinking` |
| Change what it remembers | `agent.memory_policy` |
| Add a capability | a new `Tool` in `agent/tools.py` |
| Add an engine | a new `Backend` subclass |
| Measure something new | a new `Benchmark` in `eval/benchmarks/` |
| Change what it *is* | `kumaru train` |

---

## Part 2 — Configuration, which is most of the job

### The four layers

Later wins:

```
field defaults  <  YAML file  <  KUMARU_* env vars  <  CLI flags
```

```powershell
kumaru serve                                  # defaults
kumaru serve -c local-gpu                     # + YAML profile
$env:KUMARU_BACKEND__MODEL="Qwen/Qwen2.5-14B-Instruct"   # + env
kumaru serve -c local-gpu --port 8090         # + CLI flag (wins)
```

### The single most useful command

```powershell
kumaru config -c local-gpu
```

This prints the **fully merged** configuration. Whenever Kumaru appears to be ignoring a setting, run this before anything else. It converts "which layer won?" from a guess into a fact. It is also what you should paste into a bug report.

### Env var syntax

`KUMARU_` + section + `__` + field, upper-cased:

```powershell
$env:KUMARU_BACKEND__MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
$env:KUMARU_BACKEND__LOAD_IN_4BIT = "true"
$env:KUMARU_SERVER__PORT        = "9000"
$env:KUMARU_GENERATION__TEMPERATURE = "0.2"
```

Values are parsed with YAML rules, so `true`, `42`, and `0.7` arrive as the right Python type.

### Typos fail loudly

Config models are declared `extra="forbid"`. A misspelled key is a startup error, not a silently ignored line. This removes an entire category of "why is my setting doing nothing" debugging.

---

## Part 3 — Getting good answers

In order of leverage. Do not skip to the bottom.

### 1. The system prompt (seconds, largest effect)

```yaml
agent:
  system_prompt: >-
    You are a senior Rust engineer reviewing code for a systems team.
    Point out correctness and safety issues first, then performance.
    Show corrected code. Never speculate about code you cannot see.
```

Kumaru always appends its own quality directives (`agent/prompt.py`), so you get answer-first formatting and anti-fabrication guardrails on top of whatever persona you write.

Keep it short. A 2,000-token system prompt costs 2,000 tokens on *every single request* and dilutes attention across instructions that matter.

### 2. Temperature (seconds)

| Task | Temperature |
|---|---|
| Extraction, classification, evals, anything reproducible | `0.0` |
| Code | `0.1–0.3` |
| General chat | `0.7` |
| Brainstorming | `0.9–1.2` |

At `0.0` decoding is greedy and fully deterministic. A subtlety worth knowing: low temperature does not make the model *smarter*, it makes it *consistent*. If the right answer is not near the top of the distribution, `T=0` just returns the same wrong answer every time.

### 3. Few-shot examples (minutes)

Put two or three ideal exchanges in the message list before the real question. For output-format problems this is usually more effective, and vastly cheaper, than fine-tuning.

### 4. Thinking (one click; costs time on every answer)

With `qwen36`, the **Think** button (or `kumaru chat --think`) lets the model reason privately before it answers. It helps most on multi-step math, logic and code, and little on recall or conversation. The cost is time: thousands of extra tokens per answer. The **thinking budget** slider caps them; at the cap the model is told to answer with what it has. Measured gains and costs: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

### 5. A better model (minutes + download)

```powershell
kumaru serve -c qwen36
```

Qwen3.6-35B-A3B is a mixture of experts: the knowledge of a 35B model at the speed of a much smaller one (about 110 tokens/s on a 3090). On a 24 GB GPU it is the biggest jump available without training. Other GGUF models load the same way, with `--model org/repo:QUANT`. Check the new model's card for its recommended sampling settings, because the profile's presets are Qwen's.

Before switching for good, measure: run the same benchmarks on both models and let `kumaru bench compare` tell you whether the difference is real or noise ([`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)).

### 6. Fine-tuning (an evening)

Only after 1–5 plateau. See [`docs/TRAINING.md`](docs/TRAINING.md).

---

## Part 4 — Using it as a tool

### From Python, with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="kumaru",
    messages=[{"role": "user", "content": "Summarize this error: ..."}],
    temperature=0.0,
)
print(resp.choices[0].message.content)
```

### Embedded, with no server at all

Lowest latency — no HTTP hop:

```python
import asyncio
from kumaru.agent.chat import ChatAgent
from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS
from kumaru.core.types import Message

async def main():
    cfg = load_config("local-gpu")
    backend = BACKENDS.create(cfg.backend.name, cfg.backend)
    await backend.load()

    agent = ChatAgent(backend, cfg)
    result = await agent.complete([Message.user("Explain RoPE in two sentences.")])
    print(result.message.content)
    print(f"{result.tokens_per_second:.1f} tok/s")

    await backend.aclose()

asyncio.run(main())
```

### Piping into a shell workflow

```powershell
kumaru chat -q "Write a PowerShell one-liner that finds files over 100MB" 2>$null
```

`2>$null` discards the log lines on stderr, leaving only the answer on stdout.

### Point another tool at it

Anything that accepts an OpenAI base URL works: Continue.dev, Cursor, Aider, LangChain, LlamaIndex, Open WebUI. Set:

```
Base URL: http://127.0.0.1:8080/v1
API key:  anything (unset by default)
Model:    kumaru
```

---

## Part 5 — The debugging ladder

When something is wrong, climb this in order. Each rung isolates a layer.

**1. Is the application stack healthy?**
```powershell
kumaru serve -c test
```
Echo backend, no model. If this works, the bug is in the model layer. If it does not, the bug is in the app layer. This one bisection saves hours.

**2. Is the environment right?**
```powershell
kumaru info
```
Shows Python, GPU, VRAM, bf16 support, and which optional packages are installed.

**3. Is the config what you think?**
```powershell
kumaru config -c local-gpu
```

**4. What is the server actually doing?**
```powershell
curl http://127.0.0.1:8080/healthz     # ok | loading | error
curl http://127.0.0.1:8080/v1/info     # full state, including load errors
curl http://127.0.0.1:8080/metrics     # request counts, latency, tok/s
```

**5. Turn up the logs.**
```powershell
kumaru serve --log-level DEBUG
```

**6. Measure instead of guessing.**
```powershell
kumaru eval --suite speed
kumaru eval --suite smoke
kumaru bench run golden -c qwen36     # your own questions, graded, every answer saved
```

### Common symptoms

| Symptom | Almost always |
|---|---|
| Slow on a GPU (< 5 tok/s) | Model offloaded to CPU. Check `device` in `kumaru info`. |
| `qwen36` drops from ~110 tok/s to single digits | VRAM spilled into shared system memory, usually because another app (a game) took some. Use the GPU-sharing recipe in [QUICKSTART Step 3](docs/QUICKSTART.md#step-3--run-the-smart-model-qwen36-no-pytorch-needed). |
| `llama-server exited during startup` | Not enough free VRAM, or the port is taken. The error ends with the last lines of llama-server's log. |
| Think answers take minutes | Expected: thinking spends thousands of tokens. Lower the thinking budget, or save Think for hard questions. |
| CUDA OOM at load | Model too big. Use `-c low-vram`, or lower `max_context_length`. |
| CUDA OOM mid-conversation | KV cache growth. Lower `agent.max_history_tokens`. |
| Answers ignore the system prompt | Context overflowed and trimming dropped it, or the model is too small. |
| Replies arrive all at once | A proxy is buffering the SSE stream. Connect directly. |
| Model repeats itself | Raise temperature; set `repetition_penalty: 1.1`; check stop sequences. |
| Works in `kumaru chat`, fails over HTTP | A serialization issue. Check `/v1/info` and the server logs. |

---

## Part 6 — Extending it

### Add a tool

Two steps. Nothing existing changes.

```python
# src/kumaru/agent/tools.py  (or your own module)
from kumaru.agent.tools import Tool
from kumaru.core.registry import TOOLS

@TOOLS.register("read_file")
class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file from disk. Use when the user refers to a file by path."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Absolute path."}},
        "required": ["path"],
    }
    timeout_s = 5.0
    read_only = True

    async def run(self, **kwargs) -> str:
        from pathlib import Path
        return Path(kwargs["path"]).read_text(encoding="utf-8")[:8000]
```

```yaml
agent:
  enabled_tools: [read_file]
```

The `description` field is the API documentation the *model* reads. Write it for that audience.

### Add a backend

Subclass `Backend`, implement four methods, decorate. See [`docs/howto/src/kumaru/backends/base.md`](docs/howto/src/kumaru/backends/base.md) for a complete worked example.

```python
@BACKENDS.register("my_engine")
class MyBackend(Backend):
    name = "my_engine"
    async def _load(self): ...
    def info(self) -> BackendInfo: ...
    async def stream(self, messages, params): ...
```

`generate()` is derived from `stream()`, so you implement generation once.

### Ship an extension as a separate package

```toml
[project.entry-points."kumaru.backends"]
mamba = "my_package:MambaBackend"
```

`pip install my-package` and it appears in the UI dropdown. No fork of this repo required.

### Add a benchmark

Subclass `Benchmark`: `load()` returns the questions, `grade()` scores one answer. A complete, working example (ARC-Challenge, 1,172 science questions):

```python
# src/kumaru/eval/benchmarks/arc.py
from kumaru.eval.benchmarks.base import BenchItem, Benchmark, Grade, load_hf_rows
from kumaru.eval.benchmarks.extract import extract_choice

PROMPT = (
    "Answer the following science question. Think step by step, then finish your answer "
    'with "The answer is (X)" where X is the letter of the correct option.\n\n'
    "Question: {question}\nOptions:\n{options}"
)


class ARCChallenge(Benchmark):
    name = "arc"
    description = "Grade-school science questions chosen to defeat simple retrieval."
    source = "allenai/ai2_arc (ARC-Challenge, test, 1,172 questions) - Clark et al. 2018"

    def load(self) -> list[BenchItem]:
        items = []
        for row in load_hf_rows("allenai/ai2_arc", "ARC-Challenge", "test"):
            texts, labels = row["choices"]["text"], row["choices"]["label"]
            letters = "ABCDE"[: len(texts)]  # a few questions label their options 1-4
            options = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, texts, strict=True))
            items.append(
                BenchItem(
                    id=f"arc/{row['id']}",
                    prompt=PROMPT.format(question=row["question"], options=options),
                    answer=letters[labels.index(row["answerKey"])],
                    meta={"letters": letters},
                )
            )
        return items

    def grade(self, item: BenchItem, output: str) -> Grade:
        letter, method = extract_choice(output, item.meta["letters"])
        return Grade(correct=letter == item.answer, extracted=letter, detail={"method": method})
```

Register it with one line in `src/kumaru/core/registry.py`, next to the others:

```python
BENCHMARKS.register_lazy("arc", "kumaru.eval.benchmarks.arc:ARCChallenge")
```

Then `kumaru bench run arc -c qwen36`. A separate package registers through the `kumaru.benchmarks` entry point instead, as for backends above.

**Before trusting a score, prove the grader.** Feed it the dataset's own answers, phrased the ways a model phrases them, plus wrong answers. It must accept every right one and reject every wrong one. `scripts/check_graders.py` does exactly this for the built-in benchmarks; copy its pattern. Details: [`docs/howto/src/kumaru/eval/benchmarks/base.md`](docs/howto/src/kumaru/eval/benchmarks/base.md).

---

## Part 7 — Habits that pay off

1. **Baseline first.** `kumaru eval --suite smoke --output baseline.json` for speed and sanity; `kumaru bench run golden gsm8k mmlu_pro -c <profile>` for accuracy. Every later claim is relative to these numbers.
2. **Change one thing at a time.** Two simultaneous changes tell you nothing about either.
3. **Hold the randomness fixed, then test the difference.** Compare runs with the same seed and the same decoding preset, and let `kumaru bench compare` decide whether a gap is real. Temperature 0 is the classic advice, but Qwen advises against greedy decoding for its current models (it causes endless repetition). And when several requests are batched on a GPU, even temperature 0 is not guaranteed to repeat bit for bit. So two identical runs still differ a little; [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) measures by how much.
4. **Archive the config with the artifact.** `kumaru train` does this automatically. A checkpoint without its recipe is not reproducible.
5. **Smoke-test the small model first.** `configs/train/lora-small.yaml` catches pipeline bugs in five minutes that would otherwise cost an evening.
6. **Read `example_preview.txt` before every training run.** It is the single highest-yield thirty seconds in the whole workflow.
7. **Watch `gnorm` during training.** A gradient-norm spike precedes a NaN loss by several steps.
8. **Prefer retrieval to fine-tuning for facts.** Cheaper, more accurate, and updatable.

---

## Part 8 — Where things live

| Question | File |
|---|---|
| What settings exist? | `configs/default.yaml` — every knob, commented |
| How does config resolution work? | `src/kumaru/core/config.py` |
| How does plugin discovery work? | `src/kumaru/core/registry.py` |
| What must a backend implement? | `src/kumaru/backends/base.py` |
| How does Kumaru run llama.cpp? | `src/kumaru/backends/llamacpp_server.py` |
| How is a model scored? | `src/kumaru/eval/benchmarks/`, and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| How is the prompt assembled? | `src/kumaru/agent/prompt.py` |
| What gets dropped when context fills? | `src/kumaru/agent/memory.py` |
| What are the HTTP endpoints? | `src/kumaru/server/routes_chat.py` |
| How does loss masking work? | `src/kumaru/train/data.py` |
| What is a transformer, exactly? | `src/kumaru/train/model_nano.py` |
| Per-file operating manuals | `docs/howto/` |

---

## Further reading

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — why the seams are where they are
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — how good each model measured, and how to measure your own changes
- [`docs/SCALABILITY.md`](docs/SCALABILITY.md) — every ceiling, and what to do at each
- [`docs/TRAINING.md`](docs/TRAINING.md) — the full training walkthrough
- [`docs/howto/`](docs/howto/) — one page per source file
