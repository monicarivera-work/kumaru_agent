# Architecture

> Why the system is shaped this way. Every decision here was made to keep one property true: **new capability should be an additive change, never a surgical one.**

---

## The dependency rule

```
         ┌──────────────────────────────────┐
         │  ui/          server/    cli.py  │   entry points
         └───────────────┬──────────────────┘
                         ↓
         ┌──────────────────────────────────┐
         │            agent/                │   policy
         └───────────────┬──────────────────┘
                         ↓
         ┌──────────────────────────────────┐
         │           backends/              │   mechanism
         └───────────────┬──────────────────┘
                         ↓
         ┌──────────────────────────────────┐
         │             core/                │   contracts
         └──────────────────────────────────┘

         train/ and eval/ sit beside this stack.
         train/ needs only core/; backends/ load the
         adapters it produces. eval/ calls backends/
         with agent/'s settings, so it measures the
         settings chat actually uses.
```

**Imports point downward only.** `core` imports nothing from Kumaru. `backends` imports only `core`. `agent` imports `core` and `backends`. `server` imports everything below it. Only `cli.py` imports `train` and `eval`, so the server never loads training or benchmark code.

This is not aesthetic. It buys three concrete things:

1. **No import cycles, ever.** Not by convention — by construction.
2. **Testability without a GPU.** `core` and `agent` are testable as pure functions; the `echo` backend covers the rest.
3. **Replaceability.** Any layer can be swapped without touching the others, because the layer above it depends on an *interface*, not an implementation.

---

## Decision 1 — One backend interface

**The problem.** A system that must keep absorbing new inference engines cannot hard-code them.

**The alternative we rejected:**

```python
if backend == "transformers": ...
elif backend == "llamacpp":   ...
elif backend == "vllm":       ...   # every new idea edits a core file
```

Every new engine becomes a surgical change to working code, with a matching chance of breaking it. Over years, the file becomes untouchable and the project stops absorbing anything.

**What we did.** Three abstract methods in `backends/base.py`, plus one method with a default:

```python
async def _load(self) -> None                                      # abstract
def info(self) -> BackendInfo                                      # abstract
def stream(self, messages, params) -> AsyncIterator[StreamChunk]   # abstract
def count_tokens(self, messages) -> int   # default: ~3.6 characters per token; real tokenizers override it
```

Everything above the backend layer is written once, against that interface. Five backends implement it:

| Backend | What it runs | Use it for |
|---|---|---|
| `echo` | No model; replies with your own message | Tests, and UI work without a GPU |
| `transformers` | PyTorch, in-process | Models and LoRA adapters from `kumaru train` |
| `llamacpp` | llama.cpp through its Python bindings, in-process | One user, simple setups |
| `llamacpp_server` | llama.cpp's own `llama-server`, as a child process | The default: `qwen36` and other GGUF models ([Decision 13](#decision-13--the-default-engine-runs-as-a-child-process)) |
| `openai_compat` | Any OpenAI-style server: vLLM, SGLang, a hosted API | Bigger hardware |

**The consequence.** Moving from a 7B on a desktop to a 70B on a cluster is editing YAML:

```yaml
backend:
  name: openai_compat
  base_url: http://gpu-cluster:8000/v1
  model: Qwen/Qwen2.5-72B-Instruct
```

Not every change is a YAML edit. Thinking, added with Qwen3.6, touched every layer: a `thinking` field on `GenerationParams`, a `reasoning` channel on `StreamChunk` and `Message`, the agent's defaults, the HTTP schema and a UI button. That is because it is a new *kind of output*, a change to the contract itself. The rule of thumb: a new engine is one new file; a new kind of output is a change to `core/types.py` and to everything that displays it.

---

## Decision 2 — Streaming is the primitive, batch is derived

`stream()` is abstract. `generate()` is concrete and simply drains the stream.

**Why.** Two separate generation methods drift apart. One gets a bug fix, the other does not. Deriving one from the other makes divergence impossible, and means a new backend implements generation *once*.

It also means every backend gets low time-to-first-token for free. In a chat UI, **time-to-first-token dominates perceived speed far more than total tokens/second**, so making the fast path the default path is the right bias.

---

## Decision 3 — Async interface over synchronous engines

`torch.generate()` blocks. The server must serve many connections from one process. A blocking call in an async handler freezes every other request.

`iterate_in_thread()` in `backends/base.py` bridges the gap: it runs a blocking generator on a worker thread and hands items back to the event loop through a **bounded** queue. Bounded matters — an unbounded queue lets a fast GPU run thousands of tokens ahead of a slow HTTP client and exhaust memory.

Keeping the interface async also means genuinely concurrent engines (vLLM, TGI) slot in with no caller changes.

A worker thread makes *waiting* concurrent, not *generation*. The in-process backends (`transformers`, `llamacpp`) have no batching, so they are meant for one user at a time. Serving several users well needs an engine that schedules requests together, such as `llama-server`'s slots or vLLM, which is one reason the default engine moved out of process ([Decision 13](#decision-13--the-default-engine-runs-as-a-child-process)).

---

## Decision 4 — The agent is stateless

`ChatAgent` stores no conversation history. Callers send the full message list on every turn, exactly like the OpenAI API.

**Why this is the important decision.** It is what makes horizontal scaling trivial:

- Any replica can serve any request.
- No session affinity, no sticky routing, no shared cache.
- Scaling out is `docker compose up --scale kumaru=8` behind a round-robin load balancer.

Statefulness is easy to add and very hard to remove. Persistence belongs in a store the *caller* owns — in Kumaru's UI, that is `localStorage` — not smuggled into the agent.

---

## Decision 5 — Bounded admission control

```yaml
server:
  max_concurrent_requests: 8
```

A semaphore in `server/app.py` caps in-flight generations. Requests beyond the cap wait their turn; none are rejected.

**Why it is load-bearing, not polish.** On one GPU, admitting unlimited concurrency makes every request slower *and* eventually triggers a CUDA OOM that kills the process. Bounded admission converts a hard crash into graceful queueing — the difference between a service that degrades and one that dies.

With `llamacpp_server` there are two queues. Kumaru admits `max_concurrent_requests` (8); the engine runs `llama_server.parallel` of them at once (4 slots) and queues the rest itself. The engine's limit protects VRAM: llama-server reserves its KV cache when it starts, so an extra request waits for a free slot rather than running out of memory. Kumaru's limit protects the API server from an unbounded backlog.

The same reasoning produces the tool-iteration cap in `agent/tools.py`. Tool calling is an unbounded loop driven by a stochastic process; without a cap, one confused model burns your GPU indefinitely. Limits on unbounded loops are safety properties.

---

## Decision 6 — Non-blocking startup

Loading a model takes from seconds to minutes, and the first start of a Hugging Face model spec downloads the file first (17.7 GB for `qwen36`). A synchronous startup hook would hold the port closed for the whole time, and every client would see connection-refused with no explanation.

Instead: the model loads in a background task, the port opens immediately, `/healthz` reports `loading`, and the UI shows progress. Requests that arrive early `await` readiness for up to ten minutes, then get a clear "still loading" error instead of hanging.

The same care applies to failure. If loading fails or is cancelled halfway, `llamacpp_server` stops the half-started engine process, so a failed start never leaves VRAM held.

This is the difference between "it's broken" and "it's warming up", and it costs about twenty lines.

---

## Decision 7 — Three registries, with entry points

```python
BACKENDS   = Registry("backend",   entry_point_group="kumaru.backends")
TOOLS      = Registry("tool",      entry_point_group="kumaru.tools")
BENCHMARKS = Registry("benchmark", entry_point_group="kumaru.benchmarks")
```

Components announce themselves with a decorator; the system resolves them by name from config at runtime.

**Lazy registration** matters for startup time: `BACKENDS.register_lazy("transformers", "kumaru.backends.transformers_backend:TransformersBackend")` records a dotted path without importing torch. `kumaru --help` stays instant, and `kumaru serve --backend echo` never pays for a multi-second torch import.

**Entry points** extend this across package boundaries. Someone can publish `kumaru-mamba` with:

```toml
[project.entry-points."kumaru.backends"]
mamba = "kumaru_mamba:MambaBackend"
```

`pip install kumaru-mamba`, and it appears in your UI dropdown. No fork, no patch, no change to this repository. A broken third-party plugin is logged and skipped rather than crashing startup — one bad plugin must never take down the application.

Benchmarks follow the same pattern. The built-ins are lazy entries (`BENCHMARKS.register_lazy("gsm8k", "kumaru.eval.benchmarks.gsm8k:GSM8K")`), datasets download only when a run needs them, and a separate package can add benchmarks under the `kumaru.benchmarks` group. A suite name stands for several of them: `multipl_e` runs one benchmark per programming language (`BENCHMARK_SUITES`). [GUIDE.md](../GUIDE.md#add-a-benchmark) walks through adding one.

---

## Decision 8 — Wire schemas separate from core types

`core/types.py` holds the internal vocabulary. `server/schemas.py` holds the published HTTP contract. They look similar and are deliberately not shared.

**Why.** Core types must be free to refactor. Wire schemas are what external clients depend on. Collapsing them saves a few lines today and makes every internal rename a breaking API change forever.

The wire schemas mirror OpenAI's shapes, which means the official SDK, LangChain, LlamaIndex, Continue.dev, and most chat frontends work against Kumaru with only a changed `base_url`. **You inherit an ecosystem instead of building one.**

One detail: every generation field on `ChatCompletionRequest` defaults to `None`, so "absent" is distinguishable from "explicitly zero". Otherwise a client library that helpfully sends `temperature=1.0` would silently stomp your tuned server defaults.

---

## Decision 9 — Configuration merged by hand

Four sources, explicit precedence, about forty lines of obvious code in `core/config.py`:

```
field defaults  <  YAML  <  KUMARU_* env  <  CLI overrides
```

We use Pydantic for validation but not for source resolution. **Precedence bugs in configuration are miserable to debug**, and being able to *read* the rule end to end is worth more than the abstraction that would hide it.

Two supporting choices:

- `extra="forbid"` everywhere — a typo is a startup error, not a silently ignored line.
- `deep_merge` merges nested dicts key-by-key but replaces lists wholesale. "Append or replace?" is ambiguous for lists, and silent appending produces duplicated stop sequences and CORS origins that take an hour to track down.

---

## Decision 10 — A zero-build UI

Three static files: `index.html`, `styles.css`, `app.js`. No npm, no bundler, no framework.

**Why.** A build step is a dependency on a toolchain that rots. Two years from now, `npm install` on an abandoned lockfile is a bad afternoon. Three files the server hands out directly will still work, and you can read every line that affects what you see.

Two details in `app.js` worth knowing:

- **Markdown is rendered escape-first.** All HTML is escaped before any markup is generated. A model is an untrusted input source; its output is treated exactly like user input.
- **Re-render is throttled to one animation frame.** Without that, a fast GPU triggers hundreds of layout passes per second and the UI stutters even though the model is keeping up.

---

## Decision 11 — Training writes its own provenance

Every `kumaru train` run writes `train_config.yaml`, `metrics.json`, and `example_preview.txt` into its output directory alongside the weights.

**Why.** A checkpoint without its recipe is an artifact you cannot reproduce, cannot trust, and cannot improve on. Six months later, "what produced this adapter?" needs a precise answer.

`example_preview.txt` earns its place specifically: it shows one encoded training example split into the full sequence and the supervised region. Reading it is the fastest way to catch a loss-masking bug — a class of error that otherwise produces a beautiful loss curve and a useless model.

---

## Decision 12 — A framework-free training loop

`train/pretrain_nano.py` has no `Trainer` class. The optimizer step, LR schedule, gradient accumulation, clipping, mixed precision, and checkpointing are written out.

**Why.** These are exactly the mechanisms you need to understand when a run will not converge. A framework hides them, which is correct for production and wrong for comprehension. `train/finetune_lora.py` *does* use the HuggingFace `Trainer`, because for LoRA the mechanisms are standard and the framework is a genuine win.

Having both is the point: one file to learn from, one file to work with.

---

## Decision 13 — The default engine runs as a child process

**The problem.** The models worth running change every few months, and their new parts (hybrid attention, mixture-of-experts routing) reach llama.cpp's C++ code first. The Python bindings behind the in-process `llamacpp` backend trail by weeks or months, and on Windows with CUDA, updating them means compiling.

**What we did.** `llamacpp_server` starts llama.cpp's own `llama-server` program and talks to it over HTTP. It is the `openai_compat` backend with a process attached. Its `_load()` finds or downloads the GGUF file, starts the server on a free port and waits for `/health`; after that, every request goes through the inherited OpenAI client code.

**What that buys:**

- **Upgrades are downloads.** Support for a new architecture is `.\scripts\get-llama-server.ps1 -Version bNNNNN`: a new binary, checksum-verified and swapped in whole, with no compiler and no Python change.
- **Batching.** `llama-server` runs several requests in one forward pass (`llama_server.parallel` slots). [SCALABILITY.md](SCALABILITY.md#what-batching-buys-measured-on-this-machine) shows what that is worth on this machine.
- **Crash isolation.** A CUDA out-of-memory error kills the child, not the API server. Kumaru reports the engine's last log lines with a hint.
- **Exact memory release.** Stopping the process returns all of its VRAM. An in-process engine depends on Python dropping every reference and PyTorch's caching allocator letting go. This is what makes the UI's model switch (close the old model, then load the new one) reliable.
- **Thinking handled by the engine.** With `--jinja`, `llama-server` applies the model's own chat template, which reads the `enable_thinking` switch Kumaru sends. It returns the thinking as `reasoning_content`, which maps onto Kumaru's `reasoning` channel, and it enforces the thinking budget (`reasoning_budget_tokens`).

**The hazard, and the fix.** A model server that outlives its parent silently holds about 20 GB of VRAM. On Windows the child joins a Job Object with `KILL_ON_JOB_CLOSE`, so the OS kills it even when Kumaru is killed from Task Manager. On Linux it receives `SIGTERM` when the parent dies (`PR_SET_PDEATHSIG`). An `atexit` hook covers normal exits everywhere.

**The cost.** A second process and a local HTTP hop per request, both small next to generating tokens; and a binary per platform. `get-llama-server.ps1` is Windows-only; elsewhere, install llama.cpp and put `llama-server` on `PATH`.

---

## Decision 14 — Benchmarks are paired, resumable and checked

**The problem.** Every upgrade asks one question: is B better than A? On 100 questions a single score is uncertain by as much as ±10 points, so two scores compared by eye mislead. Runs take hours on a desktop that is also used for other things, so they get interrupted. And a grader that misreads "$1,080" turns right answers into wrong ones, which looks exactly like a worse model.

**What we did** (`eval/benchmarks/`):

- **The same questions for every model.** A seeded sample (100 per benchmark by default, seed 1234) means two runs answer identical questions and can be compared question by question. Pairing removes the difference in question difficulty, which is much of the noise.
- **Paired statistics.** `kumaru bench compare` counts the questions only one model got right and tests them with an exact McNemar test. It puts a paired bootstrap interval on the difference and applies Holm's correction across benchmarks. It says "B better" only when the corrected test agrees (`stats.py`, and `verdict()` in `runner.py`).
- **Every answer is saved as soon as it is graded.** Each record is appended to `items.jsonl` and flushed. A record holds the output, the thinking, the extracted answer, the verdict, the token count and why generation stopped. A crash or Ctrl+C loses only the questions in flight; re-running with `--out` resumes, and failed questions are retried.
- **A fingerprint guards each run.** A hash of the model, adapter, decoding settings, seed and prompt mode is stored with the results; for the code benchmarks it also covers where programs run and the compiler versions. Resuming with different settings is refused, so two experiments are never silently mixed. Each record also stores a hash of its question, answer budget and seed, and a resume reuses the saved answer only if that hash still matches. (That hash leaves out the agent's system prompt, which states today's date; otherwise a run resumed the next day would ask every question again.) Neither hash covers grader code, and a resume keeps saved verdicts, so after changing a grader start a new run folder.
- **One seed per question,** derived from the run seed and the question ID, so sampling does not depend on the order in which concurrent answers finish. Batched GPU arithmetic is still not bit-exact, which is one reason [BENCHMARKS.md](BENCHMARKS.md) compares a run with a repeat of itself.
- **Graders are tested like code.** `scripts/check_graders.py` grades each dataset's reference answers, phrased the ways models phrase them. Every one must grade correct, and a wrong answer must grade wrong. The IFEval checks are compared with Google's reference implementation. MultiPL-E's languages have no reference answers, so Kumaru re-grades answers that MultiPL-E's authors graded with their own harness, and must reach the same verdicts, bar a few documented toolchain differences.
- **Model-written code is opt-in.** HumanEval and MultiPL-E run code a model wrote, so they need `--allow-code-exec`. Each program gets an empty temporary folder, a scrubbed environment and time limits (15 seconds to run), and everything it started is killed when it ends; on Windows its memory is also capped. HumanEval's Python also has the destructive parts of `os`, `shutil` and `subprocess` disabled. That stops accidents, not attackers. `--sandbox docker` runs each program in a container with no network instead, from an image built from `docker/sandbox/Dockerfile`; use it, a VM or another container for models you do not trust.
- **A missing compiler is not a score.** A MultiPL-E language whose compiler is missing, or which grades a known answer wrongly, is skipped with a note, never scored 0, so a score always means the model's code was run.

**The consequence.** A model swap is a measured decision, not an impression. The rule is in [SCALABILITY.md](SCALABILITY.md#absorbing-the-next-model): switch on "B better", or on "no clear difference" when the new model is faster or smaller.

**What it caught.** The first Qwen3.6 run hit the answer-token limit on up to 16% of one benchmark's answers, which understated its scores. Every summary now counts truncated answers (`n_truncated`), and the default answer budget was raised to 8,192 tokens. [BENCHMARKS.md §6.1](BENCHMARKS.md#61-the-truncation-confound) has the details.

---

## What is deliberately missing

| Not here | Why |
|---|---|
| A database | The agent is stateless. Adding persistence would undo the property that makes scaling trivial. |
| User accounts / multi-tenancy | This is a local-first tool. Bearer-token auth exists for when you bind beyond localhost. |
| A vector store | RAG is a `MemoryPolicy` plus a tool. Both plug in without core changes. |
| Active tool calling | The seam exists and is inert. v0.2 is still chat-only, as specified. |
| A JS framework | See Decision 10. |
| Prometheus / OpenTelemetry | `Metrics` is ~30 lines with the same call sites. Swap it when you need it. |
| A sandbox built into Kumaru | Running model-written code locally stops accidents, not attackers. Real isolation is a container (`--sandbox docker`) or a VM, and Kumaru should not pretend to be one. |
| An LLM judge for grading | Every built-in grader is deterministic and checked against answer keys. A judge model has its own error rate; see [BENCHMARKS.md §10](BENCHMARKS.md#10-where-evaluation-goes-next). |

Each of these is an *addition*, not a *refactor*. That is the test every design decision here had to pass.

---

## Reading order for the source

1. `core/types.py` — the vocabulary everything speaks
2. `core/config.py` — how settings resolve
3. `backends/base.py` — the central interface
4. `backends/echo.py` — the smallest complete implementation
5. `backends/llamacpp_server.py` — a real engine, run as a managed child process
6. `agent/chat.py` — how a turn is orchestrated
7. `server/routes_chat.py` — how it reaches the network
8. `eval/benchmarks/runner.py` and `stats.py` — how "better" is measured
9. `train/model_nano.py` — what a transformer actually is
10. `train/data.py` — why loss masking decides whether fine-tuning works

Each has a companion page in [`howto/`](howto/) with its API, usage, extension points, and failure modes.
