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
         They consume core/ and backends/, and produce
         artifacts (adapters) that backends/ consume.
```

**Imports point downward only.** `core` imports nothing from Kumaru. `backends` imports only `core`. `agent` imports `core` and `backends`. `server` imports everything below it.

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

**What we did.** Four abstract methods in `backends/base.py`:

```python
async def _load(self) -> None
def info(self) -> BackendInfo
def count_tokens(self, messages) -> int
def stream(self, messages, params) -> AsyncIterator[StreamChunk]
```

Everything above the backend layer is written once, against that interface.

**The consequence.** Moving from a 7B on a desktop to a 70B on a cluster is editing YAML:

```yaml
backend:
  name: openai_compat
  base_url: http://gpu-cluster:8000/v1
  model: Qwen/Qwen2.5-72B-Instruct
```

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
  max_concurrent_requests: 4
```

A semaphore in `server/app.py` caps in-flight generations.

**Why it is load-bearing, not polish.** On one GPU, admitting unlimited concurrency makes every request slower *and* eventually triggers a CUDA OOM that kills the process. Bounded admission converts a hard crash into graceful queueing — the difference between a service that degrades and one that dies.

The same reasoning produces the tool-iteration cap in `agent/tools.py`. Tool calling is an unbounded loop driven by a stochastic process; without a cap, one confused model burns your GPU indefinitely. Limits on unbounded loops are safety properties.

---

## Decision 6 — Non-blocking startup

Loading a 7B model takes 30–60 seconds. A synchronous startup hook would hold the port closed for the whole time, and every client would see connection-refused with no explanation.

Instead: the model loads in a background task, the port opens immediately, `/healthz` reports `loading`, and the UI shows progress. Requests that arrive early simply `await` readiness.

This is the difference between "it's broken" and "it's warming up", and it costs about twenty lines.

---

## Decision 7 — Two registries, with entry points

```python
BACKENDS = Registry("backend", entry_point_group="kumaru.backends")
TOOLS    = Registry("tool",    entry_point_group="kumaru.tools")
```

Components announce themselves with a decorator; the system resolves them by name from config at runtime.

**Lazy registration** matters for startup time: `BACKENDS.register_lazy("transformers", "kumaru.backends.transformers_backend:TransformersBackend")` records a dotted path without importing torch. `kumaru --help` stays instant, and `kumaru serve --backend echo` never pays for a multi-second torch import.

**Entry points** extend this across package boundaries. Someone can publish `kumaru-mamba` with:

```toml
[project.entry-points."kumaru.backends"]
mamba = "kumaru_mamba:MambaBackend"
```

`pip install kumaru-mamba`, and it appears in your UI dropdown. No fork, no patch, no change to this repository. A broken third-party plugin is logged and skipped rather than crashing startup — one bad plugin must never take down the application.

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

## What is deliberately missing

| Not here | Why |
|---|---|
| A database | The agent is stateless. Adding persistence would undo the property that makes scaling trivial. |
| User accounts / multi-tenancy | This is a local-first tool. Bearer-token auth exists for when you bind beyond localhost. |
| A vector store | RAG is a `MemoryPolicy` plus a tool. Both plug in without core changes. |
| Active tool calling | The seam exists and is inert. v0.1 is chat, as specified. |
| A JS framework | See Decision 10. |
| Prometheus / OpenTelemetry | `Metrics` is ~30 lines with the same call sites. Swap it when you need it. |

Each of these is an *addition*, not a *refactor*. That is the test every design decision here had to pass.

---

## Reading order for the source

1. `core/types.py` — the vocabulary everything speaks
2. `core/config.py` — how settings resolve
3. `backends/base.py` — the central interface
4. `backends/echo.py` — the smallest complete implementation
5. `agent/chat.py` — how a turn is orchestrated
6. `server/routes_chat.py` — how it reaches the network
7. `train/model_nano.py` — what a transformer actually is
8. `train/data.py` — why loss masking decides whether fine-tuning works

Each has a companion page in [`howto/`](howto/) with its API, usage, extension points, and failure modes.
