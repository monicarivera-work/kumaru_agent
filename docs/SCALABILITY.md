# What "scalable" means in Kumaru

"Scalable" is often marketing. Here it is a checklist of specific things that
can change without a rewrite. This document names each axis, says what you
change, and says what it costs.

---

## The one structural idea

```
core  <-  backends  <-  agent  <-  server  <-  ui
```

Dependencies point in one direction. `core` imports nothing from the project;
`backends` see only `core`; the `agent` sees `core` + a `Backend` interface;
the `server` sees the agent; the UI sees only HTTP.

Consequence: **every scaling move below is a change inside one layer.** If an
upgrade ever forces you to edit two layers at once, the layering has been
violated and that is the bug to fix first.

---

## Axis 1 — Model size and quality

Same code, bigger brain. Only `backend.name` and `backend.model` change.

| Stage | Hardware | Config | Realistic quality |
|---|---|---|---|
| 0 | anything | `echo` | none; plumbing test |
| 1 | 8 GB RAM laptop | `llamacpp`, 3B Q4 GGUF | short factual answers |
| 2 | 16 GB RAM | `llamacpp`, 7–8B Q4 | genuinely useful assistant |
| 3 | 16–24 GB VRAM | `transformers`, 7–14B bf16 | strong general assistant |
| 4 | 48 GB+ VRAM / multi-GPU | vLLM behind `openai_compat` | 30–70B; near frontier on many tasks |
| 5 | a cluster | vLLM/TGI cluster behind `openai_compat` | limited by budget, not by this code |

The jump from 4 to 5 is a URL change. That is the entire point of making
`openai_compat` a first-class backend rather than an afterthought.

## Axis 2 — Throughput (more requests at once)

Today: one process, one agent, backends serialised where the runtime demands it
(llama.cpp contexts and a single `transformers` model are not thread-safe).
Streaming generators run in a worker thread, so a slow generation never blocks
the event loop.

To go further, in order of effort:

1. **Batching** — put vLLM behind `openai_compat`. Continuous batching gives
   roughly an order of magnitude more throughput per GPU. No Kumaru change.
2. **Replicas** — run N copies of the inference server behind a load balancer,
   one `base_url`. No Kumaru change.
3. **Multiple Kumaru workers** — `uvicorn --workers N`. This requires memory to
   be shared, not per-process (see Axis 4).

## Axis 3 — Context length

`agent.max_history_turns` and `agent.max_history_chars` bound what is sent.
Raise them as the model's window grows. Beyond a real window (~128k) the answer
is not a bigger number, it is retrieval: summarise old turns, or embed them and
inject only the relevant ones. Both are changes to `agent/memory.py` alone,
because nothing else reads the history.

## Axis 4 — Concurrent users and persistence

`ConversationMemory` is a dict of bounded deques: correct for one machine, lost
on restart, unsharable across workers. Its interface (`add`, `history`, `clear`,
`extend`, `sessions`) is deliberately storage-shaped. Implement the same five
methods over SQLite (persistence), Redis (multi-worker), or Postgres
(multi-machine) and inject it:

```python
agent = ChatAgent(config, memory=RedisMemory(url="redis://localhost:6379"))
```

No other file changes.

## Axis 5 — Capability (beyond plain chat)

v1 is deliberately chat-only. The extension points already exist:

| Capability | Where it goes | Why there |
|---|---|---|
| Tool / function calling | a `ToolAgent` wrapping `ChatAgent.ask` | keeps the simple loop simple |
| RAG over your documents | a retriever consulted in `agent/prompt.py` | prompt assembly is already isolated |
| Multi-step planning | a new agent class reusing the same `Backend` | backends stay ignorant of strategy |
| Multi-modal input | a new backend + a wider `Message.content` | one layer each |
| Evaluation harness | new module reading the same `ChatAgent` | no production code touched |

Each is additive. None requires editing the backends, the server, or the UI.

## Axis 6 — Absorbing new research

The field moves faster than any codebase. Three mechanisms keep that
survivable:

* **A narrow backend contract.** `generate()` plus optional `stream()`. A new
  inference engine (whatever replaces vLLM) is one file that implements two
  methods, registered by name in `backends/__init__.py`.
* **A registry instead of an if/elif chain.** Adding a backend never edits a
  dispatch function, so two people can add two backends without conflicting.
* **Lazy imports.** A backend nobody selects costs nothing — no import, no
  dependency. That is why a new experimental backend can depend on a bleeding-
  edge library without breaking everyone else's install.

New prompting or decoding techniques land in `agent/prompt.py` or in the
`ChatRequest` sampling fields; neither is coupled to a particular engine.

## Axis 7 — Your own weights

See [`TRAINING.md`](TRAINING.md). A LoRA fine-tune produces a directory; point
`backend.model` at it with the `transformers` backend and the UI is unchanged.

---

## What is *not* scalable yet (be honest)

| Limitation | Consequence | Fix when you need it |
|---|---|---|
| Memory is in-process | history dies on restart; no multi-worker | implement the memory interface over SQLite/Redis |
| No authentication | must stay on `127.0.0.1` | auth proxy, or an API-key dependency in `server/app.py` |
| No rate limiting | one client can monopolise the model | reverse-proxy limits, or a semaphore per session |
| No request queue | a burst blocks on the backend lock | vLLM (queues internally) or an explicit work queue |
| No retrieval | knowledge is whatever the model memorised | add a retriever in `agent/prompt.py` |
| Metrics are log lines only | no dashboards | export Prometheus counters from `server/app.py` |

Every row is a bounded change in one layer. None requires a rewrite — which is
the only definition of "scalable" worth writing down.
