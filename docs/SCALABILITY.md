# Scalability

> You asked what "scalable" actually means for this system. This document answers it precisely: every ceiling, what you hit first, and what the next step is.

Scalability is not one property. It is **seven independent axes**, each with its own ceiling, its own warning sign, and its own next move. The design goal of Kumaru is that advancing along any axis is a *configuration* change or an *additive* change — never a rewrite.

| # | Axis | You hit this when… | Next move |
|---|---|---|---|
| 1 | [Model scale](#1-model-scale) | Answers aren't good enough | Bigger model, better quantization |
| 2 | [Serving throughput](#2-serving-throughput) | More than ~2 concurrent users | vLLM, continuous batching |
| 3 | [Context length](#3-context-length) | Conversations get truncated | Summarization, then retrieval |
| 4 | [Capability](#4-capability) | The model needs to *do* things | Tool registry, plugins |
| 5 | [Training](#5-training-scale) | LoRA isn't enough | Full FT → FSDP → multi-node |
| 6 | [Data](#6-data-scale) | Datasets exceed RAM | Memmap, streaming, sharding |
| 7 | [Evaluation](#7-evaluation-scale) | You can't tell if changes helped | Bigger suites, LLM judges, CI gates |

---

## 1. Model scale

### The mechanism

`backends/base.py` defines a four-method interface. Every engine implements it. Nothing above the backend layer — not the agent, not the server, not the UI — knows which model is loaded.

```python
class Backend(ABC):
    async def _load(self) -> None: ...
    def info(self) -> BackendInfo: ...
    def count_tokens(self, messages) -> int: ...
    def stream(self, messages, params) -> AsyncIterator[StreamChunk]: ...
```

So scaling the model is this:

```yaml
backend:
  model: Qwen/Qwen2.5-32B-Instruct   # was 7B
  load_in_4bit: true
```

### What fits where

Weight memory is `parameters × bytes_per_parameter`. Add roughly 15% for the KV cache and activations.

| Model | bf16 (2 B/param) | 8-bit | 4-bit NF4 | Minimum realistic GPU |
|---|---|---|---|---|
| 1.5B | 3.1 GB | 1.6 GB | 0.9 GB | Any, or CPU |
| 7B | 15.2 GB | 7.6 GB | 4.5 GB | 8 GB at 4-bit; 24 GB at bf16 |
| 14B | 29 GB | 15 GB | 8.5 GB | 12 GB at 4-bit |
| 32B | 66 GB | 33 GB | 19 GB | 24 GB at 4-bit |
| 70B | 142 GB | 71 GB | 40 GB | 2×24 GB, or 48 GB |
| 405B | 810 GB | 405 GB | 230 GB | 8×40 GB node |

**Your RTX 3090 (24 GB) ceiling:** a 32B model at 4-bit with an 8k context, or a 7B at bf16 with 16k. Both are real, working configurations today.

### The speed ceiling, and why quantization is a speed optimization

Single-stream decoding is **memory-bandwidth bound**, not compute bound. Generating one token requires reading essentially every weight from VRAM once:

```
tokens/second ≈ memory_bandwidth / bytes_of_weights
```

| GPU | Bandwidth | 7B bf16 ceiling | 7B 4-bit ceiling |
|---|---|---|---|
| RTX 3090 | 936 GB/s | ~61 tok/s | ~208 tok/s |
| RTX 4090 | 1008 GB/s | ~66 tok/s | ~224 tok/s |
| A100 80GB | 2039 GB/s | ~134 tok/s | ~453 tok/s |
| H100 SXM | 3350 GB/s | ~220 tok/s | ~744 tok/s |

Real numbers land at 50–70% of the ceiling. The practical consequence: **more FLOPS will not speed up single-stream generation. Only more bandwidth, or fewer bytes, will.** That is why 4-bit quantization roughly doubles your speed, and why it is a latency optimization at least as much as a memory one.

### Quality cost of quantization

| Format | Size vs bf16 | Typical benchmark loss | Use when |
|---|---|---|---|
| bf16 | 100% | baseline | It fits |
| 8-bit | 50% | < 1% | Marginal fit |
| 4-bit NF4 | 25% | 1–3% | Standard for consumer GPUs |
| 3-bit | ~19% | 5–10% | Desperate |
| 2-bit | ~13% | 15%+ | Almost never — a smaller model is better |

The degradation is not uniform. It concentrates in long-chain reasoning and exact recall; general chat and summarization barely move. **A well-run 7B beats a 2-bit 70B**, essentially always.

### The honest ceiling

Kumaru cannot make a 7B model as good as Claude. No configuration will. What it does is remove every *engineering* obstacle between you and the best model your hardware can hold, and guarantee that the day you get better hardware, the upgrade is one line of YAML.

---

## 2. Serving throughput

### The mechanism

The `transformers` backend generates one request at a time. The `openai_compat` backend forwards to a dedicated serving engine. Same interface; ~10–24× the throughput.

### Why continuous batching is such a large win

Follow the bandwidth argument from §1. Reading the weights costs the same whether you are decoding for one sequence or sixty-four — it is one pass over the weights *per step*, not per sequence. So the marginal cost of an additional concurrent user is nearly zero until you saturate compute or run out of KV-cache memory.

Two techniques make that theoretical win real:

- **Continuous batching.** A new request joins the in-flight batch at the next decode step, instead of waiting for the current batch to drain. No head-of-line blocking.
- **PagedAttention.** The KV cache is stored in fixed-size pages rather than one contiguous slab per sequence. Naive allocation reserves `max_seq_len` per request and wastes 60–80% of it; paging cuts waste to a few percent, and the recovered memory becomes a bigger batch.

### Measured shape

| Setup | Concurrent users | Aggregate throughput | Per-user latency |
|---|---|---|---|
| `transformers`, 1 GPU | 1 | ~40 tok/s | good |
| `transformers`, 1 GPU | 8 | ~45 tok/s | terrible — they queue |
| vLLM, 1 GPU | 1 | ~45 tok/s | good |
| vLLM, 1 GPU | 8 | ~280 tok/s | good |
| vLLM, 1 GPU | 64 | ~900 tok/s | acceptable |

### How to take this step

```powershell
# on Linux or WSL — vLLM has no native Windows build
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.90
```

```powershell
kumaru serve -c vllm
```

Nothing in Kumaru changes. `configs/vllm.yaml` just points `base_url` at the engine.

### Beyond one GPU

These are vLLM flags, not Kumaru changes:

| Situation | Flag |
|---|---|
| Model too big for one GPU | `--tensor-parallel-size 4` |
| Model too big for one node | `--pipeline-parallel-size 2` |
| More total throughput needed | Run N replicas behind a load balancer |

The last one works because **`ChatAgent` is stateless** — the client sends the full conversation on every request (`agent/chat.py`). No session affinity, no shared cache, no sticky routing. Horizontal scaling is `docker compose up --scale`.

### Admission control, and why it is not optional

`server/app.py` bounds in-flight generations with a semaphore:

```yaml
server:
  max_concurrent_requests: 4
```

On one GPU, admitting unlimited concurrency makes every request slower and eventually triggers a CUDA OOM that kills the process. Bounded admission converts a hard crash into graceful queueing. That is the difference between a service that degrades and one that dies.

---

## 3. Context length

### The mechanism

`agent/memory.py` defines a `MemoryPolicy` interface. Three policies ship; the interface is the extension point.

### The physics

Attention is `O(n²)` in compute and the KV cache is `O(n)` in memory. Concretely, for Qwen2.5-7B (28 layers, 4 KV heads, 128 head dim, bf16):

```
kv_bytes = 2 × 28 × 4 × 128 × 2 × seq_len = 57,344 bytes per token
```

| Context | KV cache | Notes |
|---|---|---|
| 4k | 0.23 GB | comfortable |
| 16k | 0.92 GB | the `local-gpu` default |
| 32k | 1.84 GB | fine on 24 GB |
| 128k | 7.3 GB | needs 4-bit weights to co-exist |

Grouped-query attention is why these numbers are small. With full multi-head attention (28 KV heads instead of 4) every figure would be **7× larger**, and 128k would be impossible on a consumer card.

### The ladder

| Stage | Policy | How it works | Cost |
|---|---|---|---|
| now | `sliding_window` | Keep system prompt + newest turns, drop the middle | Older turns are simply gone |
| next | `summarize` | Compress the dropped middle into a paragraph, re-inject | One extra model call per compression |
| then | `retrieve` (RAG) | Embed all history, retrieve only relevant turns | An embedding model + a vector index |
| eventually | Long-context model | Qwen2.5 supports 128k with YaRN scaling | Memory, and "lost in the middle" degradation |

Adding a policy is additive — one class plus one dict entry in `POLICIES`:

```python
class SummarizingPolicy(MemoryPolicy):
    name = "summarize"
    def apply(self, messages, budget, count_tokens): ...

POLICIES["summarize"] = SummarizingPolicy
```

Then `agent.memory_policy: summarize` in config. `ChatAgent` is untouched.

### The judgement call

More context is not automatically better. Models attend most reliably to the beginning and end of their window — the "lost in the middle" effect. **Retrieving 2,000 relevant tokens beats stuffing 100,000 mediocre ones**, and it is far cheaper. Reach for a longer window only after retrieval has failed you.

---

## 4. Capability

### The mechanism

Two registries in `core/registry.py`:

```python
BACKENDS: Registry = Registry("backend", entry_point_group="kumaru.backends")
TOOLS:    Registry = Registry("tool",    entry_point_group="kumaru.tools")
```

Components announce themselves with a decorator. The rest of the system only ever asks by name, resolved from config at runtime.

### Why this specific design

The alternative is a dispatch chain:

```python
if backend == "transformers": ...
elif backend == "vllm": ...      # every new idea edits a core file
```

That means every new capability is a *surgical* change to code that already works, with a corresponding chance of breaking it. Over years of absorbing new research, the file becomes unmaintainable and the project stops absorbing anything.

With a registry, a new capability is a **new file plus one decorator**. Nothing existing is touched. That is the mechanical property that lets a codebase keep taking on new ideas indefinitely.

### Third-party extension

`load_entry_points()` discovers components from *other installed pip packages*:

```toml
# in someone else's package
[project.entry-points."kumaru.backends"]
mamba = "kumaru_mamba:MambaBackend"
```

`pip install kumaru-mamba` and it appears in your UI dropdown. No fork, no patch, no change to this repository. One broken plugin is logged and skipped rather than crashing startup.

### The roadmap this enables

Each of these is additive. None requires touching the agent loop:

| Capability | Where it plugs in |
|---|---|
| Web search, code execution, file I/O | `@TOOLS.register(...)` |
| RAG over your documents | A `MemoryPolicy`, plus a retrieval tool |
| Multi-agent orchestration | A `Backend` that fans out to sub-agents |
| Structured output / JSON mode | `GenerationParams` field + backend support |
| Vision input | `Message.content` becomes a union type |
| New architectures (Mamba, MoE, diffusion LMs) | A new `Backend` |

---

## 5. Training scale

### The ladder

| Stage | Method | Trainable params (7B) | VRAM | Hardware |
|---|---|---|---|---|
| 1 | QLoRA | ~40M (0.5%) | ~8–11 GB | **1× RTX 3090 — you are here** |
| 2 | LoRA bf16 | ~40M | ~22 GB | 1× 24 GB (tight) |
| 3 | Full fine-tune | 7.6B | ~95 GB | 2× A100 80GB |
| 4 | FSDP / ZeRO-3 | 7.6B sharded | ~12 GB/GPU × 8 | 8× A100 |
| 5 | Multi-node FSDP | 70B+ | cluster | 32+ GPUs, InfiniBand |
| 6 | Pretrain from scratch | all | cluster-months | 1000+ GPUs |

### Where the memory actually goes

This table is the single most useful thing to internalize about training:

| Component | Full fine-tune (7B) | QLoRA (r=16) |
|---|---|---|
| Base weights | 15.2 GB (bf16) | 4.5 GB (4-bit NF4) |
| Gradients | 15.2 GB | 0.08 GB |
| Adam moments (2× fp32) | 60.8 GB | 0.32 GB |
| Activations (2k seq, bs 1) | ~4 GB | ~2.5 GB (checkpointed) |
| **Total** | **~95 GB** | **~7.5 GB** |

**Optimizer state dominates, and it scales with *trainable* parameters, not total ones.** LoRA attacks exactly that term. That single observation is why fine-tuning a 7B model on a gaming GPU is possible at all.

### Why the same loop scales

`train/pretrain_nano.py` is written for `torchrun` from the start:

```powershell
# 1 GPU
kumaru train configs/train/pretrain-nano.yaml

# 8 GPUs, same file, same code
torchrun --nproc_per_node=8 -m kumaru.train.pretrain_nano configs/train/pretrain-nano.yaml

# multiple nodes
torchrun --nnodes=4 --nproc_per_node=8 --rdzv_endpoint=$MASTER:29500 -m kumaru.train.pretrain_nano cfg.yaml
```

Data-parallel training works because **gradients are linear**: each rank computes gradients on its own shard, an all-reduce averages them, and every rank applies an identical update. The result is mathematically equivalent to one giant batch. That is the entire idea behind scaling to hundreds of GPUs.

`DistEnv` in that file reads `RANK`/`WORLD_SIZE`; with one GPU, `world_size == 1` and the same code path runs. There is no separate "distributed mode" to maintain or get wrong.

### When data-parallel stops being enough

DDP requires the full model plus optimizer state on each GPU. When that stops fitting:

| Problem | Technique | What it shards |
|---|---|---|
| Optimizer state too big | ZeRO-1 | Optimizer state |
| \+ gradients too big | ZeRO-2 | \+ gradients |
| \+ model too big | ZeRO-3 / FSDP | \+ parameters |
| A single layer too big | Tensor parallel | Individual matrices (needs NVLink) |
| Too many layers for one node | Pipeline parallel | Layer groups |

FSDP is the natural next step and requires wrapping the model in `FullyShardedDataParallel` instead of `DistributedDataParallel` — a handful of lines in `pretrain_nano.py`, deliberately kept as the only distributed-strategy-specific code in the file.

### What compute buys you

Chinchilla-optimal training is roughly **20 tokens per parameter**:

| Model | Optimal tokens | Compute (FLOPs) | On 8× A100 |
|---|---|---|---|
| 101M (the nano default) | 2B | 1.2e18 | ~1 hour |
| 1B | 20B | 1.2e20 | ~2 days |
| 7B | 140B | 5.9e21 | ~3 months |
| 70B | 1.4T | 5.9e23 | ~25 years |

The 70B row is why nobody pretrains a frontier model on their own hardware, and why the fine-tuning path is the one that matters for you. Kumaru's from-scratch trainer exists to make the mechanisms legible and to be *correct* at any scale — not to pretend the last row is reachable.

---

## 6. Data scale

| Stage | Format | Size | Mechanism |
|---|---|---|---|
| now | JSONL in memory | < 1 GB | `load_chat_jsonl()` |
| next | Memory-mapped tokens | < 1 TB | `pack_corpus_to_memmap()` — already implemented |
| then | Streaming datasets | unbounded | `datasets.load_dataset(streaming=True)` |
| cluster | Pre-tokenized shards | petabytes | One shard per rank, `webdataset`/`mosaic` |

`pack_corpus_to_memmap()` writes a flat `uint32` array and `PackedTokenDataset` reads slices from it. The OS pages in only what you touch, so the same code trains on a 10 MB toy file or a 200 GB corpus — with no loading delay at startup and no change to the training loop.

One detail that matters: the array is `uint32`, not `uint16`. Modern vocabularies exceed 65,536 tokens (Qwen2.5 is ~152k). `uint16` would silently wrap token ids, and the only symptom would be a model that trains smoothly and produces nonsense.

---

## 7. Evaluation scale

Every claim above is meant to be measurable:

```powershell
kumaru eval --suite speed   # time-to-first-token, tok/s, latency
kumaru eval --suite smoke   # quality regression tripwire
kumaru eval --suite my_cases.jsonl --output results.json
```

| Stage | Method | Signal quality |
|---|---|---|
| now | 6-case smoke suite, substring matching | Catches catastrophic regressions |
| next | 100+ domain cases as JSONL | Catches real regressions |
| then | LLM-as-judge scoring | Catches quality changes, needs calibration |
| then | Pairwise A/B with human raters | Ground truth, slow and expensive |
| always | Eval gate in CI | Prevents regressions reaching main |

The `Case` / `CaseResult` structures in `eval/harness.py` already accommodate a judge-model scorer; adding one means adding a scoring function, not restructuring the harness.

**Why this axis is listed last but matters most:** without measurement, every other axis is guesswork. You will believe every change you make is an improvement, because you chose it. A harness that runs in thirty seconds is the difference between engineering and vibes.

---

## What to do next, in order

1. **Establish a baseline.** `kumaru eval --suite smoke --output baseline.json`. Everything else is relative to this.
2. **Fix quality before speed.** A fast wrong answer is worthless. Try a bigger model or a better system prompt first.
3. **Fix latency before throughput.** Until you have more than two concurrent users, throughput is not your problem.
4. **Move to vLLM when you get a third user.** Largest single win available, and it costs one config file.
5. **Fine-tune only after prompting plateaus.** And run `configs/train/lora-small.yaml` first to validate the pipeline in five minutes.
6. **Add retrieval before adding context length.** Cheaper, more accurate, and updatable.
7. **Add tools last.** They multiply capability and they multiply failure modes.

---

## Related reading

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — why the seams are where they are
- [`TRAINING.md`](TRAINING.md) — the training walkthrough
- [`howto/src/kumaru/backends/base.md`](howto/src/kumaru/backends/base.md) — the backend contract
- [`howto/src/kumaru/core/registry.md`](howto/src/kumaru/core/registry.md) — the plugin mechanism
