# Scalability

> You asked what "scalable" actually means for this system. This document answers it precisely: every ceiling, what you hit first, and what the next step is.

Scalability is not one property. It is **seven independent axes**, each with its own ceiling, its own warning sign, and its own next move. The design goal of Kumaru is that advancing along any axis is a *configuration* change or an *additive* change — never a rewrite.

| # | Axis | You hit this when… | Next move |
|---|---|---|---|
| 1 | [Model scale](#1-model-scale) | Answers aren't good enough | A newer or bigger model (MoE), thinking, better quantization |
| 2 | [Serving throughput](#2-serving-throughput) | More simultaneous users than the engine has slots | More slots, then vLLM, then replicas |
| 3 | [Context length](#3-context-length) | Conversations get truncated | Summarization, then retrieval |
| 4 | [Capability](#4-capability) | The model needs to *do* things | Tool registry, plugins |
| 5 | [Training](#5-training-scale) | LoRA isn't enough | Full FT → FSDP → multi-node |
| 6 | [Data](#6-data-scale) | Datasets exceed RAM | Memmap, streaming, sharding |
| 7 | [Evaluation](#7-evaluation-scale) | You can't tell if changes helped | `kumaru bench`, then full test sets, harder benchmarks, LLM judges, CI gates |

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

So scaling the model is a config change. Compare `configs/qwen25-7b-gguf.yaml` with `configs/qwen36.yaml`: apart from each vendor's recommended sampling settings, the upgrade is these lines:

```yaml
backend:
  name: llamacpp_server
  model: unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS   # was Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
  max_context_length: 131072                      # was 32768
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

**Your RTX 3090 (24 GB) ceiling, for dense models:** a 32B at 4-bit with an 8k context, or a 7B at bf16 with 16k.

**The `qwen36` default goes past that.** It holds 35B parameters at ~4 bits *and* a 131k context in ~19.7 GiB (measured), and generates ~110 tok/s. A dense 32B at the same bits per weight would be about as big on disk, but its speed ceiling would be under 60 tok/s, and its KV cache at 131k would need 32 GiB on its own. Two architectural ideas make the difference: [mixture of experts](#mixture-of-experts) for speed and [hybrid attention](#hybrid-attention) for context.

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

### Mixture of experts

A dense model reads every weight for every token. A mixture-of-experts (MoE) model splits each layer's feed-forward block into many small "experts" and adds a router that picks a few per token. Qwen3.6-35B-A3B has 256 experts per layer and uses 8 of them, plus one shared expert, for each token. Two sizes describe it, and they govern different things:

- **Total parameters (35B) set memory and knowledge.** The router may pick any expert for the next token, so all of them must be loaded. What a model knows grows with its total size.
- **Active parameters (~3B) set speed.** Only these are read for a given token, so the bandwidth formula above uses *active* bytes.

```
bits per weight ≈ 17.7 GB file × 8 ÷ 35e9 params   ≈ 4.05
bytes per token ≈ 3e9 active × 4.05 ÷ 8            ≈ 1.5 GB
speed ceiling   ≈ 936 GB/s ÷ 1.5 GB                 ≈ 600 tok/s
measured        ≈ 110 tok/s, one request, RTX 3090
```

The measured speed is under a fifth of the ceiling, well below the 50–70% that dense models reach. Each token's work is split into small expert matrices across 40 layers, plus routing, so fixed per-step overheads weigh more than raw bandwidth. New llama.cpp builds keep narrowing that gap, and picking one up is a download ([below](#absorbing-the-next-model)).

**System RAM becomes a usable tier.** Since a token touches only 8 of 256 experts, the expert weights can stay in system RAM and run on the CPU while attention and the shared parts stay on the GPU. `llama_server.n_cpu_moe: N` does this for the first N layers and frees ~0.37 GiB of VRAM per layer. Measured here with `n_cpu_moe: 18` and a 64k context: 11.9 GiB of VRAM at ~27 tok/s, with room left for a game. Offloading part of a dense model costs far more, because every offloaded weight is read for every token. For MoE models, more system RAM is a real upgrade path.

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

No configuration of Kumaru makes a model that fits on one consumer GPU as good as Claude. Frontier models are built with far more training compute and post-training data than any open model you can run locally (exact figures aren't public). What Kumaru does instead:

- removes every *engineering* obstacle between you and the best model your hardware can hold;
- makes each upgrade a configuration change, never a rewrite;
- measures each upgrade ([§7](#7-evaluation-scale)), so you know where you stand instead of guessing.

### Absorbing the next model

Open models keep improving, and new architectures usually reach llama.cpp first. Taking one up is three steps:

1. **Engine, only if the architecture is new.** `.\scripts\get-llama-server.ps1 -Version bNNNNN` installs a newer llama.cpp build, and later plain runs keep it. Builds are tagged `b` plus a number on the llama.cpp releases page.
2. **Profile.** Copy `configs/qwen36.yaml`, set `backend.model` to the new `org/repo:QUANT`, and copy the sampling settings from the model card.
3. **Verdict.** Repeat your baseline `kumaru bench run …` with `-c` set to the new profile, then run `kumaru bench compare <baseline> <new>`. Switch on `B better`, or on `no clear difference` if the new model is faster or smaller.

---

## 2. Serving throughput

### The mechanism

How well simultaneous requests share the GPU is decided inside the backend:

| Backend | Simultaneous requests | Runs on |
|---|---|---|
| `transformers`, `llamacpp` (in-process) | No batching; meant for one user | anywhere |
| `llamacpp_server` (the `qwen36` default) | `llama_server.parallel` slots (4), batched into one forward pass per step | Windows, Linux, macOS |
| `openai_compat` → vLLM or SGLang | Dozens to hundreds, batched | Linux or WSL |

The server, agent and UI cannot tell them apart. vLLM's authors measured up to 24× the throughput of serving with plain Hugging Face `transformers`.

### Why continuous batching is such a large win

Follow the bandwidth argument from §1. Reading the weights costs the same whether you are decoding for one sequence or sixty-four — it is one pass over the weights *per step*, not per sequence. So the marginal cost of an additional concurrent user is nearly zero until you saturate compute or run out of KV-cache memory.

Two techniques make that theoretical win real:

- **Continuous batching.** A new request joins the in-flight batch at the next decode step, instead of waiting for the current batch to drain. No head-of-line blocking.
- **PagedAttention.** The KV cache is stored in fixed-size pages rather than one contiguous slab per sequence. Naive allocation reserves `max_seq_len` per request and wastes 60–80% of it; paging cuts waste to a few percent, and the recovered memory becomes a bigger batch.

### What batching buys, measured on this machine

Output tokens per second, all requests together, on the RTX 3090. The "bench" rows are `tokens/answer × answers/min ÷ 60` from `kumaru bench report`, so they include prompt processing and the end-of-run tail. A game was open on the same GPU during them.

| Model | Placement | Requests at once | Output tok/s |
|---|---|---|---|
| Qwen3.6-35B-A3B | all on GPU | 1 | ~110 |
| Qwen3.6-35B-A3B | `n_cpu_moe: 18`, 64k context | 1 | ~27 |
| Qwen3.6-35B-A3B | `n_cpu_moe: 18`, 64k context | 4 (bench) | ~47–49 |
| Qwen2.5-7B, Q4_K_M | all on GPU | 4 (bench) | ~175–195 |

Four requests at once gave the MoE about 1.8× its single-request speed, not 4×. Batching helps MoE less than dense models: four tokens decoded together can route to up to 32 different experts, so "read the weights once, use them for every sequence" holds only for the shared parts.

### The classic shape, for a dense 7B (illustrative)

Order-of-magnitude figures to show the shape, not measurements from this machine:

| Setup | Concurrent users | Aggregate throughput | Per-user latency |
|---|---|---|---|
| `transformers`, 1 GPU | 1 | ~40 tok/s | good |
| `transformers`, 1 GPU | 8 | ~45 tok/s | terrible — they queue |
| vLLM, 1 GPU | 1 | ~45 tok/s | good |
| vLLM, 1 GPU | 8 | ~280 tok/s | good |
| vLLM, 1 GPU | 64 | ~900 tok/s | acceptable |

### How to take this step

On Windows you already have batching: `llamacpp_server` runs 4 slots. Raise `llama_server.parallel` first. With `kv_unified: true` the slots share one KV pool of `max_context_length` tokens, so more slots means less room per conversation when they are all busy. vLLM is the step after that, when you need dozens of concurrent users:

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

With `llamacpp_server` there are two knobs: the `qwen36` profile admits 8 requests (`server.max_concurrent_requests`), and the engine runs 4 at a time (`llama_server.parallel`) and queues the rest.

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

### Hybrid attention

`qwen36` goes further. Only 10 of its 40 layers use full attention with a growing KV cache. The other 30 are Gated DeltaNet ("linear attention") layers, which carry a fixed-size state no matter how long the conversation gets:

```
kv_bytes = 10 layers × 2 KV heads × 256 dim × 2 (K,V) × 2 bytes = 20 KiB per token
```

| Context (tokens) | Qwen2.5-7B (56 KiB/token) | Qwen3.6-35B-A3B (20 KiB/token) |
|---|---|---|
| 32,768 | 1.75 GiB | 0.63 GiB |
| 131,072 | 7.0 GiB | 2.5 GiB |

A model about five times larger needs about a third of the cache per token, and the 30 linear layers process a long prompt in linear rather than quadratic time. Expect more of this. Research results arrive as architecture changes, and they reach you as a bigger usable window in the same VRAM.

### The ladder

| Stage | Policy | How it works | Cost |
|---|---|---|---|
| now | Long-context model | `qwen36` keeps up to 64k tokens of history (`agent.max_history_tokens`) in a 131k window | "Lost in the middle" degradation |
| now | `sliding_window` | Past that budget, keep system prompt + newest turns, drop the middle | Older turns are simply gone |
| next | `summarize` | Compress the dropped middle into a paragraph, re-inject | One extra model call per compression |
| then | `retrieve` (RAG) | Embed all history, retrieve only relevant turns | An embedding model + a vector index |

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

Three registries in `core/registry.py`:

```python
BACKENDS:   Registry = Registry("backend",   entry_point_group="kumaru.backends")
TOOLS:      Registry = Registry("tool",      entry_point_group="kumaru.tools")
BENCHMARKS: Registry = Registry("benchmark", entry_point_group="kumaru.benchmarks")
```

Components announce themselves with a decorator. Built-ins with heavy imports use a one-line `register_lazy(...)` instead, so torch or `datasets` loads only when that component is selected. The rest of the system only ever asks by name, resolved from config at runtime.

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

`pip install kumaru-mamba` and it appears in your UI dropdown. No fork, no patch, no change to this repository. One broken plugin is logged and skipped rather than crashing startup. The `kumaru.benchmarks` group works the same way, so a private evaluation can ship as its own package.

### The roadmap this enables

Each of these is additive. None requires touching the agent loop:

| Capability | Where it plugs in |
|---|---|
| Web search, code execution, file I/O | `@TOOLS.register(...)` |
| RAG over your documents | A `MemoryPolicy`, plus a retrieval tool |
| Multi-agent orchestration | A `Backend` that fans out to sub-agents |
| Structured output / JSON mode | `GenerationParams` field + backend support |
| Vision input | `Message.content` becomes a union type |
| New architectures llama.cpp supports (MoE and hybrid attention arrived this way) | A newer `llama-server` build plus a profile ([§1](#absorbing-the-next-model)) |
| Architectures it doesn't serve | A new `Backend`, or `openai_compat` pointed at an engine that does |
| A new benchmark | A `Benchmark` subclass plus one `register_lazy` line ([GUIDE.md](../GUIDE.md#add-a-benchmark)) |

Thinking was built exactly the way the structured-output row describes. It added two `GenerationParams` fields (`thinking`, `thinking_budget`), a translation into each engine's dialect, a `thinking_mode` sampling preset in config, and the UI's Think button. The agent loop only learned to apply the preset and to leave room for longer replies.

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

**Mixture-of-experts models change one row of that table.** LoRA keeps optimizer state small, but frozen base weights scale with *total* parameters. A 35B MoE at 4 bits is ~18 GB of weights before any activations, even though only ~3B are active per token. Kumaru's fine-tuning path (PyTorch + PEFT) therefore targets dense models like Qwen2.5-7B on a 24 GB card; fine-tuning `qwen36` itself needs a bigger GPU.

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

### Distillation: a big model's compute, a small model's speed

A large teacher answers questions slowly, a grader keeps its correct answers, and a small student trained on them answers fast ([TRAINING.md Part 3](TRAINING.md#part-3--distillation)). The other axes then feed the student:

- **A stronger teacher** (§1) raises what the data can teach.
- **More answers per question.** Rerun the teacher with another `--seed` and pass every run to `kumaru distill build`, which keeps up to 4 distinct correct answers per question. Yuan et al. (2023) found that distinct correct solutions help, and help weaker students most.
- **Serving throughput** (§2) sets how fast the data is written: an estimated 16 hours for GSM8K's 7,473 training questions on this machine, beside a game.
- **Any task with a grader and a training split** can be a source, not just GSM8K ([how](howto/src/kumaru/train/distill.md#how-to-extend-it)).

The limit is the student. Li et al. (2025) found that students of 3B parameters or fewer often learn less from a strong teacher's long reasoning than from short reasoning, so a stronger teacher alone does not guarantee a better small student. Measure each step against a control arm, as Part 3 does.

---

## 6. Data scale

| Stage | Format | Size | Mechanism |
|---|---|---|---|
| now | JSONL in memory | < 1 GB | `load_chat_jsonl()` |
| next | Memory-mapped tokens | < 1 TB | `pack_corpus_to_memmap()` — already implemented |
| then | Streaming datasets | unbounded | `datasets.load_dataset(streaming=True)` |
| cluster | Pre-tokenized shards | petabytes | One shard per rank, `webdataset`/`mosaic` |

`pack_corpus_to_memmap()` writes a flat `uint32` array and `PackedTokenDataset` reads slices from it. The OS pages in only what you touch, so the same code trains on a 10 MB toy file or a 200 GB corpus — with no loading delay at startup and no change to the training loop. Packing streams too: each chunk of text is tokenized and appended to disk before the next is read, so packing memory stays flat (about 0.2 GB) however large the corpus is, and the packed file takes 4 bytes per token.

One detail that matters: the array is `uint32`, not `uint16`. Modern vocabularies exceed 65,536 tokens (Qwen2.5 is ~152k). `uint16` would silently wrap token ids, and the only symptom would be a model that trains smoothly and produces nonsense.

---

## 7. Evaluation scale

Every claim above is meant to be measurable. Two commands do the measuring:

```powershell
kumaru eval --suite smoke -c qwen36    # 6 cases: did anything break?
kumaru eval --suite speed -c qwen36    # time-to-first-token, tok/s, latency
kumaru bench run golden gsm8k mmlu_pro ifeval -c qwen36    # did it get smarter?
kumaru bench compare runs\bench\<before> runs\bench\<after>
```

`kumaru eval` is a tripwire. `kumaru bench` is a measuring instrument: public benchmarks plus your own golden questions, validated graders, confidence intervals, and a paired significance test, so that "B is better" means something. [BENCHMARKS.md](BENCHMARKS.md) is its manual and lab notebook.

### Precision costs questions, quadratically

The 95% error bar on an accuracy *p* measured on *n* questions is about `±1.96 × √(p(1−p)/n)`:

| Questions | Error bar at 75% accuracy | Enough to see | Example |
|---|---|---|---|
| 27 | ±16 points | Only disasters | the golden set |
| 100 | ±8.5 points | Model swaps | `kumaru bench` default (`-n 100`) |
| 1,000 | ±2.7 points | Prompt and sampling changes | `-n 1000` |
| 12,032 | ±0.8 points | Small fine-tuning effects | MMLU-Pro with `--full` |

Halving the error bar takes four times the questions. Two things make each question count for more:

- **Pairing.** Both runs answer the same questions with the same per-question seeds, and McNemar's test looks only at the questions where they disagree. It detects differences that two overlapping error bars would hide.
- **Validated graders.** A grader that misreads 5% of answers adds an error that no sample size removes. That is why `scripts\check_graders.py` and the IFEval reference cross-check exist ([BENCHMARKS.md §5](BENCHMARKS.md#5-are-the-graders-right)).

### The ladder

| Stage | Method | Detects | Status |
|---|---|---|---|
| tripwire | `kumaru eval --suite smoke` | Catastrophic breakage | shipped |
| now | `kumaru bench`, 100 questions per benchmark, McNemar + Holm | ~10-point differences | shipped |
| next | `kumaru bench run … --full`: every test question | 1–2 point differences on the large sets | shipped; MMLU-Pro takes 120× as long |
| then | Harder benchmarks, as current ones saturate | Differences near the frontier | one new file each |
| then | Pairwise LLM-as-judge for open-ended answers | Quality no answer key captures | a stronger judge, calibrated against your own ratings |
| always | A CI gate on `bench compare` | Regressions reaching main | compute on every pull request |

Every stage is additive. A judge is another kind of grader, so it arrives as a new benchmark. Runs resume where they stopped (`-o <run folder>`), so a `--full` run can span several evenings.

### Two traps that grow with scale

- **Contamination.** Public test sets leak into training data, so part of a newer model's score can be memory. Questions you wrote yourself and never published can't have leaked ([BENCHMARKS.md §8](BENCHMARKS.md#8-your-own-golden-questions)); treat them as the tie-breaker.
- **Saturation.** When every candidate scores in the 90s, a benchmark stops telling them apart. GSM8K is nearly there for current models; MMLU-Pro exists because MMLU got there first.

**Why this axis is listed last but matters most:** without measurement, every other axis is guesswork. You will believe every change you make is an improvement, because you chose it. The first benchmark run in [BENCHMARKS.md §6.1](BENCHMARKS.md#61-the-truncation-confound) is the cautionary tale: a token cap silently cut off up to 16% of one model's answers, so the first comparison partly measured the cap rather than the model. [§10 there](BENCHMARKS.md#10-where-evaluation-goes-next) lists the next concrete steps.

---

## What to do next, in order

1. **Establish a baseline.** `kumaru bench run golden gsm8k mmlu_pro ifeval -c qwen36`, and keep the run folder. Every later change is a `kumaru bench compare` against it.
2. **Fix quality before speed.** A fast wrong answer is worthless. Try a better model, Think, or a better system prompt first, and measure each.
3. **Fix latency before throughput.** Until you have more than two concurrent users, throughput is not your problem.
4. **Use the engine's slots before adding hardware.** `llamacpp_server` already batches 4 requests. Move to vLLM (Linux or WSL) when you need dozens; it costs one config file.
5. **Fine-tune only after prompting plateaus.** And run `configs/train/lora-small.yaml` first to validate the pipeline in five minutes.
6. **Add retrieval before adding context length.** Cheaper, more accurate, and updatable.
7. **Add tools last.** They multiply capability and they multiply failure modes.

---

## Related reading

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — why the seams are where they are
- [`BENCHMARKS.md`](BENCHMARKS.md) — measuring whether a change helped, with results
- [`howto/src/kumaru/backends/llamacpp_server.md`](howto/src/kumaru/backends/llamacpp_server.md) — the engine behind `qwen36`, and its VRAM knobs
- [`TRAINING.md`](TRAINING.md) — the training walkthrough
- [`howto/src/kumaru/backends/base.md`](howto/src/kumaru/backends/base.md) — the backend contract
- [`howto/src/kumaru/core/registry.md`](howto/src/kumaru/core/registry.md) — the plugin mechanism
