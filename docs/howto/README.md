# How-to index

One page per file. Each page answers the same seven questions: what it does, why it exists, the mental model, the public API, how to use it, how to extend it, and how it fails.

**Not sure where to start?** Read the source in [the order suggested in ARCHITECTURE.md](../ARCHITECTURE.md#reading-order-for-the-source), with each file's page open beside it.

---

## Core — the contracts everything depends on

| File | Page | Read it when |
|---|---|---|
| `src/kumaru/__init__.py` | [page](src/kumaru/__init__.md) | You want the public import surface |
| `src/kumaru/cli.py` | [page](src/kumaru/cli.md) | Adding a command, or a flag behaves oddly |
| `src/kumaru/core/__init__.py` | [page](src/kumaru/core/__init__.md) | Understanding the layering rule |
| `src/kumaru/core/types.py` | [page](src/kumaru/core/types.md) | Anything to do with messages or generation params |
| `src/kumaru/core/errors.py` | [page](src/kumaru/core/errors.md) | Raising a new error, or mapping one to HTTP |
| `src/kumaru/core/config.py` | [page](src/kumaru/core/config.md) | Adding a setting, or a setting is being ignored |
| `src/kumaru/core/registry.py` | [page](src/kumaru/core/registry.md) | Adding a plugin, or writing an extension package |
| `src/kumaru/core/logging.py` | [page](src/kumaru/core/logging.md) | Changing log format, or shipping logs off-box |

## Backends — inference engines behind one interface

| File | Page | Read it when |
|---|---|---|
| `backends/__init__.py` | [page](src/kumaru/backends/__init__.md) | Orienting yourself in the backend layer |
| `backends/base.py` | [page](src/kumaru/backends/base.md) | **Writing a new backend** — start here |
| `backends/echo.py` | [page](src/kumaru/backends/echo.md) | You want the smallest complete example |
| `backends/transformers_backend.py` | [page](src/kumaru/backends/transformers_backend.md) | Running locally on a GPU; tuning VRAM or speed |
| `backends/llamacpp_backend.py` | [page](src/kumaru/backends/llamacpp_backend.md) | Low VRAM, CPU-only, or GGUF models |
| `backends/llamacpp_server.py` | [page](src/kumaru/backends/llamacpp_server.md) | **The engine behind `qwen36`**: thinking, batching, sharing the GPU |
| `backends/openai_compat.py` | [page](src/kumaru/backends/openai_compat.md) | Moving to vLLM, Ollama, or a remote endpoint |

## Agent — prompt, memory, tools

| File | Page | Read it when |
|---|---|---|
| `agent/__init__.py` | [page](src/kumaru/agent/__init__.md) | Orienting yourself in the agent layer |
| `agent/chat.py` | [page](src/kumaru/agent/chat.md) | Changing how a turn is orchestrated |
| `agent/prompt.py` | [page](src/kumaru/agent/prompt.md) | Tuning answer quality, or injecting context |
| `agent/memory.py` | [page](src/kumaru/agent/memory.md) | Conversations get truncated; adding summarization or RAG |
| `agent/knowledge.py` | [page](src/kumaru/agent/knowledge.md) | Coding guidance; importing reference packs with `kumaru learn` |
| `agent/tools.py` | [page](src/kumaru/agent/tools.md) | **Adding a capability the model can call** |
| `agent/workspace.py` | [page](src/kumaru/agent/workspace.md) | The project folder: what the agent may list, read, create and edit |
| `agent/commands.py` | [page](src/kumaru/agent/commands.md) | Which programs the agent may run, and how they are contained |

## Server — the HTTP surface

| File | Page | Read it when |
|---|---|---|
| `server/__init__.py` | [page](src/kumaru/server/__init__.md) | Orienting yourself in the server layer |
| `server/app.py` | [page](src/kumaru/server/app.md) | Startup, lifecycle, metrics, concurrency limits |
| `server/schemas.py` | [page](src/kumaru/server/schemas.md) | Changing the public API contract |
| `server/routes_chat.py` | [page](src/kumaru/server/routes_chat.md) | **Every HTTP endpoint, documented** |

## Training

| File | Page | Read it when |
|---|---|---|
| `train/__init__.py` | [page](src/kumaru/train/__init__.md) | Deciding which training path you need |
| `train/config.py` | [page](src/kumaru/train/config.md) | **Every training YAML field, explained** |
| `train/data.py` | [page](src/kumaru/train/data.md) | Dataset format, loss masking, token packing |
| `train/distill.py` | [page](src/kumaru/train/distill.md) | **Distilling a large model into a small one**: building and auditing the data |
| `train/finetune_lora.py` | [page](src/kumaru/train/finetune_lora.md) | Running or debugging a LoRA fine-tune |
| `train/model_nano.py` | [page](src/kumaru/train/model_nano.md) | **What a transformer actually is**, component by component |
| `train/pretrain_nano.py` | [page](src/kumaru/train/pretrain_nano.md) | The training loop; scaling to multi-GPU |
| `train/runner.py` | [page](src/kumaru/train/runner.md) | Adding a training task |

## Evaluation

| File | Page | Read it when |
|---|---|---|
| `eval/__init__.py` | [page](src/kumaru/eval/__init__.md) | Orienting yourself |
| `eval/harness.py` | [page](src/kumaru/eval/harness.md) | Writing your own eval suite; adding an LLM judge |

## Benchmarks: how good is the model, and did a change help?

Results and method: [docs/BENCHMARKS.md](../BENCHMARKS.md).

| File | Page | Read it when |
|---|---|---|
| `eval/benchmarks/__init__.py` | [page](src/kumaru/eval/benchmarks/__init__.md) | Orienting yourself |
| `eval/benchmarks/base.py` | [page](src/kumaru/eval/benchmarks/base.md) | **Adding a benchmark** — start here |
| `eval/benchmarks/runner.py` | [page](src/kumaru/eval/benchmarks/runner.md) | Running, resuming and comparing; what a run folder holds |
| `eval/benchmarks/stats.py` | [page](src/kumaru/eval/benchmarks/stats.md) | Confidence intervals and significance tests, in plain terms |
| `eval/benchmarks/extract.py` | [page](src/kumaru/eval/benchmarks/extract.md) | A right answer was graded wrong |
| `eval/benchmarks/gsm8k.py` | [page](src/kumaru/eval/benchmarks/gsm8k.md) | Math word problems |
| `eval/benchmarks/mmlu_pro.py` | [page](src/kumaru/eval/benchmarks/mmlu_pro.md) | Knowledge and reasoning, 14 subjects |
| `eval/benchmarks/ifeval.py` | [page](src/kumaru/eval/benchmarks/ifeval.md) | Instruction following and its four scores |
| `eval/benchmarks/ifeval_checks.py` | [page](src/kumaru/eval/benchmarks/ifeval_checks.md) | One IFEval instruction type behaves oddly |
| `eval/benchmarks/humaneval.py` | [page](src/kumaru/eval/benchmarks/humaneval.md) | Code generation; what the code-execution guard does and does not stop |
| `eval/benchmarks/golden.py` | [page](src/kumaru/eval/benchmarks/golden.md) | **Writing your own questions**: the format and the graders |
| `data/bench/golden.jsonl` | [page](data/bench/golden.jsonl.md) | Your golden set: what each question tests |
| `scripts/check_graders.py` | [page](scripts/check_graders.py.md) | Proving the graders right against the datasets' answer keys |

## UI

| File | Page | Read it when |
|---|---|---|
| `ui/index.html` | [page](ui/index.html.md) | Adding a control to the sidebar |
| `ui/styles.css` | [page](ui/styles.css.md) | Changing the theme |
| `ui/app.js` | [page](ui/app.js.md) | Streaming, markdown rendering, client state |

## Configuration profiles

| File | Page | Targets |
|---|---|---|
| `configs/default.yaml` | [page](configs/default.yaml.md) | **Every setting, documented** |
| `configs/qwen36.yaml` | [page](configs/qwen36.yaml.md) | **Recommended**: Qwen3.6-35B-A3B on a 24 GB GPU, with thinking |
| `configs/qwen25-7b-gguf.yaml` | [page](configs/qwen25-7b-gguf.yaml.md) | The previous default, on the same engine: the benchmark baseline |
| `configs/local-gpu.yaml` | [page](configs/local-gpu.yaml.md) | 24 GB GPU |
| `configs/low-vram.yaml` | [page](configs/low-vram.yaml.md) | 8–12 GB GPU |
| `configs/cpu-only.yaml` | [page](configs/cpu-only.yaml.md) | No GPU |
| `configs/vllm.yaml` | [page](configs/vllm.yaml.md) | Production throughput |
| `configs/student.yaml` | [page](configs/student.yaml.md) | Benchmarking the distillation student, Qwen2.5-1.5B, in batches |
| `configs/test.yaml` | [page](configs/test.yaml.md) | No model, for tests |

## Training recipes and data

| File | Page | Purpose |
|---|---|---|
| `configs/train/lora-qwen.yaml` | [page](configs/lora-qwen.yaml.md) | QLoRA on 7B, 24 GB GPU |
| `configs/train/lora-small.yaml` | [page](configs/lora-small.yaml.md) | Five-minute pipeline smoke test |
| `configs/train/pretrain-nano.yaml` | [page](configs/pretrain-nano.yaml.md) | From-scratch transformer |
| `configs/train/distill-gsm8k.yaml` | [page](configs/distill-gsm8k.yaml.md) | **Distillation**: Qwen2.5-1.5B learns from the teacher's verified answers |
| `configs/train/distill-gsm8k-gold.yaml` | [page](configs/distill-gsm8k-gold.yaml.md) | Distillation's control arm: the same questions, human-written solutions |
| `configs/train/distill-gsm8k-0.5b.yaml` | [page](configs/distill-gsm8k-0.5b.yaml.md) | Distillation pilot on Qwen2.5-0.5B: measures time and memory first |
| `data/samples/chat_train.jsonl` | [page](configs/chat_train.jsonl.md) | **The SFT data format**, and how to build your own |
| `data/samples/corpus.txt` | [page](configs/corpus.txt.md) | Pretraining corpus format |

## Project root

| File | Page | Read it when |
|---|---|---|
| `pyproject.toml` | [page](pyproject.toml.md) | **Which extra to install**, adding a dependency, releasing a version, pytest/ruff/mypy settings |
| `.gitignore` | [page](.gitignore.md) | A file is unexpectedly ignored or not; sharing a benchmark run |

## Scripts and tests

| File | Page | Purpose |
|---|---|---|
| `scripts/setup.ps1` | [page](scripts/setup.ps1.md) | One-shot install |
| `scripts/get-llama-server.ps1` | [page](scripts/get-llama-server.ps1.md) | Installing or upgrading the llama.cpp server |
| `scripts/verify.ps1` | [page](scripts/verify.ps1.md) | Full self-check, layer by layer |
| `tests/` | [page](tests/README.md) | What the suite protects, and how to add to it |
