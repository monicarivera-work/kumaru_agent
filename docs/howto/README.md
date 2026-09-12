# How-to index

One page per file. Each page answers the same six questions: what it does, why it exists, the mental model, the public API, how to use it, how to extend it, and how it fails.

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
| `backends/openai_compat.py` | [page](src/kumaru/backends/openai_compat.md) | Moving to vLLM, Ollama, or a remote endpoint |

## Agent — prompt, memory, tools

| File | Page | Read it when |
|---|---|---|
| `agent/__init__.py` | [page](src/kumaru/agent/__init__.md) | Orienting yourself in the agent layer |
| `agent/chat.py` | [page](src/kumaru/agent/chat.md) | Changing how a turn is orchestrated |
| `agent/prompt.py` | [page](src/kumaru/agent/prompt.md) | Tuning answer quality, or injecting context |
| `agent/memory.py` | [page](src/kumaru/agent/memory.md) | Conversations get truncated; adding summarization or RAG |
| `agent/tools.py` | [page](src/kumaru/agent/tools.md) | **Adding a capability the model can call** |

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
| `train/finetune_lora.py` | [page](src/kumaru/train/finetune_lora.md) | Running or debugging a LoRA fine-tune |
| `train/model_nano.py` | [page](src/kumaru/train/model_nano.md) | **What a transformer actually is**, component by component |
| `train/pretrain_nano.py` | [page](src/kumaru/train/pretrain_nano.md) | The training loop; scaling to multi-GPU |
| `train/runner.py` | [page](src/kumaru/train/runner.md) | Adding a training task |

## Evaluation

| File | Page | Read it when |
|---|---|---|
| `eval/__init__.py` | [page](src/kumaru/eval/__init__.md) | Orienting yourself |
| `eval/harness.py` | [page](src/kumaru/eval/harness.md) | Writing your own eval suite; adding an LLM judge |

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
| `configs/local-gpu.yaml` | [page](configs/local-gpu.yaml.md) | 24 GB GPU |
| `configs/low-vram.yaml` | [page](configs/low-vram.yaml.md) | 8–12 GB GPU |
| `configs/cpu-only.yaml` | [page](configs/cpu-only.yaml.md) | No GPU |
| `configs/vllm.yaml` | [page](configs/vllm.yaml.md) | Production throughput |
| `configs/test.yaml` | [page](configs/test.yaml.md) | No model, for tests |

## Training recipes and data

| File | Page | Purpose |
|---|---|---|
| `configs/train/lora-qwen.yaml` | [page](configs/lora-qwen.yaml.md) | QLoRA on 7B, 24 GB GPU |
| `configs/train/lora-small.yaml` | [page](configs/lora-small.yaml.md) | Five-minute pipeline smoke test |
| `configs/train/pretrain-nano.yaml` | [page](configs/pretrain-nano.yaml.md) | From-scratch transformer |
| `data/samples/chat_train.jsonl` | [page](configs/chat_train.jsonl.md) | **The SFT data format**, and how to build your own |
| `data/samples/corpus.txt` | [page](configs/corpus.txt.md) | Pretraining corpus format |

## Scripts and tests

| File | Page | Purpose |
|---|---|---|
| `scripts/setup.ps1` | [page](scripts/setup.ps1.md) | One-shot install |
| `scripts/verify.ps1` | [page](scripts/verify.ps1.md) | Full self-check, layer by layer |
| `tests/` | [page](tests/README.md) | What the suite protects, and how to add to it |
