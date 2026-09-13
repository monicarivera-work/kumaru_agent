# Kumaru operator guide

Everything a software engineer needs to install, run, operate and debug Kumaru
on their own machine. Read top to bottom the first time; after that use it as a
reference.

---

## 1. Install

You need Python 3.10 or newer. Nothing else is required for the first run.

```bash
git clone <this repo> && cd kumaru_agent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

`pip install -e .` installs the package in editable mode: edits to `src/` take
effect on the next run, with no reinstall.

Check it:

```bash
kumaru --version
kumaru backends          # echo, llamacpp, openai_compat, transformers
```

## 2. First run (no model)

```bash
kumaru serve
```

Open <http://127.0.0.1:8000>. Type something. You get your own words back,
because the default backend is `echo`.

This is not a toy step — it proves the browser, the HTTP layer, SSE streaming,
memory and the CLI all work *before* a model is in the picture. When something
breaks later, `kumaru serve --backend echo` tells you in ten seconds whether
the problem is the model or everything else.

## 3. Choose a backend

| Backend | Needs | Use it when |
|---|---|---|
| `echo` | nothing | Smoke tests, UI work, CI. |
| `openai_compat` | a running server | You already use Ollama / LM Studio / llama.cpp / vLLM, or you want a remote model. **Start here.** |
| `llamacpp` | `pip install -e '.[llamacpp]'` + a `.gguf` file | No GPU, want a real model in-process. |
| `transformers` | `pip install -e '.[transformers]'` + torch | You have a GPU, or you want to load your own fine-tuned weights. |

### 3a. `openai_compat` with Ollama (recommended first real model)

```bash
# terminal 1
ollama serve
ollama pull llama3.1:8b

# terminal 2
kumaru serve -c configs/openai-compat.yaml
```

`configs/openai-compat.yaml` points at `http://localhost:11434/v1`. For
llama.cpp's `llama-server` use `http://localhost:8080/v1`; for vLLM,
`http://localhost:8000/v1` (and change Kumaru's own port so they do not clash).

For a hosted OpenAI-compatible endpoint, set `base_url` and export the key:

```bash
export KUMARU_API_KEY=...        # never put the key in the YAML file
```

### 3b. `llamacpp` — a real model with no GPU

```bash
pip install -e '.[llamacpp]'
mkdir -p models
# download a 4-bit GGUF (~5 GB for a 7-8B model) into models/
kumaru serve -c configs/cpu-only.yaml
```

Edit `backend.model` in the config to match your filename. Rough expectations
on a modern 8-core CPU: 5–15 tokens/second, ~5 GB RAM for a 7B Q4 model.

If you have a little VRAM, `configs/low-vram.yaml` offloads some layers:
raise `gpu_layers` until loading fails, then drop back by four.

### 3c. `transformers` — GPU, or your own fine-tune

```bash
pip install -e '.[transformers]'          # install a CUDA torch build first
kumaru serve -c configs/local-gpu.yaml
```

Point `backend.model` at a Hugging Face id (`Qwen/Qwen2.5-7B-Instruct`) or at a
local directory containing your own weights. Rough VRAM needs: bf16 ≈ 2 GB per
billion parameters, so a 7B model wants ~16 GB.

## 4. Day-to-day use

```bash
kumaru serve                       # web UI + API           (the normal way)
kumaru serve --port 9000           # different port
kumaru chat                        # terminal REPL, /reset and /exit
kumaru ask "explain mmap in one paragraph"
kumaru ask --no-stream "..." > answer.txt
kumaru config -c configs/cpu-only.yaml   # effective settings, as JSON
```

In the browser UI:

| Action | How |
|---|---|
| Send | `Enter` |
| New line | `Shift+Enter` |
| Stop generating | **Stop** button (aborts the request immediately) |
| Clear conversation | **Clear** — also clears server-side memory for the session |
| Light/dark | **Theme** |

Each browser tab keeps a session id in `localStorage`, so a reload replays the
conversation from the server instead of losing it.

## 5. Configuration

Precedence, lowest to highest:

1. built-in defaults (`src/kumaru/core/config.py`)
2. the YAML file passed with `-c`
3. `KUMARU_*` environment variables
4. CLI flags (`--backend`, `--model`, `--host`, `--port`, `--log-level`)

The settings you will actually touch:

| Setting | Meaning | Typical |
|---|---|---|
| `backend.name` | which engine generates tokens | `openai_compat` |
| `backend.model` | model id, name, or path to a `.gguf` | depends |
| `backend.base_url` | HTTP endpoint for `openai_compat` | `http://localhost:11434/v1` |
| `backend.context_length` | KV-cache window (llamacpp) | 4096 |
| `backend.gpu_layers` | layers offloaded to the GPU (llamacpp) | 0–35 |
| `agent.system_prompt` | the model's standing instructions; supports `{date}`/`{time}` | see `configs/default.yaml` |
| `agent.max_tokens` | cap on reply length | 512–1024 |
| `agent.temperature` | 0 = deterministic, 1 = creative | 0.7 (0.1–0.3 for factual work) |
| `agent.max_history_turns` | messages kept per session | 12 |
| `agent.max_history_chars` | hard character budget per session | 12000 |
| `server.host` | bind address — **keep it on loopback** | `127.0.0.1` |

Secrets are never stored in YAML. `backend.api_key_env` names an environment
variable; the value is read at request time and is not returned by
`/api/config`.

## 6. The HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | `{"message": "...", "session_id": "..."}` → `{"reply": ..., "usage": ...}` |
| `POST` | `/api/chat/stream` | same body, SSE frames: `{"delta": ...}` then `{"done": true, "usage": ...}` |
| `GET` | `/api/history/{session_id}` | replay a conversation |
| `DELETE` | `/api/history/{session_id}` | forget a conversation |
| `GET` | `/api/health` | backend, model, readiness |
| `GET` | `/api/config` | effective non-secret settings |
| `GET` | `/docs` | interactive OpenAPI docs (FastAPI) |

```bash
curl -N -X POST localhost:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"hello","session_id":"demo"}'
```

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| UI says *server unreachable* | server not running, or a different port | check the terminal running `kumaru serve` |
| *could not reach http://localhost:11434/v1* | the model server is not up | start `ollama serve` (or your llama.cpp/vLLM server) |
| `ModelNotAvailableError: model 'x' not found` | wrong model name | `ollama list`, then match `backend.model` exactly |
| `ModelNotAvailableError: ... requires torch` | optional extra not installed | `pip install -e '.[transformers]'` or `'.[llamacpp]'` |
| `GGUF file not found` | wrong path in `backend.model` | paths are relative to your working directory |
| Process is killed while loading | out of RAM/VRAM | smaller quantisation, lower `context_length`, fewer `gpu_layers` |
| Answers are extremely slow | model too large for the hardware | smaller model, or a Q4 GGUF instead of bf16 |
| Model repeats or rambles | sampling too hot | lower `agent.temperature` to 0.2–0.4 |
| Model forgets earlier turns | history budget reached | raise `agent.max_history_turns` / `max_history_chars` |
| `ConfigError: unknown key(s)` | typo in YAML | the message names the key and lists the valid ones |
| Port already in use | vLLM also defaults to 8000 | `kumaru serve --port 8010` |

More detail in the logs:

```bash
kumaru serve --log-level DEBUG
```

## 8. Developing

```bash
pip install -e '.[dev]'
pytest                    # ~50 fast tests, no model required
```

Where to make a change:

| You want to… | Edit |
|---|---|
| support another inference engine | add `src/kumaru/backends/<name>.py`, register it in `backends/__init__.py` |
| change what the model is told | `src/kumaru/agent/prompt.py` |
| change what is remembered | `src/kumaru/agent/memory.py` |
| add an endpoint | `src/kumaru/server/routes_chat.py` + `schemas.py` |
| change the UI | `ui/app.js`, `ui/styles.css`, `ui/index.html` — reload the page, no build |
| add a setting | `src/kumaru/core/config.py` (and document it here) |

Every file also has a how-to in [`howto/`](howto/README.md).

## 9. Where to go next

* [`SCALABILITY.md`](SCALABILITY.md) — how this design grows from a laptop to a cluster.
* [`TRAINING.md`](TRAINING.md) — fine-tune a model on your own data and serve it.
