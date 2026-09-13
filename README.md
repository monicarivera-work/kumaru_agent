# Kumaru

A local-first LLM chat agent with a browser UI. It runs on your own machine,
answers questions in a chat window, and is built so the model behind it can be
swapped — from "no model at all" to a quantised 7B on your CPU to a GPU cluster
— without changing a line of the UI or the agent.

```
git clone <this repo> && cd kumaru_agent
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/kumaru serve
```

Then open **http://127.0.0.1:8000**. That works immediately, with no model
download, because the default backend is `echo` (it reflects your message back
so you can verify the whole pipeline). To get real answers, pick a backend —
see [Getting a real model](#3-get-real-answers) below.

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\pip install -e .
.venv\Scripts\kumaru serve
```

---

## What you get

| | |
|---|---|
| **Chat UI** | Streaming answers, markdown rendering, stop button, light/dark, history that survives a reload. No build step, no npm. |
| **HTTP API** | `POST /api/chat`, `POST /api/chat/stream` (SSE), `GET /api/history/{id}`, `GET /api/health`. |
| **CLI** | `kumaru serve`, `kumaru chat`, `kumaru ask "..."`, `kumaru config`, `kumaru backends`. |
| **Four backends** | `echo` (no deps), `openai_compat` (Ollama / llama.cpp server / vLLM / LM Studio / OpenAI), `transformers` (in-process GPU), `llamacpp` (in-process quantised CPU). |
| **Docs per file** | Every source file has a how-to under [`docs/howto/`](docs/howto/README.md). |

## Project layout

```
src/kumaru/
  core/       types, config, errors, logging, registry   (depends on nothing)
  backends/   anything that turns a prompt into tokens   (depends on core)
  agent/      prompt + memory + the chat loop            (depends on core, backends)
  server/     FastAPI app, schemas, routes               (depends on all of the above)
  cli.py      the only place that configures logging or exits
ui/           index.html, styles.css, app.js  - served at /
configs/      ready-made YAML configs for common hardware
docs/         GUIDE.md, SCALABILITY.md, TRAINING.md, howto/
tests/        fast tests; no model, no GPU, no network
```

Dependencies point one way only: `core <- backends <- agent <- server`. That is
what lets you replace any layer without touching the others.

---

## 3. Get real answers

Pick whichever matches your machine. Full walkthroughs are in
[`docs/GUIDE.md`](docs/GUIDE.md).

### Easiest: an existing local server (Ollama, LM Studio, llama.cpp)

```bash
ollama serve            # in another terminal
ollama pull llama3.1:8b
kumaru serve -c configs/openai-compat.yaml
```

### No GPU: a quantised model in-process

```bash
pip install -e '.[llamacpp]'
# download a *.gguf into ./models/, then:
kumaru serve -c configs/cpu-only.yaml
```

### With a GPU: Hugging Face weights in-process

```bash
pip install -e '.[transformers]'     # install a CUDA build of torch first
kumaru serve -c configs/local-gpu.yaml
```

## Configuration

Precedence, lowest to highest: **built-in defaults → YAML file → `KUMARU_*`
environment variables → CLI flags**. Print what is actually in effect:

```bash
kumaru config -c configs/cpu-only.yaml
```

Common environment variables: `KUMARU_BACKEND`, `KUMARU_MODEL`,
`KUMARU_BASE_URL`, `KUMARU_PORT`, `KUMARU_SYSTEM_PROMPT`, `KUMARU_LOG_LEVEL`.
API keys are read from the environment only (`KUMARU_API_KEY` by default) and
are never written to a config file or returned by `/api/config`.

## Tests

```bash
pip install -e '.[dev]'
pytest
```

The suite runs against the `echo` backend, so it needs no weights, no GPU and
no network.

## Security note

`server.host` defaults to `127.0.0.1` on purpose: the API has no
authentication, so binding it to `0.0.0.0` puts an open text generator on your
network. Put it behind an authenticating reverse proxy before exposing it.

## Documentation

* [`docs/GUIDE.md`](docs/GUIDE.md) — install, run, operate, troubleshoot.
* [`docs/SCALABILITY.md`](docs/SCALABILITY.md) — what "scalable" means here, concretely.
* [`docs/TRAINING.md`](docs/TRAINING.md) — how to fine-tune a model and plug it in.
* [`docs/howto/`](docs/howto/README.md) — one how-to per source file.

## Licence

MIT.
