# Quickstart

From nothing to a working local assistant. Roughly 10 minutes of your attention, plus download time.

---

## Step 0 — Check your hardware

Your GPU's VRAM decides which path you take. On Windows:

```powershell
nvidia-smi --query-gpu=name,memory.total --format=csv
```

| VRAM | What you can run | Profile |
|---|---|---|
| 24 GB+ | 7B in bf16, 16k context, **and** QLoRA fine-tuning | `local-gpu` |
| 12–16 GB | 7B in 4-bit, 8k context | `low-vram` |
| 8–11 GB | 7B in 4-bit, 4k context, or a 3B in bf16 | `low-vram` |
| < 8 GB or no GPU | 1.5B on CPU via llama.cpp | `cpu-only` |

No GPU is not a blocker. It changes which profile you use, nothing else.

---

## Step 1 — Install

The scripted way:

```powershell
cd <your-clone-of-kumaru_agent>
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

Or by hand, if you prefer to see each step:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

macOS / Linux: `source .venv/bin/activate` instead.

This installs FastAPI, Pydantic, Typer, and httpx. No PyTorch, no CUDA, nothing heavy. It takes about twenty seconds.

---

## Step 2 — Prove the install works, without downloading a model

```powershell
kumaru serve -c test
```

Open **http://127.0.0.1:8080** and send a message. You will get an instant, streaming, canned reply from the `echo` backend.

**This step is not optional, and it is not a toy.** It verifies the server, the routing, the SSE streaming protocol, and the UI independently of anything model-related. If something breaks later, coming back here tells you immediately which half of the system is at fault.

Stop it with `Ctrl+C`.

---

## Step 3 — Install PyTorch

PyTorch is a separate step because the correct wheel depends on your CUDA version, and pip cannot work that out for you.

Scripted:

```powershell
.\scripts\setup.ps1 -Gpu -Quant
```

Or by hand:

```powershell
# NVIDIA GPU (most people)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then the ML stack:

```powershell
pip install -e ".[train]"
```

Verify:

```powershell
kumaru info
```

You want to see a `cuda:0` row naming your GPU. If it says `unavailable (CPU only)` while you do have an NVIDIA card, you installed the CPU wheel — reinstall with the CUDA index URL above.

---

## Step 4 — Run a real model

```powershell
kumaru serve -c local-gpu
```

**First run downloads ~15 GB.** It goes to your Hugging Face cache (`%USERPROFILE%\.cache\huggingface`), so it happens exactly once. The server starts immediately and the UI shows `loading model…` — the download is happening in the background, and `/healthz` reports `loading` rather than failing.

Once the dot turns green, ask it something.

**Impatient?** Start with the 1.5B model — about 3 GB, downloads in a minute or two, and is genuinely usable:

```powershell
kumaru serve --model Qwen/Qwen2.5-1.5B-Instruct
```

---

## Step 5 — Confirm it is actually working

```powershell
kumaru eval --suite smoke
```

Six quick questions checking factual recall, arithmetic, format compliance, code generation, and calibrated uncertainty. You get a pass rate and speed numbers.

Record this number. It is your baseline. Every future change — a new model, quantization, a fine-tune — gets compared against it.

Expected on an RTX 3090 with Qwen2.5-7B in bf16: **6/6 passed, 30–45 tok/s, time-to-first-token 200–500 ms.**

---

## Step 6 — Use it from your own code

The API is OpenAI-compatible, so the official SDK works unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="kumaru",
    messages=[{"role": "user", "content": "Explain gradient accumulation in two sentences."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Or without any SDK:

```powershell
curl http://127.0.0.1:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'
```

> PowerShell mangles quotes inside `curl` arguments. If you get a JSON decode error, that is PowerShell, not Kumaru. Use the Python snippet, or put the body in a file and use `-d '@body.json'`.

---

## Step 7 — Terminal-only workflow

No server, no browser:

```powershell
kumaru chat                                    # interactive REPL
kumaru chat -q "What is grouped-query attention?"   # one question, then exit
```

`/reset` clears history, `/exit` quits.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` at startup | Model too large for your VRAM | Use `-c low-vram` (4-bit), or a smaller model |
| Status stuck on `loading model…` | First-run download in progress | Watch the terminal — HF prints download progress |
| `torch not installed` in `kumaru info` | Step 3 skipped | See Step 3 |
| `cuda: unavailable` but you have a GPU | CPU-only PyTorch wheel | Reinstall with `--index-url .../cu126` |
| Generation is very slow (< 5 tok/s) on a GPU | Model silently offloaded to CPU | `kumaru info` shows the device; lower `max_context_length` or use 4-bit |
| `Port 8080 is already in use` | Another process has it | `kumaru serve --port 8090` |
| UI loads but replies never arrive | A proxy is buffering the SSE stream | Connect directly, bypassing the proxy |
| `bitsandbytes` errors with `load_in_4bit` | Optional dep missing | `pip install "kumaru[quant]"` |

When in doubt, these two commands answer most questions:

```powershell
kumaru info              # what is installed, what hardware is visible
kumaru config -c local-gpu   # what settings actually won, after all four config layers
```

---

## Where to go next

- **Use it well** → [`../GUIDE.md`](../GUIDE.md)
- **Train it** → [`TRAINING.md`](TRAINING.md)
- **Scale it** → [`SCALABILITY.md`](SCALABILITY.md)
- **Understand it** → [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`howto/`](howto/)
