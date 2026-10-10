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
| 24 GB+ | **Qwen3.6-35B-A3B**, the smartest model here: 131k context, optional thinking | `qwen36` (**recommended**, Step 3) |
| 24 GB+ | 7B in bf16, 16k context, **and** QLoRA fine-tuning | `local-gpu` |
| 12–16 GB | Qwen3.6 with part of its weights in system RAM: slower | `qwen36` + `n_cpu_moe` ([how](howto/configs/qwen36.yaml.md#share-the-gpu-with-a-game-or-use-a-smaller-card)) |
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
pip install -e ".[dev]"
```

macOS / Linux: `source .venv/bin/activate` instead.

This installs FastAPI, Pydantic, Typer and httpx, plus pytest, ruff and mypy (the `dev` extra) so `scripts\verify.ps1` can run the tests. No PyTorch, no CUDA, nothing heavy. It took 95 seconds in a fresh virtual environment with an empty pip cache; re-runs are faster.

---

## Step 2 — Prove the install works, without downloading a model

```powershell
kumaru serve -c test
```

Open **http://127.0.0.1:8080** and send a message. You will get an instant, streaming, canned reply from the `echo` backend.

**This step is not optional, and it is not a toy.** It verifies the server, the routing, the SSE streaming protocol, and the UI independently of anything model-related. If something breaks later, coming back here tells you immediately which half of the system is at fault.

Stop it with `Ctrl+C`.

---

## Step 3 — Run the smart model: Qwen3.6 (no PyTorch needed)

On an NVIDIA GPU with 24 GB (RTX 3090 or 4090), this is the model to use. Qwen3.6-35B-A3B is a *mixture of experts*: 35B parameters hold its knowledge, but only about 3B of them work on any given token. So it answers like a large model at the speed of a small one, about 110 tokens/s on a 3090. Kumaru runs it with llama.cpp's own server, a single program you download once, so this step needs no PyTorch.

**3a. Install the engine** (once):

```powershell
.\scripts\get-llama-server.ps1
```

It downloads a pinned build of llama.cpp's `llama-server` into `tools\llama.cpp\`, checks it against GitHub's published SHA-256, and lists the GPUs it can see. Re-running it is safe: it does nothing when that build is already installed. NVIDIA driver older than 580? Add `-Flavor cuda-12.4`. AMD or Intel GPU: `-Flavor vulkan`.

**3b. Choose where the weights go** (optional). The model file is 17.7 GB, saved by default under `%USERPROFILE%\.cache\kumaru\models`. To keep it off `C:`:

```powershell
setx KUMARU_BACKEND__MODELS_DIR D:\kumaru\models
```

`setx` only reaches terminals opened *after* it runs, so open a new one.

**3c. Serve it:**

```powershell
kumaru serve -c qwen36
```

Open **http://127.0.0.1:8080**. The first start downloads the weights; the UI shows `loading model…` until they are in place. Later starts take about 15 seconds.

**Thinking.** The **Think** button under the message box makes the model reason step by step before it answers. It is off by default, so answers arrive in seconds. Turn it on for multi-step math, logic puzzles and tricky code, the problems it is designed for; it can take minutes. So far it has not measurably beaten Think off, but that test capped thinking and chat does not ([BENCHMARKS.md](BENCHMARKS.md)).

**Gaming on the same GPU?** Qwen3.6 alone uses about 19.7 GB of the 3090's 24 GB. When something else needs VRAM, start it like this instead (measured next to World of Warcraft: 11.9 GB, about 27 tokens/s):

```powershell
$env:KUMARU_BACKEND__LLAMA_SERVER__N_CPU_MOE = '18'
kumaru serve -c qwen36 --context 65536
```

If you only want to chat, you are done: skip to [Step 6](#step-6--confirm-it-is-actually-working). Steps 4 and 5 add PyTorch, which fine-tuning and the `local-gpu` / `low-vram` profiles need.

---

## Step 4 — Install PyTorch

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

You want to see a `cuda:0` row naming your GPU. If it says `unavailable (CPU only)` while you do have an NVIDIA card, you installed the CPU wheel. Running the CUDA command again will not replace it, because pip sees torch is already installed. Uninstall it first: `pip uninstall -y torch`, then run the CUDA command above.

---

## Step 5 — Run a model through PyTorch

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

## Step 6 — Confirm it is actually working

```powershell
kumaru eval --suite smoke -c qwen36      # or -c local-gpu, whichever you serve
```

Six quick questions checking factual recall, arithmetic, format compliance, code generation, and calibrated uncertainty. You get a pass rate and speed numbers.

Record this number. It is your baseline. Every future change — a new model, quantization, a fine-tune — gets compared against it.

Expected on an RTX 3090 with Qwen2.5-7B in bf16: **6/6 passed, 30–45 tok/s, time-to-first-token 200–500 ms.**

The smoke suite tells you the model runs. To measure how good it is (accuracy on public benchmarks, with error bars, and whether a change really helped), use `kumaru bench`:

```powershell
pip install -e ".[bench]"
kumaru bench run golden -c qwen36            # your own 27 questions, a few minutes
kumaru bench run gsm8k mmlu_pro -c qwen36    # math and knowledge, 100 questions each
```

[BENCHMARKS.md](BENCHMARKS.md) explains the method and the measured results.

---

## Step 7 — Use it from your own code

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

## Step 8 — Terminal-only workflow

No server, no browser:

```powershell
kumaru chat -c qwen36                                      # interactive REPL
kumaru chat -c qwen36 -q "What is grouped-query attention?"   # one question, then exit
kumaru chat -c qwen36 --think -q "How many primes are below 100?"   # reason first
```

`/reset` clears history, `/exit` quits.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` at startup | Model too large for your VRAM | Use `-c low-vram` (4-bit), or a smaller model |
| Status stuck on `loading model…` | First-run download in progress | Watch the terminal — HF prints download progress |
| `llama-server was not found` | Step 3a skipped | `.\scripts\get-llama-server.ps1` |
| `llama-server exited during startup` | Usually not enough free VRAM | Close other GPU apps, or use the gaming recipe in Step 3 |
| Qwen3.6 drops from ~110 to single-digit tok/s | VRAM overflowed into shared system memory (Task Manager → GPU → "Shared GPU memory" grows) | The gaming recipe in Step 3; raise `N_CPU_MOE` further if needed |
| `You are sending unauthenticated requests to the HF Hub` | No Hugging Face token set | Harmless. Setting `HF_TOKEN` only makes downloads faster |
| `torch not installed` in `kumaru info` | Step 4 skipped | See Step 4 (not needed for `qwen36`) |
| `cuda: unavailable` but you have a GPU | CPU-only PyTorch wheel | Reinstall with `--index-url .../cu126` |
| Generation is very slow (< 5 tok/s) on a GPU | Model silently offloaded to CPU | `kumaru info` shows the device; lower `max_context_length` or use 4-bit |
| `Port 8080 is already in use` | Another process has it | `kumaru serve --port 8090` |
| UI loads but replies never arrive | A proxy is buffering the SSE stream | Connect directly, bypassing the proxy |
| `bitsandbytes` errors with `load_in_4bit` | Optional dep missing | `.\scripts\setup.ps1 -Quant` (or `pip install -e ".[quant]"` in the Kumaru folder) |

When in doubt, these two commands answer most questions:

```powershell
kumaru info              # what is installed, what hardware is visible
kumaru config -c local-gpu   # what settings actually won, after all four config layers
```

---

## Where to go next

- **Use it well** → [`../GUIDE.md`](../GUIDE.md)
- **Measure it** → [`BENCHMARKS.md`](BENCHMARKS.md)
- **Train it** → [`TRAINING.md`](TRAINING.md)
- **Scale it** → [`SCALABILITY.md`](SCALABILITY.md)
- **Understand it** → [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`howto/`](howto/)
