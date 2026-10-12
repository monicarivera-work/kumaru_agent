# `scripts/setup.ps1`

> One-shot installer that builds `.venv` and installs the right combination of optional dependencies.

**Read this when:** you are setting up on a fresh machine, adding GPU support to an existing install, or an install step failed and you need to know what it was doing.

---

## What it does

Creates a virtual environment in the repo root, installs Kumaru in editable mode together with its test tools (the `dev` extra: pytest, ruff, mypy), and — on request — installs PyTorch with the correct CUDA wheel plus the training, quantization, and GGUF extras. It finishes by running `kumaru info` so you see immediately whether the GPU was detected.

## Why it exists

PyTorch cannot be a normal dependency. The correct wheel depends on your CUDA version, and pip has no way to determine that. Installing plain `pip install torch` on Windows commonly yields a CPU-only build, which then silently runs your 7B model at 3 tokens/second with no error message. This script makes the CUDA index URL explicit and verifies the outcome, converting a silent misconfiguration into a visible one.

It also keeps the base install genuinely light. `pip install -e ".[dev]"` pulls only pure-Python packages and prebuilt wheels, needs no compiler, and finished in 95 seconds in a fresh virtual environment with an empty pip cache, so you can prove the stack works before committing to a multi-gigabyte download. The test tools are included because `scripts/verify.ps1`, the next step, runs the test suite.

## Parameters

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `-Gpu` | switch | off | Installs PyTorch from the CUDA index named by `-CudaVersion`, then runs `pip install -e ".[train]"` (transformers, peft, accelerate, datasets) |
| `-Quant` | switch | off | Runs `pip install -e ".[quant]"`: bitsandbytes, required for `load_in_4bit` and QLoRA |
| `-Gguf` | switch | off | Runs `pip install -e ".[gguf]"`: llama-cpp-python, for the in-process `llamacpp` backend (`llamacpp_server` needs no Python extra) |
| `-CudaVersion` | string | `cu126` | Which PyTorch CUDA index to use |

## How to use it

```powershell
# Minimal: server, UI, CLI and test tools. No GPU deps. About 1.5 minutes.
.\scripts\setup.ps1

# Standard for an NVIDIA GPU: adds PyTorch CUDA + the training stack.
.\scripts\setup.ps1 -Gpu

# Everything, including 4-bit quantization and QLoRA.
.\scripts\setup.ps1 -Gpu -Quant

# Older CUDA runtime.
.\scripts\setup.ps1 -Gpu -CudaVersion cu118
```

Re-running is safe. It reuses an existing `.venv` rather than recreating it.

### Picking a CUDA version

Check what your driver supports:

```powershell
nvidia-smi
```

The `CUDA Version` in the top-right is the **maximum** your driver supports. Pick a PyTorch build at or below it:

| Driver reports | Use |
|---|---|
| 12.6+ | `cu126` (default) |
| 12.1–12.5 | `cu121` |
| 11.8–12.0 | `cu118` |

## Verifying it worked

```powershell
.\.venv\Scripts\Activate.ps1
kumaru info
```

You want a `cuda:0` row naming your GPU with its VRAM and `bf16=yes`. If it says `unavailable (CPU only)` while you have an NVIDIA card, a CPU-only torch got installed (for example by a plain `pip install -e ".[train]"` before `-Gpu`). Re-running `-Gpu` alone keeps it, because pip sees torch is already installed. Run `.\.venv\Scripts\python.exe -m pip uninstall -y torch` first, then `.\scripts\setup.ps1 -Gpu`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `python : The term 'python' is not recognized` | Python not on PATH | Install Python 3.10+ and tick "Add to PATH" |
| `running scripts is disabled on this system` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `[WinError 32] ... kumaru.exe ... used by another process` | A `kumaru serve` is running | Stop it, then re-run |
| Install succeeds but `cuda: unavailable` | CPU-only torch wheel, or a `-CudaVersion` newer than your driver supports | `pip uninstall -y torch`, then re-run with `-Gpu` (pip never replaces an installed torch on its own); check `-CudaVersion` against your driver |
| `bitsandbytes` fails to build | No prebuilt wheel for your platform | Skip `-Quant`; set `load_in_4bit: false` and use bf16 |
| Very slow download | PyTorch CUDA wheels are ~2.5 GB | Expected; it happens once |

## Related files

- `pyproject.toml` — defines the `torch`, `train`, `quant`, `gguf`, and `dev` extras this script installs
- `scripts/verify.ps1` — run this next to confirm every layer works
- `docs/QUICKSTART.md` — the same steps, done manually, with more explanation
- `src/kumaru/cli.py` — the `info` command used for verification
