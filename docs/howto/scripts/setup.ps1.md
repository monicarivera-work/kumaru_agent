# `scripts/setup.ps1`

> One-shot installer that builds `.venv` and installs the right combination of optional dependencies.

**Read this when:** you are setting up on a fresh machine, adding GPU support to an existing install, or an install step failed and you need to know what it was doing.

---

## What it does

Creates a virtual environment in the repo root, installs Kumaru in editable mode together with its test tools (the `dev` extra: pytest, ruff, mypy, numpy), and — on request — installs PyTorch with the correct CUDA wheel plus the training, quantization, and GGUF extras. It finishes by running `kumaru info` so you see immediately whether the GPU was detected. Every install step checks pip's exit code, so a failed step stops the script with `<step> failed (exit code N)` instead of printing `Setup complete.`. With `-Gpu` it also replaces a CPU-only PyTorch it finds in `.venv`, and stops with an error if PyTorch still cannot use the GPU.

## Why it exists

PyTorch cannot be a normal dependency. The correct wheel depends on your CUDA version, and pip has no way to determine that. Installing plain `pip install torch` on Windows commonly yields a CPU-only build, which then silently runs your 7B model at 3 tokens/second with no error message. This script makes the CUDA index URL explicit and verifies the outcome, converting a silent misconfiguration into a visible one.

It also keeps the base install genuinely light. `pip install -e ".[dev]"` pulls only pure-Python packages and prebuilt wheels, needs no compiler, and finished in 75 seconds in a fresh virtual environment with an empty pip cache (about 100 seconds for the whole minimal run, including creating the venv and upgrading pip), so you can prove the stack works before committing to a multi-gigabyte download. The test tools are included because `scripts/verify.ps1`, the next step, runs the test suite.

## Parameters

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `-Gpu` | switch | off | Installs PyTorch (`torch>=2.3`) from the CUDA index named by `-CudaVersion`, replacing a CPU-only build if one is installed; runs `pip install -e ".[train]"` (transformers, peft, accelerate, datasets); then checks that PyTorch is a CUDA build that sees a GPU, and stops with an error if not |
| `-Quant` | switch | off | Runs `pip install -e ".[quant]"`: bitsandbytes, required for `load_in_4bit` and QLoRA |
| `-Gguf` | switch | off | Runs `pip install -e ".[gguf]"`: llama-cpp-python, for the in-process `llamacpp` backend (`llamacpp_server` needs no Python extra) |
| `-CudaVersion` | string | `cu126` | Which PyTorch CUDA index to use |

## How to use it

```powershell
# Minimal: server, UI, CLI and test tools. No GPU deps. Under 2 minutes.
.\scripts\setup.ps1

# Standard for an NVIDIA GPU: adds PyTorch CUDA + the training stack.
.\scripts\setup.ps1 -Gpu

# Everything, including 4-bit quantization and QLoRA.
.\scripts\setup.ps1 -Gpu -Quant

# Older CUDA runtime.
.\scripts\setup.ps1 -Gpu -CudaVersion cu118
```

Re-running is safe. It reuses an existing `.venv` rather than recreating it.

Re-running with `-Gpu` also repairs a CPU-only torch. pip keeps any installed torch that satisfies `torch>=2.3`, even PyPI's CPU-only Windows wheel (which a plain `pip install -e ".[train]"` installs), so the script first asks the installed torch for its CUDA version. If it has none, the script force-reinstalls torch from the CUDA index. `--no-deps` keeps that reinstall from touching torch's dependencies; the `.[train]` step after it resolves them. An existing CUDA build is kept, whatever `-CudaVersion` says. To switch CUDA versions, run `.\.venv\Scripts\python.exe -m pip uninstall -y torch` first.

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

With `-Gpu`, the script checks for you. After the installs it asks PyTorch for its CUDA version and whether it sees a GPU, then prints `CUDA 12.6 build, GPU available`. Otherwise it stops with one of the errors under [Failure modes](#failure-modes) instead of printing `Setup complete.`.

To check again later:

```powershell
.\.venv\Scripts\Activate.ps1
kumaru info
```

You want a `cuda:0` row naming your GPU with its VRAM and `bf16=yes`. If it says `unavailable (CPU only)` while you have an NVIDIA card, a CPU-only torch got installed, for example by a plain `pip install -e ".[train]"`. Run `.\scripts\setup.ps1 -Gpu` again: it replaces that build with the CUDA one.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `python : The term 'python' is not recognized` | Python not on PATH | Install Python 3.10+ and tick "Add to PATH" |
| `running scripts is disabled on this system` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `[WinError 32] ... kumaru.exe ... used by another process` | A `kumaru serve` is running | Stop it, then re-run |
| `kumaru info` shows `cuda: unavailable (CPU only)` after an install without `-Gpu` | torch came from PyPI, whose Windows wheel is CPU-only (a plain `pip install` of `torch`, `train` or `all`) | Run `.\scripts\setup.ps1 -Gpu`; it replaces the CPU-only build |
| `Installing PyTorch from https://download.pytorch.org/whl/cu126 failed (exit code N)` | Network error, or no torch wheel on that index matches your Python version | Read pip's message above it; check `-CudaVersion` |
| `Installing Kumaru failed`, `Installing the training stack failed`, or `Installing llama-cpp-python failed` (each with `(exit code N)`) | pip could not install a package: a network error, no wheel for your Python version, or (llama-cpp-python) no prebuilt wheel and no C/C++ compiler | Read pip's error above the message, fix it, and re-run; the script is safe to re-run. For llama-cpp-python, drop `-Gguf` and use the `llamacpp_server` backend, which needs no Python extra |
| `PyTorch (CUDA 12.6) sees no GPU.` | No NVIDIA GPU, or a driver older than the build (`nvidia-smi` shows a lower `CUDA Version`) | Update the driver, or run `.\.venv\Scripts\python.exe -m pip uninstall -y torch` and re-run with an older `-CudaVersion` (an existing CUDA build is kept otherwise) |
| `PyTorch is still a CPU-only build after installing from ...` | The `.[train]` step swapped the CUDA build for a CPU-only one, or `-CudaVersion` names a CPU-only index | Check `-CudaVersion` against the table above, then re-run with `-Gpu` and read pip's output |
| `PyTorch does not import.` | A broken or half-installed torch, or a missing system DLL | Run the printed command to see the error. A broken install is fixed by `.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps "torch>=2.3" --index-url https://download.pytorch.org/whl/cu126` |
| `Installing bitsandbytes failed (exit code N)` | No prebuilt wheel for your platform | Skip `-Quant`; set `load_in_4bit: false` and use bf16 |
| `Verifying the installation (kumaru info) failed (exit code N)` | Everything installed, but importing Kumaru or one of its optional packages raised an error | Run `.\.venv\Scripts\python.exe -m kumaru.cli info` to see the traceback |
| Very slow download | PyTorch CUDA wheels are ~2.5 GB | Expected; it happens once |

## Related files

- `pyproject.toml` — defines the `torch`, `train`, `quant`, `gguf`, and `dev` extras this script installs
- `scripts/verify.ps1` — run this next to confirm every layer works
- `docs/QUICKSTART.md` — the same steps, done manually, with more explanation
- `src/kumaru/cli.py` — the `info` command used for verification
