# One-shot setup for Kumaru on Windows.
#
#   .\scripts\setup.ps1              # base install + test tools, no GPU deps
#   .\scripts\setup.ps1 -Gpu         # + PyTorch CUDA + training stack
#   .\scripts\setup.ps1 -Gpu -Quant  # + bitsandbytes for 4-bit / QLoRA
#
# Everything installs into .venv in the repo root. Nothing touches your system
# Python, so removing .venv fully uninstalls Kumaru.

[CmdletBinding()]
param(
    [switch]$Gpu,
    [switch]$Quant,
    [switch]$Gguf,
    [string]$CudaVersion = "cu126"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }

# A failing native command (pip, python) sets $LASTEXITCODE instead of throwing,
# even under ErrorActionPreference=Stop. Check it after every step that must
# succeed, so a failed install never ends in "Setup complete."
function Assert-Success($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE); see the output above." }
}

# --- virtualenv ------------------------------------------------------------
if (-not (Test-Path ".venv")) {
    Step "Creating virtual environment"
    python -m venv .venv
} else {
    Step "Reusing existing .venv"
}

$py = Join-Path $repo ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip --quiet

# --- base ------------------------------------------------------------------
# The dev extra (pytest, ruff, mypy, numpy) is small, and verify.ps1 runs the tests.
Step "Installing Kumaru (base: server, UI, CLI, plus the test tools)"
& $py -m pip install -e ".[dev]" --quiet
Assert-Success "Installing Kumaru"

# --- torch -----------------------------------------------------------------
# PyTorch is installed separately because the correct wheel depends on your
# CUDA version, which pip cannot determine on its own.
#
# The probe prints "<CUDA version, or cpu> <GPU usable>", or "none" when torch
# does not import. It sends Python's stderr to nul, because Windows PowerShell
# can turn stderr lines into errors under ErrorActionPreference=Stop.
$torchProbe = @'
import os
os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
try:
    import torch
except Exception:
    print('none')
else:
    print(torch.version.cuda or 'cpu', torch.cuda.is_available())
'@

if ($Gpu) {
    $torchIndex = "https://download.pytorch.org/whl/$CudaVersion"
    # pip keeps an installed torch that satisfies the version range, even a
    # CPU-only one (such as PyPI's Windows wheel), so replace that explicitly.
    # --no-deps leaves torch's dependencies alone; the ".[train]" step below
    # brings them in line.
    $cuda = ("$(& $py -c $torchProbe)" -split " ")[0]
    if ($cuda -eq "cpu") {
        Step "Replacing the CPU-only PyTorch with the CUDA build ($CudaVersion)"
        & $py -m pip install "torch>=2.3" --force-reinstall --no-deps --index-url $torchIndex
    } else {
        Step "Installing PyTorch with CUDA ($CudaVersion)"
        & $py -m pip install "torch>=2.3" --index-url $torchIndex
    }
    Assert-Success "Installing PyTorch from $torchIndex"

    Step "Installing training stack (transformers, peft, accelerate, datasets)"
    & $py -m pip install -e ".[train]" --quiet
    Assert-Success "Installing the training stack"

    Step "Checking that PyTorch can use the GPU"
    $cuda, $gpuFound = "$(& $py -c $torchProbe)" -split " "
    if ($cuda -eq "cpu") {
        throw "PyTorch is still a CPU-only build after installing from $torchIndex; see pip's output above."
    }
    if (-not $gpuFound) {
        throw "PyTorch does not import. Run: .\.venv\Scripts\python.exe -c 'import torch'"
    }
    if ($gpuFound -ne "True") {
        throw ("PyTorch (CUDA $cuda) sees no GPU. Check that nvidia-smi lists one and reports " +
            "CUDA Version $cuda or later. If the driver is older, run " +
            ".\.venv\Scripts\python.exe -m pip uninstall -y torch, then this script with an older -CudaVersion.")
    }
    Write-Host "   CUDA $cuda build, GPU available"
}

if ($Quant) {
    Step "Installing bitsandbytes (4-bit / QLoRA)"
    & $py -m pip install -e ".[quant]" --quiet
    Assert-Success "Installing bitsandbytes"
}

if ($Gguf) {
    Step "Installing llama-cpp-python (GGUF backend)"
    & $py -m pip install -e ".[gguf]" --quiet
    Assert-Success "Installing llama-cpp-python"
}

# --- verify ----------------------------------------------------------------
Step "Verifying installation"
& $py -m kumaru.cli info
Assert-Success "Verifying the installation (kumaru info)"

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Next:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  kumaru serve -c test        # verify the stack, no model download"
if ($Gpu) {
    Write-Host "  kumaru serve -c local-gpu   # real model (~15 GB on first run)"
} else {
    Write-Host "  .\scripts\setup.ps1 -Gpu    # add GPU support when ready"
}
