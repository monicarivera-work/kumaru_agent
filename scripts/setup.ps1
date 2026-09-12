# One-shot setup for Kumaru on Windows.
#
#   .\scripts\setup.ps1              # base install, no GPU deps
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
Step "Installing Kumaru (base: server, UI, CLI)"
& $py -m pip install -e . --quiet

# --- torch -----------------------------------------------------------------
# PyTorch is installed separately because the correct wheel depends on your
# CUDA version, which pip cannot determine on its own.
if ($Gpu) {
    Step "Installing PyTorch with CUDA ($CudaVersion)"
    & $py -m pip install torch --index-url "https://download.pytorch.org/whl/$CudaVersion"

    Step "Installing training stack (transformers, peft, accelerate, datasets)"
    & $py -m pip install -e ".[train]" --quiet
}

if ($Quant) {
    Step "Installing bitsandbytes (4-bit / QLoRA)"
    & $py -m pip install -e ".[quant]" --quiet
}

if ($Gguf) {
    Step "Installing llama-cpp-python (GGUF backend)"
    & $py -m pip install -e ".[gguf]" --quiet
}

# --- verify ----------------------------------------------------------------
Step "Verifying installation"
& $py -m kumaru.cli info

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Next:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  kumaru serve -c test        # verify the stack, no model download"
if ($Gpu) {
    Write-Host "  kumaru serve -c local-gpu   # real model (~15 GB on first run)"
} else {
    Write-Host "  .\scripts\setup.ps1 -Gpu    # add GPU support when ready"
}
