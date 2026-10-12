# Full self-check: proves every layer of Kumaru works on this machine.
#
#   .\scripts\verify.ps1          # fast checks only (no GPU, no downloads)
#   .\scripts\verify.ps1 -Full    # + a real model on the GPU (downloads ~1 GB)
#
# Run this after setup, after upgrading a dependency, or whenever you want to
# know whether a problem is yours or Kumaru's. Each stage isolates one layer,
# so the first failure tells you where to look.

[CmdletBinding()]
param(
    [switch]$Full,
    [int]$Port = 8099,
    [string]$SmallModel = "Qwen/Qwen2.5-0.5B-Instruct"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$py = Join-Path $repo ".venv\Scripts\python.exe"
$kumaru = Join-Path $repo ".venv\Scripts\kumaru.exe"

if (-not (Test-Path $py)) {
    Write-Host "No .venv found. Run .\scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
}

$results = [ordered]@{}

function Check($name, [scriptblock]$body) {
    Write-Host "`n>> $name" -ForegroundColor Cyan
    # PowerShell turns *any* stderr write from a native command into a
    # terminating error while $ErrorActionPreference is "Stop". Kumaru logs to
    # stderr by design, so every successful command would "fail". Preference
    # variables are dynamically scoped, so relaxing it here applies inside
    # $body. Each check therefore validates explicitly and throws on failure.
    $ErrorActionPreference = "Continue"
    try {
        & $body
        $results[$name] = "PASS"
        Write-Host "   PASS" -ForegroundColor Green
    } catch {
        $results[$name] = "FAIL: $($_.Exception.Message)"
        Write-Host "   FAIL: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# 1. Environment ------------------------------------------------------------
Check "environment" { & $kumaru info | Out-Host }

# 2. Config resolution ------------------------------------------------------
Check "config" {
    foreach ($profile in @("default", "test", "local-gpu", "low-vram", "cpu-only", "vllm", "qwen36", "qwen25-7b-gguf")) {
        & $kumaru config -c $profile *> $null
        if ($LASTEXITCODE -ne 0) { throw "profile '$profile' failed to load" }
    }
}

# 3. Unit tests -------------------------------------------------------------
Check "unit tests" {
    & $py -m pytest tests -q | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
}

# 4. Agent stack, no model --------------------------------------------------
# Kumaru writes logs to stderr and answers to stdout. We capture stdout only:
# merging stderr with 2>&1 would make PowerShell treat ordinary log lines as
# terminating errors under $ErrorActionPreference = "Stop".
Check "agent (echo backend)" {
    $out = (& $kumaru chat -c test -q "verification probe" 2>$null) | Out-String
    if ($LASTEXITCODE -ne 0) { throw "kumaru chat exited with $LASTEXITCODE" }
    if ($out -notmatch "echo:") { throw "echo backend produced no reply" }
}

# 5. HTTP server + streaming ------------------------------------------------
Check "server + SSE streaming" {
    $job = Start-Job -ScriptBlock {
        param($exe, $port) & $exe serve -c test --port $port
    } -ArgumentList $kumaru, $Port

    try {
        Start-Sleep -Seconds 6
        $probe = @"
import json, sys, httpx
B = "http://127.0.0.1:$Port"
assert httpx.get(f"{B}/healthz", timeout=20).json()["status"] == "ok", "healthz not ok"
r = httpx.post(f"{B}/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "ping"}]}, timeout=60).json()
assert r["choices"][0]["message"]["content"], "empty completion"
frames = 0
with httpx.stream("POST", f"{B}/v1/chat/completions",
                  json={"messages": [{"role": "user", "content": "ping"}], "stream": True},
                  timeout=60) as s:
    for line in s.iter_lines():
        if line.startswith("data:"):
            frames += 1
            if line[5:].strip() == "[DONE]":
                break
assert frames > 2, f"only {frames} SSE frames"
assert httpx.get(f"{B}/", timeout=20).status_code == 200, "UI not served"
print(f"  healthz ok, completion ok, {frames} SSE frames, UI served")
"@
        $probe | & $py -
        if ($LASTEXITCODE -ne 0) { throw "HTTP probe failed" }
    } finally {
        Stop-Job $job -ErrorAction SilentlyContinue
        Remove-Job $job -Force -ErrorAction SilentlyContinue
    }
}

# 6. llama-server binary (optional) -----------------------------------------
# The engine behind GGUF profiles such as qwen36. Found where Kumaru looks:
# the copy get-llama-server.ps1 installs, then PATH. --version loads the binary
# and its DLLs without a model or any GPU memory.
$llamaServer = Join-Path $repo "tools\llama.cpp\llama-server.exe"
if (-not (Test-Path $llamaServer)) {
    $onPath = Get-Command llama-server -ErrorAction SilentlyContinue
    $llamaServer = if ($onPath) { $onPath.Source } else { $null }
}
if ($llamaServer) {
    Check "llama-server binary" {
        $out = (& $llamaServer --version 2>&1) | Out-String
        if ($LASTEXITCODE -ne 0) { throw "llama-server --version exited with $LASTEXITCODE" }
        # "version: 0.6.0-dev (build 11541, commit ...)" today; "version: 4567 (abc123)" in older builds.
        if ($out -match "\(build (\d+)" -or $out -match "version:\s*(\d+)\s*\(") {
            Write-Host "   build b$($Matches[1]): $llamaServer"
        }
        else { throw "unexpected --version output: $($out.Trim())" }
    }
} else {
    Write-Host "`n>> llama-server binary" -ForegroundColor Cyan
    Write-Host "   not installed; skipped (.\scripts\get-llama-server.ps1 installs it)" -ForegroundColor DarkGray
}

# 7. Real model on real hardware -------------------------------------------
if ($Full) {
    if ($llamaServer) {
        Check "llama-server sees the GPU" {
            $devices = (& $llamaServer --list-devices 2>&1) | Out-String
            if ($LASTEXITCODE -ne 0) { throw "llama-server --list-devices exited with $LASTEXITCODE" }
            $gpu = ($devices -split "`n") | Where-Object { $_ -match "CUDA\d+:" } | Select-Object -First 1
            if (-not $gpu) { throw "no CUDA device listed; check nvidia-smi and get-llama-server.ps1 -Flavor" }
            Write-Host "   $($gpu.Trim())"
        }
    }
    Check "training pipeline (dry run)" {
        & $kumaru train configs/train/lora-small.yaml --dry-run | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "training dry run failed" }
    }

    Check "GPU inference ($SmallModel)" {
        $out = (& $kumaru chat --backend transformers --model $SmallModel `
            -q "Reply with the single word: ready" 2>$null) | Out-String
        if ($LASTEXITCODE -ne 0) { throw "kumaru chat exited with $LASTEXITCODE" }
        if ($out.Trim().Length -lt 3) { throw "no output from model" }
        Write-Host "   model said: $($out.Trim())"
    }

    Check "eval harness" {
        & $kumaru eval --backend transformers --model $SmallModel --suite smoke | Out-Host
    }
}

# Summary -------------------------------------------------------------------
Write-Host "`n================ SUMMARY ================" -ForegroundColor Cyan
$failed = 0
foreach ($k in $results.Keys) {
    $v = $results[$k]
    $color = if ($v -eq "PASS") { "Green" } else { "Red" }
    if ($v -ne "PASS") { $failed++ }
    Write-Host ("{0,-28} {1}" -f $k, $v) -ForegroundColor $color
}
Write-Host "=========================================" -ForegroundColor Cyan

if ($failed -gt 0) {
    Write-Host "`n$failed stage(s) failed. The FIRST failure is the one to fix." -ForegroundColor Red
    exit 1
}
Write-Host "`nAll stages passed." -ForegroundColor Green
if (-not $Full) { Write-Host "Run with -Full to also exercise the GPU and training paths." }
