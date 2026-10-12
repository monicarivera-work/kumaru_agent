# Install a pinned, prebuilt llama.cpp `llama-server` into tools\llama.cpp\.
#
#   .\scripts\get-llama-server.ps1                     # NVIDIA, CUDA 13.4 build (driver 580+)
#   .\scripts\get-llama-server.ps1 -Flavor cuda-12.4   # NVIDIA with an older driver (551+)
#   .\scripts\get-llama-server.ps1 -Flavor vulkan      # AMD / Intel GPUs
#   .\scripts\get-llama-server.ps1 -Flavor cpu         # no GPU at all
#   .\scripts\get-llama-server.ps1 -Version b11600     # a different llama.cpp release
#   .\scripts\get-llama-server.ps1 -Force              # download again and reinstall
#
# Re-running is safe. With no arguments it keeps the flavor you installed,
# follows $PinnedVersion below (unless you chose a version with -Version), and
# does nothing when that build is already in place.
#
# Why a prebuilt server binary rather than `pip install llama-cpp-python`:
#   * New model architectures land in llama.cpp first. The Python bindings lag
#     by weeks, and building them on Windows needs a CUDA toolchain + MSVC.
#   * llama-server batches concurrent requests across "slots". That is what
#     makes benchmark runs finish in minutes instead of hours.
#   * It is a separate process. A crash or OOM in the engine cannot take the
#     Kumaru server down with it, and VRAM is returned to the OS on exit.
#
# Safety: every zip is checked against the SHA-256 digest GitHub publishes for
# the release, and the new build is unpacked and test-started next to the old
# one before it replaces anything. A failed download or update leaves the
# current install working. Updating while Kumaru runs is fine: the running
# llama-server keeps its old build until Kumaru restarts.
#
# The Kumaru backend `llamacpp_server` finds the binary here automatically.
# Nothing is installed system-wide: deleting tools\llama.cpp uninstalls it.

[CmdletBinding()]
param(
    # Release tag such as b11541. Default: the version you last chose with
    # -Version, otherwise $PinnedVersion.
    [string]$Version = "",
    # Default: the flavor already installed, otherwise cuda-13.4.
    [ValidateSet("cuda-13.4", "cuda-12.4", "vulkan", "cpu")]
    [string]$Flavor,
    [string]$Destination = "",
    [switch]$Force
)

# The llama.cpp build this version of Kumaru is tested and benchmarked with.
$PinnedVersion = "b11541"
$DefaultFlavor = "cuda-13.4"

$ErrorActionPreference = "Stop"
# Invoke-WebRequest's progress bar slows large downloads by an order of
# magnitude on Windows PowerShell 5.1.
$ProgressPreference = "SilentlyContinue"
# Windows PowerShell 5.1 can default to TLS 1.0, which GitHub refuses.
[Net.ServicePointManager]::SecurityProtocol =
    [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

function Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }

function Show-PinHint($version) {
    if ($version -ne $PinnedVersion) {
        Write-Host ("Kumaru is tested with llama.cpp $PinnedVersion; this is $version. Plain re-runs keep " +
            "$version. To go back: .\scripts\get-llama-server.ps1 -Version $PinnedVersion") -ForegroundColor Yellow
    }
}

# Antivirus scanners briefly lock freshly unpacked files, which makes a folder
# rename fail, so retry for a couple of seconds before giving up.
function Move-Folder($from, $to) {
    for ($attempt = 1; ; $attempt++) {
        try { [System.IO.Directory]::Move($from, $to); return }
        catch { if ($attempt -ge 5) { throw }; Start-Sleep -Milliseconds 500 }
    }
}

$repo = Split-Path -Parent $PSScriptRoot
$defaultDestination = Join-Path $repo "tools\llama.cpp"
if (-not $Destination) { $Destination = $defaultDestination }
# Absolute and without a trailing slash, so "$Destination.new" is a sibling folder.
$Destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Destination).TrimEnd('\', '/')
$marker = Join-Path $Destination "kumaru-install.json"
$server = Join-Path $Destination "llama-server.exe"

# ---- What to install --------------------------------------------------------
$installed = $null
if (Test-Path -LiteralPath $marker) {
    # A corrupt marker counts as "nothing installed".
    try { $installed = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json } catch { $installed = $null }
}
# A version chosen with -Version is recorded as pinned=false and kept by later
# plain runs. Everything else follows $PinnedVersion, so pulling a newer Kumaru
# and re-running this script also moves the engine to the build it was tested on.
$targetVersion =
    if ($Version) { $Version }
    elseif ($installed -and $installed.pinned -eq $false -and $installed.version) { $installed.version }
    else { $PinnedVersion }
$targetFlavor =
    if ($Flavor) { $Flavor }
    elseif ($installed -and $installed.flavor) { $installed.flavor }
    else { $DefaultFlavor }

$upToDate = -not $Force -and $installed -and $installed.version -eq $targetVersion -and
    $installed.flavor -eq $targetFlavor -and (Test-Path -LiteralPath $server)

# The folder is replaced wholesale, so refuse anything that is not a previous
# install, such as a mistyped -Destination that points at your documents.
if (-not $upToDate -and (Test-Path -LiteralPath $Destination) -and -not (Test-Path -LiteralPath $marker) -and
    -not (Test-Path -LiteralPath $server) -and
    (Get-ChildItem -LiteralPath $Destination -Force | Select-Object -First 1)) {
    throw "$Destination is not empty and holds no llama.cpp install, so it will not be replaced. Pass an empty or new folder to -Destination."
}

# A build replaced while its llama-server was still running stays behind as
# <Destination>.old-<time> until that process exits. Delete the ones no longer in use.
$leaf = Split-Path -Leaf $Destination
Get-ChildItem -LiteralPath (Split-Path -Parent $Destination) -Directory -Filter "$leaf.old-*" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name.StartsWith("$leaf.old-", [StringComparison]::OrdinalIgnoreCase) } |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }

if ($upToDate) {
    Write-Host "llama-server $targetVersion ($targetFlavor) is already installed in $Destination. Use -Force to reinstall."
    Show-PinHint $targetVersion
    exit 0
}

$base = "https://github.com/ggml-org/llama.cpp/releases/download/$targetVersion"
$assets = @("llama-$targetVersion-bin-win-$targetFlavor-x64.zip")
if ($targetFlavor -like "cuda-*") {
    # The CUDA runtime DLLs ship separately so the engine zip stays small.
    $assets += "cudart-llama-bin-win-$targetFlavor-x64.zip"
}

Step "Installing llama-server $targetVersion ($targetFlavor) into $Destination"

# ---- Release metadata and checksums -----------------------------------------
# GitHub publishes a SHA-256 digest for every release file. If the API cannot
# be reached (offline, proxy, rate limit), the install still works, unverified.
$digests = @{}
$release = $null
try {
    $release = Invoke-RestMethod -UseBasicParsing -UserAgent "kumaru-get-llama-server" `
        -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$targetVersion"
}
catch {
    $status = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 }
    if ($status -eq 404) {
        throw "llama.cpp has no release '$targetVersion'. Tags look like $PinnedVersion; see https://github.com/ggml-org/llama.cpp/releases"
    }
    Write-Warning "Could not read the release from the GitHub API ($($_.Exception.Message)). Downloads will not be checksum-verified."
}
if ($release) {
    $names = @($release.assets | ForEach-Object { $_.name })
    foreach ($asset in $assets) {
        if ($names -notcontains $asset) {
            $flavors = foreach ($name in $names) { if ($name -match "^llama-.+-bin-win-(.+)-x64\.zip$") { $Matches[1] } }
            throw ("Release $targetVersion has no $asset. Its Windows x64 builds are: $($flavors -join ', '). " +
                "If a flavor was renamed, add the new name to the -Flavor ValidateSet at the top of this script.")
        }
    }
    foreach ($item in $release.assets) { if ($item.digest) { $digests[$item.name] = $item.digest } }
}

# ---- Download ---------------------------------------------------------------
# Zips are cached here, so a reinstall needs no network. -Force downloads again.
$staging = Join-Path ([System.IO.Path]::GetTempPath()) "kumaru-llama-$targetVersion"
New-Item -ItemType Directory -Force -Path $staging | Out-Null

foreach ($asset in $assets) {
    $zip = Join-Path $staging $asset
    if ($Force -and (Test-Path -LiteralPath $zip)) { Remove-Item -LiteralPath $zip -Force }
    if (Test-Path -LiteralPath $zip) {
        Step "Using cached $asset"
    }
    else {
        Step "Downloading $asset"
        # A temporary name, so an interrupted transfer never looks like a finished zip.
        $part = "$zip.part"
        if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
        $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
        if ($curl) {
            # curl draws its progress bar on stderr; judge it by the exit code instead.
            & { $ErrorActionPreference = "Continue"; & $curl.Source -L --fail --retry 3 --progress-bar -o $part "$base/$asset" }
            if ($LASTEXITCODE -ne 0) { throw "Download failed (curl exit code $LASTEXITCODE): $base/$asset" }
        }
        else {
            Invoke-WebRequest -UseBasicParsing -Uri "$base/$asset" -OutFile $part
        }
        Move-Item -LiteralPath $part -Destination $zip -Force
    }
    $expected = $digests[$asset]
    if ($expected) {
        $actual = "sha256:" + (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected) {
            Remove-Item -LiteralPath $zip -Force
            throw "Checksum mismatch for $asset (expected $expected, got $actual). The file was deleted; run this script again to download it fresh."
        }
        Write-Host "   checksum OK ($expected)"
    }
    elseif ($release) {
        Write-Warning "GitHub publishes no checksum for $asset, so it is not verified."
    }
}

# ---- Unpack and test-start next to the current install ----------------------
$newDir = "$Destination.new"
if (Test-Path -LiteralPath $newDir) {
    try { Remove-Item -LiteralPath $newDir -Recurse -Force }
    catch { throw "Could not remove the leftover folder $newDir ($($_.Exception.Message)). Delete it and run this script again." }
}
try {
    New-Item -ItemType Directory -Force -Path $newDir | Out-Null
    foreach ($asset in $assets) {
        Step "Extracting $asset"
        Expand-Archive -LiteralPath (Join-Path $staging $asset) -DestinationPath $newDir -Force
    }
    # Some releases nest the binaries one folder down; flatten so the path is stable.
    $exe = Get-ChildItem -LiteralPath $newDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
    if (-not $exe) { throw "llama-server.exe is not in $($assets[0]); the release layout may have changed." }
    if ($exe.DirectoryName -ne $newDir) {
        Get-ChildItem -LiteralPath $exe.DirectoryName -Force | Move-Item -Destination $newDir -Force
    }

    Step "Test-starting the new build"
    # llama-server logs to stderr. Under ErrorActionPreference=Stop, Windows
    # PowerShell would turn each stderr line into an error, so relax it here
    # and judge the probe by its exit code.
    $probe = & { $ErrorActionPreference = "Continue"; & (Join-Path $newDir "llama-server.exe") --version 2>&1 | ForEach-Object { "$_" } }
    if ($LASTEXITCODE -ne 0) {
        throw "The new llama-server.exe did not run (exit code $LASTEXITCODE):`n$($probe -join "`n")"
    }
    $probe | ForEach-Object { Write-Host "   $_" }

    [ordered]@{
        version      = $targetVersion
        flavor       = $targetFlavor
        pinned       = ($targetVersion -eq $PinnedVersion)
        installed_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $newDir "kumaru-install.json") -Encoding utf8
}
catch {
    Remove-Item -LiteralPath $newDir -Recurse -Force -ErrorAction SilentlyContinue
    throw
}

# ---- Swap ---------------------------------------------------------------------
# Two folder renames, so the install is never half-replaced. Windows allows
# renaming the folder of a running llama-server: that process keeps running
# from the renamed copy, and Kumaru picks up the new build on its next start.
Step "Swapping the new build into place"
$oldDir = "$Destination.old-" + (Get-Date -Format "yyyyMMdd-HHmmss")
$hadOld = Test-Path -LiteralPath $Destination
if ($hadOld) {
    try { Move-Folder $Destination $oldDir }
    catch {
        Remove-Item -LiteralPath $newDir -Recurse -Force -ErrorAction SilentlyContinue
        throw ("Could not move the current install aside ($($_.Exception.Message)). A program is using the " +
            "folder, typically a terminal whose current directory is inside it. Close it and run this " +
            "script again; nothing was changed.")
    }
}
try { Move-Folder $newDir $Destination }
catch {
    $reason = $_.Exception.Message
    if ($hadOld) { Move-Folder $oldDir $Destination }
    Remove-Item -LiteralPath $newDir -Recurse -Force -ErrorAction SilentlyContinue
    throw "Could not move the new build into place ($reason). Nothing was changed."
}
if ($hadOld) {
    Remove-Item -LiteralPath $oldDir -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $oldDir) {
        Write-Host ("The previous build is still running (a llama-server started by Kumaru?) and keeps " +
            "working from $oldDir. Restart Kumaru to use the new build; the next run of this script " +
            "deletes that folder once nothing uses it.") -ForegroundColor Yellow
    }
}

# ---- Report -------------------------------------------------------------------
# A CUDA build on a too-old driver still starts, then silently runs on the CPU.
# Listing devices makes that visible now rather than as a slow server later.
if ($targetFlavor -ne "cpu") {
    Step "Compute devices visible to llama.cpp"
    $devices = & { $ErrorActionPreference = "Continue"; & $server --list-devices 2>&1 | ForEach-Object { "$_" } }
    $devices | ForEach-Object { Write-Host "   $($_.TrimEnd())" }
    # Device lines look like "  CUDA0: NVIDIA GeForce RTX 3090 (24575 MiB, 23306 MiB free)".
    if (-not ($devices -match "^\s*(CUDA|Vulkan)\d+:")) {
        if ($targetFlavor -like "cuda-*") {
            Write-Host "`nNo CUDA device detected. Check 'nvidia-smi'; with a driver older than 580, rerun with -Flavor cuda-12.4." -ForegroundColor Yellow
        }
        else {
            Write-Host "`nNo Vulkan device detected. Update the GPU driver, or rerun with -Flavor cpu." -ForegroundColor Yellow
        }
    }
}

Write-Host "`nInstalled: $server" -ForegroundColor Green
Show-PinHint $targetVersion
if ($Destination -ne $defaultDestination) {
    Write-Host "Kumaru looks in tools\llama.cpp and on PATH only. To use this copy, set backend.llama_server.binary to $server" -ForegroundColor Yellow
}
Write-Host "Next:  kumaru serve -c qwen36" -ForegroundColor Green
