# `scripts\get-llama-server.ps1`

> Installs a pinned, prebuilt llama.cpp `llama-server` (Windows x64) into `tools\llama.cpp\`, where the `llamacpp_server` backend finds it automatically. Downloads are checksum-verified, and an update never leaves you with a half-replaced install.

**Read this when:**
- you set up the `qwen36` or `qwen25-7b-gguf` profiles;
- you upgrade llama.cpp for a new model architecture, or go back to the tested build;
- you switch between CUDA, Vulkan and CPU builds;
- the install, the checksum or the GPU detection failed;
- you change the script and want to test it without touching your real install.

---

## What it does

1. **Works out what to install.**
   - Flavor: `-Flavor` if given, else the flavor recorded in the install's marker, `kumaru-install.json`, else `cuda-13.4`.
   - Version: `-Version` if given; else the installed version, if you chose it yourself with `-Version` earlier (the marker says `"pinned": false`); else the pin, `$PinnedVersion = "b11541"` at the top of the script.
2. **Refuses a foreign folder.** The destination is replaced wholesale. If an install is needed, and the destination exists, is not empty, and holds neither the marker nor `llama-server.exe`, the script stops and changes nothing. This protects you from a mistyped `-Destination`.
3. **Clears leftovers**, on every run including the no-op one. It deletes the `<Destination>.old-*` folders that step 9 leaves while an old build is still running, once nothing uses them. This is best effort and silent.
4. **Stops early when up to date.** If the marker records that version and flavor, `llama-server.exe` is present, and there is no `-Force`, it prints `llama-server <Version> (<Flavor>) is already installed in <Destination>. Use -Force to reinstall.` and exits 0. When the kept version is not the pin, a yellow line names the pinned tag and the command to go back. Nothing is downloaded.
5. **Reads the release from the GitHub API** (`https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/<Version>`):
   - **No such tag (HTTP 404):** it stops with `llama.cpp has no release '<Version>'` before downloading anything.
   - **Flavor missing:** if the release has no zip for your flavor, it stops and lists the Windows x64 flavors that release does have.
   - **Otherwise:** it records the SHA-256 digest GitHub publishes for each file.
   - **API unreachable** (offline, a proxy, or the limit of 60 unauthenticated API requests per hour): it prints a yellow warning and continues *without* verification.
6. **Downloads** each zip into `%TEMP%\kumaru-llama-<Version>\`:
   - It needs `llama-<Version>-bin-win-<Flavor>-x64.zip`, plus `cudart-llama-bin-win-<Flavor>-x64.zip` (the CUDA runtime DLLs) for CUDA flavors.
   - It uses `curl.exe -L --fail --retry 3`, or `Invoke-WebRequest` when `curl.exe` is missing.
   - Each transfer goes to `<zip>.part` and is renamed only once complete, so an interrupted download never looks finished.
   - A zip already in the cache is reused. `-Force` deletes it first.
7. **Verifies** every zip, cached or fresh, against its digest and prints `checksum OK (sha256:…)`.
   - On a mismatch it deletes the zip and stops; your install is untouched, and the next run downloads it again.
   - If the release lists no digest for a file, as can happen with older releases, it warns `GitHub publishes no checksum for …` and continues.
8. **Unpacks and test-starts the new build next to the old one:**
   - It extracts into `<Destination>.new`. If a release nests the binaries one folder down, it flattens them so the path is always `llama-server.exe` at the top.
   - It runs `llama-server.exe --version` from there. If that fails, it deletes `.new` and stops, and your install is untouched.
   - It writes the marker into `.new`.
9. **Swaps**:
   - It renames `<Destination>` to `<Destination>.old-<yyyyMMdd-HHmmss>`, renames `.new` to `<Destination>`, then deletes the old folder.
   - Windows allows renaming the folder of a *running* program, so this works while `kumaru serve` is up. The running `llama-server` keeps using the old build until Kumaru restarts, and the old folder cannot be deleted until then. The script says so in yellow, and step 3 of any later run removes the folder.
   - If a process has its *current directory* inside the install, the first rename fails. The script then deletes `.new` and stops; nothing was changed.
10. **Checks the GPU** (GPU flavors only). It runs `llama-server.exe --list-devices` and prints the result. If no device line such as `CUDA0: NVIDIA GeForce RTX 3090 (…)` or `Vulkan0: …` appears, it prints a yellow warning.
11. **Reports.** It prints `Installed: <path>`. If you used a non-default `-Destination`, a reminder follows that Kumaru only looks in `tools\llama.cpp\` and on `PATH`, with the path to set as `backend.llama_server.binary`. Last comes `Next:  kumaru serve -c qwen36`.

Nothing is installed system-wide, and nothing is added to `PATH`. `tools\llama.cpp\` is ignored by git, and deleting it uninstalls everything. The zips stay cached in `%TEMP%\kumaru-llama-<Version>\`: about 580 MB for CUDA and 20 MB for CPU. Delete that folder to reclaim the space.

### The marker file

`tools\llama.cpp\kumaru-install.json` records what is installed:

```json
{
    "version":  "b11541",
    "flavor":  "cuda-13.4",
    "pinned":  true,
    "installed_at":  "2026-10-10T15:14:38.3773620-07:00"
}
```

- `pinned` is `true` when the installed version is the script's pin, and `false` when you chose another one with `-Version`.
- Markers written by the first version of this script have no `pinned` field. They are treated as following the pin.
- Only this script reads the marker. Kumaru itself doesn't.

## Why it exists

- **New architectures land in llama.cpp first.** The Python bindings (`llama-cpp-python`, the `llamacpp` backend) lag by weeks. Building them on Windows needs a CUDA toolchain plus MSVC. A prebuilt release zip needs neither, and upgrading means changing one tag.
- **Pinned for reproducibility.** Benchmark numbers are only comparable between runs on the same engine build, so the default is one fixed, tested release, not "latest".
  - When a Kumaru update moves the pin, a plain re-run moves you with it.
  - Your own `-Version` choice is remembered, so a plain re-run never silently swaps your engine.
- **Never half-installed.** The new build is downloaded, verified and test-started before the swap. The swap itself is two folder renames. Every failure before the swap leaves the old install exactly as it was.
- **Verified downloads.** A truncated or corrupted zip, or one altered by a proxy, is caught before anything is unpacked.
  - The digests come from GitHub's API over HTTPS, so they share their root of trust with the download.
  - They don't protect against a compromised llama.cpp release itself. Only a signature check could, and this script performs none.
- **Repo-local.** Every checkout carries its own engine, so two worktrees can run different llama.cpp builds side by side.
- **Probes instead of hope.** A CUDA build on a driver that is too old still starts, but it finds no GPU and runs on the CPU. The `--list-devices` probe makes that visible at install time, rather than later as a mysteriously slow server.

## Parameters

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `-Version` | string | your earlier `-Version` choice, else `$PinnedVersion` (`b11541`) | Release tag (`bNNNNN`). A tag other than the pin is remembered (`"pinned": false`), and later plain runs keep it. Passing the pinned tag returns you to following the pin. |
| `-Flavor` | `cuda-13.4` \| `cuda-12.4` \| `vulkan` \| `cpu` | the installed flavor, else `cuda-13.4` | Which build. Any other value is rejected by `ValidateSet`. |
| `-Destination` | string | `<repo>\tools\llama.cpp` | Install folder. It must be new, empty, or a previous install; anything else is refused. The backend auto-detects only the default location. |
| `-Force` | switch | off | Reinstall even when up to date, and download the zips again instead of using the cache. |

Choosing `-Flavor`:

| Your machine | Flavor |
|---|---|
| NVIDIA, driver 580 or newer (`nvidia-smi` → "Driver Version") | `cuda-13.4` (default) |
| NVIDIA, driver 551–579 | `cuda-12.4` |
| AMD or Intel GPU | `vulkan` |
| No usable GPU | `cpu` |

The pin is the line `$PinnedVersion = "b11541"` near the top of the script. Bumping it is a deliberate engine upgrade for everyone who follows the pin. Re-run the benchmarks when you do (see [BENCHMARKS.md](../../BENCHMARKS.md)).

## How to use it

```powershell
.\scripts\get-llama-server.ps1                     # NVIDIA, CUDA 13.4 build (driver 580+)
.\scripts\get-llama-server.ps1 -Flavor cuda-12.4   # NVIDIA with an older driver (551+)
.\scripts\get-llama-server.ps1 -Flavor vulkan      # AMD / Intel GPUs
.\scripts\get-llama-server.ps1 -Flavor cpu         # no GPU at all
.\scripts\get-llama-server.ps1 -Version b11600     # a different llama.cpp release
.\scripts\get-llama-server.ps1 -Force              # download again and reinstall
```

The script finds the repo from its own location, so it works from any current directory. Re-running is safe: with no arguments it keeps your flavor and does nothing when the build is already in place. A different version or flavor reinstalls without `-Force`.

### Upgrade (or roll back) llama.cpp

1. Pick a tag from https://github.com/ggml-org/llama.cpp/releases and run `.\scripts\get-llama-server.ps1 -Version bNNNNN`.
   - Your flavor is remembered.
   - Kumaru may keep running. Its `llama-server` keeps the old build until you restart.
2. Restart Kumaru and check the build: `(Invoke-RestMethod http://127.0.0.1:8080/v1/info).agent.backend.extra.build`.
3. Re-run your benchmark baseline before comparing scores across builds.
4. To return to the tested build, run `.\scripts\get-llama-server.ps1 -Version b11541` (the pin). From then on, plain runs follow the pin again.

Upgrade when a model fails with an unknown-architecture error in the llama-server log, or for new kernels and fixes.

**After a `git pull` that moves the pin**, run the script with no arguments:
- if you follow the pin, it installs the new pinned build;
- if you chose a version yourself, it keeps yours and names the pinned tag in yellow.

**If a release renames its zips**, for example `cuda-13.5` instead of `cuda-13.4`, the script stops and lists the flavors the release has. Add the new name to `-Flavor`'s `ValidateSet` at the top of the script, and to the table above.

### Other platforms, or your own build

The asset names are fixed to `win-…-x64`. On Linux or macOS, or with a custom Windows build, install llama.cpp yourself. Then either put `llama-server` on `PATH` (the backend searches it after `tools\llama.cpp\`), or set `backend.llama_server.binary` to the full path.

## Verifying it worked

```powershell
Get-Content .\tools\llama.cpp\kumaru-install.json     # version, flavor, pinned, installed_at
Test-Path .\tools\llama.cpp\llama-server.exe          # True
```

The script's own output is the real check:
- `checksum OK (sha256:…)` appears under every zip;
- `Test-starting the new build` prints the llama.cpp build;
- for GPU flavors, `Compute devices visible to llama.cpp` lists your GPU, for example `CUDA0: NVIDIA GeForce RTX 3090 (…)`;
- no yellow warning appears.

`.\scripts\verify.ps1` re-checks the binary as stage 6, and its GPU visibility under `-Full`.

End to end, `kumaru serve -c qwen25-7b-gguf` (4.7 GB download) or `-c qwen36` should log `starting llama-server on port …`, and `/v1/info` should report `extra.build`.

## Testing changes to the script

Never test edits against your real `tools\llama.cpp\`. Point the script at a scratch folder with the 20 MB CPU build, and give it its own download cache. `GetTempPath()` reads `TMP`/`TEMP`, so the cache moves too. Do this in a **new** PowerShell window, because the variables last until you close it:

```powershell
$scratch = "$env:TEMP\kumaru-installer-test"
New-Item -ItemType Directory -Force "$scratch\tmp" | Out-Null
$env:TEMP = "$scratch\tmp"; $env:TMP = "$scratch\tmp"
.\scripts\get-llama-server.ps1 -Flavor cpu -Destination "$scratch\dest"
```

These scenarios were checked under both Windows PowerShell 5.1 and PowerShell 7. Several of them need an update that does not download anything. To get one, edit `dest\kumaru-install.json` and set `"version"` to `"b11540"`: the next run then "updates" to the pin from the cached zip.

| Scenario | Set-up | Expected |
|---|---|---|
| Fresh install | the command above | `checksum OK`, the build printed, `Installed:` |
| Up to date | the same command without `-Flavor` | `already installed`, nothing downloaded |
| Damaged install | delete `dest\llama-server.exe`, re-run | reinstalls from the cached zip |
| Corrupt download | append a byte to the cached zip, then force an update | `Checksum mismatch`, zip deleted, install untouched; the next run downloads it again |
| Pin moved | marker `"version": "b11540", "pinned": true` | updates to the pin |
| Your own version | marker `"version": "b99999", "pinned": false` | kept, with the yellow pin hint |
| Bad tag | `-Version b0` | `has no release`, nothing touched, exit code 1 |
| Update while running | copy `PING.EXE` into `dest`, start it from there with `-n 60 127.0.0.1`, force an update | swap succeeds; the yellow "still running" note; `dest.old-*` is removed by the first run after `PING` exits |
| Folder in use | start a process with `-WorkingDirectory dest`, force an update | `Could not move the current install aside`, nothing changed |
| Not an install | `-Destination` at a folder holding other files | refused, files untouched |
| API unreachable | PowerShell 7 with `$env:HTTPS_PROXY = 'http://127.0.0.1:9'`, force an update | yellow warning, installs unverified from the cached zip |

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `running scripts is disabled on this system` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `Cannot validate argument on parameter 'Flavor'` | A typo in `-Flavor` | Use one of the four listed values |
| `llama.cpp has no release 'bNNNNN'` | A typo in `-Version`, or the tag does not exist | Check the releases page. Tags look like `b11541`. |
| `Release bNNNNN has no <zip>. Its Windows x64 builds are: …` | That release renamed or dropped the flavor | Pick a listed flavor. If it was renamed, extend the `ValidateSet` (see Upgrade). |
| Yellow `Could not read the release from the GitHub API … not be checksum-verified` | Offline, a proxy, or the hourly API limit | The install continues unverified. For verification, retry later or from another network. |
| `Download failed (curl exit code N): …` | Offline or proxy | Fix the network and re-run. The unfinished `.part` file is never used; the next run starts it over. |
| `Checksum mismatch for …` | A corrupted or truncated zip, or one altered in transit | Re-run: the bad zip was deleted and is downloaded again. If it repeats, distrust that network. |
| `… is not empty and holds no llama.cpp install, so it will not be replaced` | `-Destination` points at a folder with other files | Use a new or empty folder |
| `The new llama-server.exe did not run (exit code N)` | A broken build, or a DLL blocked by antivirus | The output shows llama-server's own error, and your install is untouched. Try another `-Version`. |
| `Could not move the current install aside … A program is using the folder` | A terminal or other process has its current directory inside the install | `cd` out of it or close it, then re-run |
| `Could not move the new build into place … Nothing was changed.` | Antivirus or another program held the new folder for longer than the script's ~2 seconds of retries | Re-run |
| Yellow `The previous build is still running …` | Kumaru (`kumaru serve`, or a `kumaru bench` run) was using llama-server during the update | Not an error. Restart Kumaru to use the new build. A later run deletes the old folder. |
| `Could not remove the leftover folder …\llama.cpp.new` | An earlier run was killed mid-install, and something holds a file in it | Delete the folder by hand |
| `llama-server.exe is not in llama-…zip; the release layout may have changed.` | The zip layout changed | Inspect the zip. Install manually and set `backend.llama_server.binary`. |
| Yellow `No CUDA device detected…` | A driver older than 580 with `cuda-13.4`, no NVIDIA GPU, or a driver problem | Check `nvidia-smi`. Re-run with `-Flavor cuda-12.4`, or use `vulkan`/`cpu`. |
| Yellow `No Vulkan device detected…` | A GPU driver without Vulkan support | Update the driver, or use `-Flavor cpu` |
| Kumaru does not find an install in a custom `-Destination` | The backend only checks `tools\llama.cpp\` and `PATH` | Set `backend.llama_server.binary` to the path the script printed |

## Related files

- `src\kumaru\backends\llamacpp_server.py` — `BUNDLED_BINARY` and `find_llama_server`: explicit `binary`, then `tools\llama.cpp\`, then `PATH`.
- `configs\qwen36.yaml`, `configs\qwen25-7b-gguf.yaml` — the profiles that need this binary.
- `scripts\verify.ps1` — stage 6 runs the installed binary; `-Full` checks that it sees the GPU.
- `scripts\setup.ps1` — the Python environment. This script is independent of it and needs no Python.
- `.gitignore` — excludes `tools\llama.cpp\` from the repo.
