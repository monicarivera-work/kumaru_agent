# `scripts/verify.ps1`

> A layered self-check that proves each part of Kumaru works on this machine, in an order where the first failure identifies the broken layer.

**Read this when:** something is wrong and you do not yet know where; you have just upgraded a dependency; or you want to confirm a fresh install before relying on it.

---

## What it does

Runs six checks in dependency order — environment, config resolution, unit tests, the agent stack, the HTTP server with real SSE streaming, and (with `-Full`) the training and GPU paths. It prints a PASS/FAIL summary and exits non-zero if anything failed.

## Why it exists

"It doesn't work" is not a diagnosis. This system has six layers, and a symptom at the top (no reply in the browser) can originate at any of them. Checking them in order turns an open-ended debugging session into a bisection: **the first failing stage is the layer to fix, and every stage after it is noise.**

The ordering is the whole design. Each stage depends only on the stages before it:

| Stage | Proves | Needs a GPU? | Needs a download? |
|---|---|---|---|
| 1. environment | Python, packages, CUDA visibility | no | no |
| 2. config | Every YAML profile parses and validates | no | no |
| 3. unit tests | Core logic, types, policies, schemas | no | no |
| 4. agent (echo) | Prompt assembly, memory, backend contract | no | no |
| 5. server + SSE | HTTP routing, streaming protocol, static UI | no | no |
| 6. GPU + training | Real weights, real CUDA, LoRA pipeline, eval | yes | ~1 GB |

Stages 1–5 run in a few seconds with no GPU and no network. That is deliberate: it means you can run them constantly.

## Parameters

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `-Full` | switch | off | Adds stage 6: training dry run, GPU inference, and the eval harness |
| `-Port` | int | `8099` | Port for the temporary test server |
| `-SmallModel` | string | `Qwen/Qwen2.5-0.5B-Instruct` | Model used by stage 6 |

## How to use it

```powershell
# Fast: no GPU, no downloads. Run this often.
.\scripts\verify.ps1

# Complete: also exercises CUDA, LoRA, and eval. Downloads ~1 GB once.
.\scripts\verify.ps1 -Full

# Avoid a port conflict.
.\scripts\verify.ps1 -Port 8123
```

## Reading the output

```
================ SUMMARY ================
environment                  PASS
config                       PASS
unit tests                   PASS
agent (echo backend)         PASS
server + SSE streaming       FAIL: HTTP probe failed
=========================================
```

The example above says: Python, packages, config, core logic, and the agent are all fine — the problem is in the HTTP layer. That narrows an investigation from ~30 source files to three.

**Only the first failure matters.** Later stages depend on earlier ones, so they fail for derived reasons.

## What stage 5 actually checks

It starts a real server as a background job and probes it with `httpx`:

1. `GET /healthz` reports `status: "ok"`
2. `POST /v1/chat/completions` returns a non-empty completion
3. The same endpoint with `stream: true` emits multiple SSE frames terminating in `data: [DONE]`
4. `GET /` serves the UI

That is the full request path, including the streaming protocol that a plain health check would miss. The server is always stopped in a `finally` block, so a failure does not leave a process holding your port.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `No .venv found` | Setup not run | `.\scripts\setup.ps1` |
| Stage 1 fails | Broken install | Re-run setup; check `kumaru info` output |
| Stage 2 fails | A YAML profile has an invalid key | The error names the file and key; configs use `extra="forbid"` |
| Stage 3 fails | A genuine regression in core logic | `pytest tests -q` for the full traceback |
| Stage 5 fails, others pass | Port in use, or firewall blocking localhost | `-Port 8123`, or check the job output |
| Stage 6 fails with CUDA OOM | Another process is holding VRAM | `nvidia-smi` to find it |
| Stage 6 fails to download | No network, or HF rate limit | Set `HF_TOKEN`, or skip `-Full` |

## Extending it

Add a stage with the `Check` helper. Place it in dependency order — after everything it relies on:

```powershell
Check "my new subsystem" {
    & $kumaru my-command --flag | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "my-command failed" }
}
```

`Check` catches exceptions, records PASS/FAIL, and keeps going, so one failure still gives you a complete picture rather than stopping at the first problem.

## Related files

- `scripts/setup.ps1` — run this first
- `tests/` — the suite invoked by stage 3
- `configs/test.yaml` — the echo profile used by stages 4 and 5
- `docs/QUICKSTART.md` — the troubleshooting table for individual symptoms
- `GUIDE.md` §5 — the same debugging ladder, done by hand
