# `configs/test.yaml`

> No-model echo profile for fast installation, UI, routing, and streaming checks.

**Read this when:** you want to verify Kumaru without downloading weights, using a GPU, or debugging model code.

---

## What it does

This profile uses `backend.name: echo`. The echo backend returns deterministic canned responses and simulates token streaming with a small configurable delay. It exercises the CLI, FastAPI routes, UI, streaming parser, metrics, and config loading without loading a real LLM.

## When to use this profile

Use it for:

- install smoke tests;
- UI development;
- route and streaming debugging;
- failure isolation: if `test` works but a model profile fails, the bug is likely in model loading or inference.

## Every field explained

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `backend.name` | string | `echo` | Selects the no-model backend. | Keep for this profile. |
| `backend.model` | string | `echo` | Model label shown in metadata. | Keep as `echo`. |
| `backend.max_context_length` | int | `8192` | Simulated context cap. | Leave default unless testing context behavior. |
| `backend.extra.delay_s` | float | `0.01` | Simulated delay between output tokens. | Set `0` for instant tests; increase to test streaming UI. |
| `generation.max_tokens` | int | `256` | Reply cap passed through generation params. | Usually irrelevant for echo; keep small. |
| `generation.temperature` | float | `0.0` | Deterministic sampling setting. | Keep zero for repeatable tests. |
| `agent.max_history_tokens` | int | `2048` | Conversation history budget. | Small budget is enough for route tests. |
| `server.port` | int | `8080` | HTTP/UI port. | Change if occupied. |
| `server.max_concurrent_requests` | int | `16` | Admission limit. | Higher is safe because no GPU work occurs. |
| `telemetry.log_level` | string | `DEBUG` | Verbose logging. | Useful for tests and route debugging. |

## How to use it

Prerequisites:

```powershell
pip install kumaru
```

Run the server:

```powershell
kumaru serve -c test
```

Run one prompt:

```powershell
kumaru chat -c test -q "hello"
```

Inspect config:

```powershell
kumaru config -c test
```

## Tuning it

1. Set `backend.extra.delay_s: 0` for fastest tests.
2. Increase `backend.extra.delay_s` to make streaming UI states easier to observe.
3. Change `server.port` when running beside another profile.
4. Keep `generation.temperature: 0.0` for deterministic output.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `kumaru serve -c test` fails | Installation, config parsing, or server startup issue. | Run `kumaru info` and inspect the stack trace; no model download is involved. |
| UI cannot connect | Server not running or wrong port. | Open `http://127.0.0.1:8080` or match your changed port. |
| Streaming is too fast to see | `delay_s` is too low. | Raise `backend.extra.delay_s`. |
| A real model behavior does not reproduce | Echo is not an LLM. | Use this only to isolate non-model layers. |
| Concurrent tests fail unexpectedly | Port or shared localStorage collision. | Use separate ports or clear browser storage. |

## Related files

- `src/kumaru/core/config.py` — `backend.extra` accepts backend-specific keys.
- `src/kumaru/server/routes_chat.py` — streaming and health routes exercised by this profile.
- `ui/app.js` — streaming parser and status polling tested with echo.

