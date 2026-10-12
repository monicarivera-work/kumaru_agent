# `configs\test.yaml`

> The no-model profile: the `echo` backend behind the real CLI, agent, server, SSE stream and web UI. No GPU, no downloads, deterministic replies.

**Read this when:**
- you want to check an install, or the CLI/server/UI stack, in seconds;
- you are developing the UI, routes or streaming and don't want a model in the loop;
- you are bisecting a failure: if `test` works and a model profile does not, the problem is in the model layer;
- you want to watch the streaming or thinking UI states in slow motion.

---

## What it does

`backend.name: echo` selects [`EchoBackend`](../src/kumaru/backends/echo.md). It loads nothing and streams a reply built from the request:

```text
[echo:8f434346] You said: hi
Context: 2 message(s), ~218 prompt tokens, max_tokens=256, temperature=0.0.
```

- `8f434346` is the first 8 hex digits of the SHA-256 of the last user message, so the same question always gets the same reply.
- `Context:` reports how many messages the agent sent (here the system prompt plus the question), the backend's token estimate, and the `max_tokens` and `temperature` that reached the backend. That makes this profile a cheap check of parameter precedence: `kumaru chat -c test -t 0.3 --max-tokens 99 -q hi` reports `max_tokens=99, temperature=0.3`.
- With thinking on (`--think`, the UI's **Think** button, or `"thinking": true` in a request), echo first streams a stand-in thought on the reasoning channel (`StreamChunk.reasoning`, SSE `reasoning_content`), then the answer. A thinking budget caps the thought at that many 8-character pieces.

Everything above the backend is real: the agent's prompt and history handling, the FastAPI routes, SSE framing, metrics and the UI. `scripts\verify.ps1` uses this profile for its "agent (echo backend)" and "server + SSE streaming" checks, and the closing hint of `scripts\setup.ps1` suggests `kumaru serve -c test` to verify the stack without a model download.

## When to use this profile

| Use it for | Not for |
|---|---|
| Install smoke tests; checking that a machine can run the stack at all | Anything about answer quality, speed, memory or tokenization |
| UI, route and SSE development | Model-specific behaviour: chat templates, a real engine's `<think>` handling, `finish_reason` `length` (echo always finishes with `stop`) |
| Bisecting a failure between the application layer and the model layer | Measuring throughput: the timing is `delay_s`, not inference |
| Dry-running another profile's sampling with `-b echo` (see below) | |

## Every field explained

| Field | Value | Code default | Why this value |
|---|---|---|---|
| `backend.name` | `echo` | `transformers` | The no-model backend. |
| `backend.model` | `echo` | `Qwen/Qwen2.5-7B-Instruct` | A label; echo's `info()` reports `model_id` `echo` whatever this says. |
| `backend.max_context_length` | `8192` | `8192` | Shown by `info()` (the CLI prints "8,192 token context") and used by the agent to size the history budget. Echo enforces no window. |
| `backend.extra.delay_s` | `0.01` | `0.01` (echo's fallback in `EchoBackend.__init__`) | Sleep before every streamed piece. Non-zero so the stream really awaits between pieces, as a real engine does. See [Choosing `delay_s`](#choosing-delay_s). |
| `generation.max_tokens` | `256` | `1024` | Only echoed back in the reply. Echo never truncates. |
| `generation.temperature` | `0.0` | `0.7` | Only echoed back. Echo is deterministic at any temperature. |
| `agent.max_history_tokens` | `2048` | `6144` | The agent's history budget: `min(2048, max(512, 8192 − min(256, 8192 // 2) − 128))` = 2,048 estimated tokens before older turns are trimmed. |
| `server.port` | `8080` | `8080` | Same as the default; stated so the profile says where the UI is (`http://127.0.0.1:8080`). |
| `server.max_concurrent_requests` | `16` | `8` | Size of the server's generation semaphore; further requests wait for a slot. Echo has no GPU to protect, so it is doubled. |
| `telemetry.log_level` | `DEBUG` | `INFO` | Verbose logs while debugging the application layer. Applied by `kumaru serve` only; `kumaru chat` and `kumaru bench run` read `$env:KUMARU_LOG_LEVEL` instead. |

Everything else takes the code default; `kumaru config -c test` prints the merged result. `generation.thinking` is unset, so thinking stays off until a flag, the UI or a request turns it on, and `thinking_mode` is empty, so turning thinking on changes no sampling value.

### Choosing `delay_s`

Echo sleeps `delay_s` before every piece (6-character answer pieces and, with thinking on, 8-character reasoning pieces) and skips the sleep entirely when it is `0`. Measured on Windows 11 for the 18-piece reply to `hello`; asyncio timers there tick about every 15.6 ms, so any delay below one tick rounds up to it:

| `delay_s` | Per piece | Whole reply | Use it for |
|---|---|---|---|
| `0` | ~0 | < 1 ms | Automated tests. The pytest suite never loads this profile: its fixtures (`tests\conftest.py`) build an echo config with `extra={"delay_s": 0}`. |
| `0.01` (profile) | ~15.5 ms | ~0.27 s | Smoke tests: the stream awaits between pieces, yet looks instant to a human. |
| `0.05` | ~61 ms | ~1.1 s | The smallest value at which you can watch the UI stream: the cursor, the live thinking panel, the **Stop** button. |
| `0.1` | ~108 ms | ~1.9 s | Slow-motion UI work, such as pressing **Stop** mid-thought. |

With thinking on, the thought for `hello` adds 12 pieces. With `0`, the generator never awaits, so nothing else on the event loop runs until the reply is complete. Tests of interleaving or cancellation need a non-zero delay.

## How to use it

### Smoke-test an install

```powershell
.\scripts\setup.ps1                                         # base install: no GPU deps, no model
.\.venv\Scripts\kumaru.exe chat -c test -q "hello"          # prints the echo reply and token counts
.\.venv\Scripts\kumaru.exe chat -c test --think --thinking-budget 2 -q "hello"
.\.venv\Scripts\kumaru.exe config -c test                   # merged config; loads nothing
.\scripts\verify.ps1                                        # staged self-check; its server stage uses port 8099
```

The `--think --thinking-budget 2` run prints `thinking…`, the two-piece thought `The user asked: `, then the answer; the token line counts 2 reasoning plus 18 answer pieces as 20 completion tokens.

### Serve the UI

```powershell
.\.venv\Scripts\kumaru.exe serve -c test                    # http://127.0.0.1:8080
.\.venv\Scripts\kumaru.exe serve -c test --port 8081        # beside another server
```

Click **Think** to exercise the reasoning panel. The UI's thinking-budget slider defaults to unlimited and its smallest step is 256 tokens, far above the length of echo's thought, so it has no visible effect here; use a request or the CLI to see truncation.

### Watch streaming in slow motion

Override the delay without editing the file. Environment values are parsed as YAML scalars, so `'0.05'` arrives as a float:

```powershell
$env:KUMARU_BACKEND__EXTRA__DELAY_S = '0.05'
.\.venv\Scripts\kumaru.exe serve -c test
Remove-Item Env:KUMARU_BACKEND__EXTRA__DELAY_S                # afterwards
```

`kumaru config -c test` shows `delay_s: 0.05` under `backend.extra` while the variable is set.

### Per-request thinking over the API

```powershell
$body = @{ messages = @(@{ role = "user"; content = "hello" }); thinking = $true; thinking_budget = 2 } | ConvertTo-Json -Depth 4
(Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/chat/completions -ContentType application/json -Body $body).choices[0].message
```

The message carries `content` (the echo reply) and `reasoning_content` (`The user asked: `). `usage.completion_tokens` is 20.

### Dry-run another profile's sampling

`-b echo` replaces only the backend; the profile's `generation` and `thinking_mode` still apply, and echo prints what reached it:

```powershell
.\.venv\Scripts\kumaru.exe chat -c qwen36 -b echo --think -q "hi"
```

This prints `max_tokens=32768, temperature=1.0`, the Qwen3.6 thinking preset, without starting llama-server or downloading anything.

## Tuning it

1. `backend.extra.delay_s`: `0` in automated tests, `0.01` for smoke tests, `0.05` or more to watch the UI (table above).
2. `server.port`, or `--port`, to run beside a model server; `verify.ps1` takes `-Port` (default 8099).
3. Leave the rest alone: echo ignores sampling, and the values only show up in its reply text.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `kumaru chat -c test` fails | Install or config problem; no model is involved. | Re-run `.\scripts\setup.ps1`; `kumaru config -c test` isolates config errors. |
| The UI shows each reply at once | `delay_s` is `0` or `0.01`; a whole reply takes under 0.3 s. | Set `KUMARU_BACKEND__EXTRA__DELAY_S` to `0.05` or more. |
| `serve` fails to bind port 8080 | Another server already uses the port. | `--port 8081`, or stop the other server. |
| An interleaving or cancellation test behaves as if the requests ran one after another | `delay_s: 0` never awaits, so nothing else runs until the reply completes. | Give that test a small non-zero delay. |
| The UI's "No answer: the max-tokens cap ran out while the model was still thinking" note can't be reproduced | Echo always finishes with `stop`, never `length`. | Use a real model, or a test double that emits `finish_reason="length"`. |
| The thinking-budget slider changes nothing | The slider's smallest budget is 256; echo's thought for a short question is about a dozen pieces. | Use `--thinking-budget 2` or `"thinking_budget": 2`. |
| `test` works but a model profile fails | The failure is in the model layer: dependencies, download, VRAM, engine. | Go to that backend's page, e.g. [llamacpp_server.md](../src/kumaru/backends/llamacpp_server.md). |

## Related files

- `src\kumaru\backends\echo.py` — the backend ([echo.md](../src/kumaru/backends/echo.md)).
- `tests\conftest.py` — the echo fixtures the test suite uses instead of this profile (`delay_s: 0`).
- `scripts\verify.ps1`, `scripts\setup.ps1` — use or suggest this profile ([verify.ps1.md](../scripts/verify.ps1.md), [setup.ps1.md](../scripts/setup.ps1.md)).
- `ui\app.js` — the streaming, thinking and stop handling this profile exercises.
- `configs\default.yaml` — every setting and its default ([default.yaml.md](default.yaml.md)).
