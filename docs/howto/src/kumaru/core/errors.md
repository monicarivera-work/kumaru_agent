# `src\kumaru\core\errors.py`

> Exception hierarchy for deliberate Kumaru failures; each carries a message, an optional operator hint, and the HTTP status the server returns for it.

**Read this when:** a command or request fails with a Kumaru error, you add a new failure type, or you need to know which HTTP status a client will see and why.

---

## What it does

This file defines one base exception, `KumaruError`, and eight subclasses. Each carries three things:

- `message`: what happened.
- `hint`: what to do next. It is optional.
- `http_status`: the status the server answers with.

It also defines `install_command(extra)`, which spells out how to install one of Kumaru's optional extras. `DependencyMissingError` and the training preflight in `src\kumaru\train\runner.py` put it in their hints.

There are two consumers:

- **The CLI.** `kumaru.cli.main()` catches every `KumaruError`. It prints `error: <message>`, plus `hint: <hint>` when there is one, and exits with code 1.
- **The server.** One FastAPI exception handler in `src\kumaru\server\app.py` turns any `KumaruError` into a JSON error body with the class's `http_status`.

v0.2 adds `RequestError` (400) and `AuthError` (401). Before them, both failures surfaced as the base class's 500.

## Why it exists

The server must map failures to status codes, and the CLI must print actionable text. With a single exception type, both would string-match error messages, and that breaks silently when wording changes. With one base class plus specific subclasses, both callers branch on *type*.

The chosen statuses matter to clients:

- **401 for a bad API key, not 500.** HTTP clients, the OpenAI SDK included, retry 5xx responses. A mistyped key would otherwise cost several slow, misleading retries.
- **503 for backend trouble.** It is the retryable "not now" status. A server whose model is still loading should be retried.
- **4xx for request problems.** They tell the caller to change the request rather than retry it.

## Mental model

- Raise a specific subclass for failures that a user or operator can act on. Leave programmer bugs as ordinary exceptions. They then surface as a traceback in the CLI and a plain 500 over HTTP, and are not mistaken for handled conditions.
- `message` and `hint` are separate fields. `str(exc)` joins them as `"message (hint)"` for logs. Interfaces print them separately.
- A hint can come from three places:
  - the `raise` site (`hint=...`);
  - a class-level default (`hint = "..."` on the subclass);
  - a custom constructor, as `DependencyMissingError` does.

  `__init__` sets the instance hint only when it is not `None`, so a class default survives `hint=None`.
- `http_status` is a class attribute that is read only by the server's exception handler. The CLI always exits 1, whatever the status.
- **Install hints never say `pip install "kumaru[...]"`.** Kumaru installs from its source folder, not from PyPI, so that command would ask the package index for a different project. `install_command()` returns one of two forms:
  - on Windows, the `setup.ps1` switch when one exists. PyTorch's Windows wheel on PyPI is CPU-only, so `torch` and `train` must go through `-Gpu`, which installs a CUDA build first;
  - otherwise, `pip install -e ".[<extra>]" in the Kumaru folder`. `setup.ps1` is Windows-only, and PyTorch's Linux wheels on PyPI include CUDA.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `KumaruError` | `KumaruError(message: str, *, hint: str \| None = None)` | Base class for deliberate errors; `http_status = 500`, `hint = None`. |
| `KumaruError.message` | `str` | The error summary without the hint. |
| `KumaruError.hint` | `str \| None` | Operator guidance; class default unless a non-`None` `hint` is passed. |
| `KumaruError.http_status` | `int` (class attribute) | Status used by the server's exception handler. |
| `KumaruError.__str__` | `__str__(self) -> str` | `"message (hint)"`, or `message` when there is no hint. |
| `ConfigError` | `ConfigError(message: str, *, hint: str \| None = None)` | Missing, malformed, or inconsistent configuration or data files; `http_status = 400`. |
| `RegistryError` | `RegistryError(message: str, *, hint: str \| None = None)` | Unknown or duplicate registered component; `http_status = 400`. |
| `RequestError` | `RequestError(message: str, *, hint: str \| None = None)` | Well-formed API request that asks for something invalid; `http_status = 400`. |
| `AuthError` | `AuthError(message: str, *, hint: str \| None = None)` | Missing or wrong API key; `http_status = 401`. |
| `BackendError` | `BackendError(message: str, *, hint: str \| None = None)` | Backend failed to load or to generate; `http_status = 503`. |
| `DependencyMissingError` | `DependencyMissingError(package: str, *, extra: str \| None = None)` | Optional dependency missing; subclass of `BackendError`; `http_status = 501`. Message `Optional dependency '<package>' is not installed.`; hint `Install it with: <install_command(extra)>`, or `Install it with: pip install <package>` without `extra`. Sets `.package`. |
| `install_command` | `install_command(extra: str, *, windows: bool = os.name == "nt") -> str` | The install instruction for an optional extra (table below). The `windows` default is evaluated once, at import; tests pass it explicitly. |
| `_SOURCE_INSTALL` | `'pip install -e ".[{extra}]" in the Kumaru folder'` | Private format string: the source install, used on every OS. |
| `_SETUP_INSTALL` | `dict[str, str]`, keys `torch`, `train`, `quant`, `gguf` | Private: the Windows commands that name a `setup.ps1` switch. |
| `ContextLengthExceededError` | `ContextLengthExceededError(message: str, *, hint: str \| None = None)` | Conversation does not fit the token budget; `http_status = 413`. |
| `ToolError` | `ToolError(message: str, *, hint: str \| None = None)` | A tool failed while executing; `http_status = 500`. |

### `install_command()` output

| `extra` | Windows | Other OSes |
|---|---|---|
| `torch`, `train` | `.\scripts\setup.ps1 -Gpu (a CUDA build of PyTorch, then pip install -e ".[train]")` | `pip install -e ".[<extra>]" in the Kumaru folder` |
| `quant` | `.\scripts\setup.ps1 -Quant, or pip install -e ".[quant]" in the Kumaru folder` | `pip install -e ".[quant]" in the Kumaru folder` |
| `gguf` | `.\scripts\setup.ps1 -Gguf, or pip install -e ".[gguf]" in the Kumaru folder` | `pip install -e ".[gguf]" in the Kumaru folder` |
| anything else, e.g. `bench` | `pip install -e ".[bench]" in the Kumaru folder` | the same |

The extras the code asks for today:

- `torch`: `transformers` for the transformers backend, and `torch` for nano pretraining.
- `train`: `peft` for the transformers backend's adapter; `transformers`, `peft` and PyTorch for LoRA training.
- `quant`: `bitsandbytes`.
- `gguf`: `llama-cpp-python`.
- `bench`: `datasets`, plus `langdetect`, `nltk` and `regex` for IFEval.

### Exception → HTTP status

The single handler, `kumaru_error_handler` in `src\kumaru\server\app.py`, does three things for any `KumaruError`:

- returns `exc.http_status`;
- returns the body `{"error": {"message": exc.message, "type": "<ClassName>", "hint": exc.hint}}`;
- increments `metrics.errors`.

The last column below lists what actually reaches an HTTP client in v0.2.

| Exception / condition | Status | Raised over HTTP by (v0.2) |
|---|---|---|
| `RequestError` | 400 | `POST /v1/chat/completions` with an empty `messages` list. `POST /v1/backend` with no allowed field (the hint lists the allowed fields). |
| `ConfigError`, `RegistryError` | 400 | No route raises them during a request. If they occur while the backend loads, they reach clients as the 503 below. |
| `AuthError` | 401 | `require_auth` on `POST /v1/chat/completions` and `POST /v1/backend` when `server.api_key` is set. |
| `ContextLengthExceededError` | 413 | `agent.memory_policy: none` and a conversation over the budget, in non-streaming requests. |
| FastAPI `RequestValidationError` | 422 | The body violates the request schema: a bad `role`, an out-of-range `temperature`, a negative `thinking_budget`, a missing `messages`, or invalid JSON. Not a Kumaru error, so the body is FastAPI's `{"detail": [...]}`, and it is not counted in `metrics.errors`. |
| `ToolError` | 500 | Not raised over HTTP: `Tool.safe_run` turns tool failures into text for the model. |
| Any other exception | 500 | A bug in a non-streaming request. Starlette answers with a plain-text `Internal Server Error` body, not the JSON shape above. |
| `DependencyMissingError` | 501 | No request-time raise site. Dependencies are imported while the backend loads, so a missing one reaches clients as the 503 below. |
| `BackendError` | 503 | Four cases. (1) `Backend failed to load: <str(exc)>`: the background load raised anything; the original message and hint end up in this message. (2) `Backend is still loading.`: no backend after `ensure_ready` waited 600 s. (3) `Failed to load new backend: ...`: a hot-swap failed. (4) The backend failed during a non-streaming generation. |
| Any exception after an SSE stream started | 200 | The stream carries `data: {"error": {"message", "type", "hint"}}`, then `data: [DONE]`. A non-Kumaru exception produces no `hint` key, and `message` is `str(exc)`. This includes `ContextLengthExceededError` with `stream: true`. |

## How to use it

### Raise with a hint at the call site

```python
from kumaru.core.errors import BackendError

raise BackendError(
    f"Upstream returned {status}: {body}",
    hint="Check that backend.base_url points at a running OpenAI-compatible server.",
)
```

### What the operator sees

The CLI output below was captured from `kumaru config -c nope`:

```text
error: Config file not found: nope
hint: Available profiles: cpu-only, default, local-gpu, low-vram, qwen25-7b-gguf, qwen36, test, vllm
```

The exit code is 1. Over HTTP, this is the body captured with `server.api_key` set and no `Authorization` header:

```json
HTTP 401
{"error": {"message": "Invalid or missing API key.", "type": "AuthError", "hint": "Send 'Authorization: Bearer <key>'."}}
```

### Catch all deliberate errors

```python
from kumaru.core.errors import KumaruError

try:
    run_operation()
except KumaruError as exc:
    print(f"error: {exc.message}")
    if exc.hint:
        print(f"hint: {exc.hint}")
```

### Report a missing optional package

```python
from kumaru.core.errors import DependencyMissingError

raise DependencyMissingError("llama-cpp-python", extra="gguf")
# message: Optional dependency 'llama-cpp-python' is not installed.
# hint:    Install it with: .\scripts\setup.ps1 -Gguf, or pip install -e ".[gguf]" in the Kumaru folder
#          (other OSes: Install it with: pip install -e ".[gguf]" in the Kumaru folder)
```

Outside `DependencyMissingError`, build the same hint with `install_command()`, as the training preflight does:

```python
from kumaru.core.errors import ConfigError, install_command

raise ConfigError("PyTorch is not installed.", hint=f"Install it with: {install_command('train')}")
```

Check the output for both platforms from any machine:

```powershell
.\.venv\Scripts\python.exe -c "from kumaru.core.errors import install_command as i; print(i('quant', windows=True)); print(i('quant', windows=False))"
```

## How to extend it

Add a subclass when callers need to branch on a distinct failure, or the failure needs a distinct status. Do not create one for a wording difference; use the existing class with a clearer message and hint.

1. **Subclass the closest parent.** For example, subclass `BackendError` for a backend failure, so existing `except BackendError` sites still catch it.
2. **Set `http_status` deliberately.**
   - Use a 4xx status when the caller must change the request. Clients do not retry 4xx.
   - Use 503 when retrying later can succeed.
   - Use 500 for the server's own fault.
3. **Give it a hint.** Use a class-level `hint` when the remedy is always the same, and pass `hint=` at the raise site when it depends on context. A non-`None` raise-site hint overrides the class default.
4. **Raise it.** You do not need to change the server or the CLI: the server handler maps every `KumaruError` by `http_status`, and `main()` prints every one.
5. **Add a test** that asserts the status and `error.type`, for example with FastAPI's `TestClient` as in `tests\test_server.py`.

The behaviour of this example was verified:

```python
from kumaru.core.errors import KumaruError

class RateLimitedError(KumaruError):
    """A caller exceeded its request quota."""

    http_status = 429
    hint = "Wait a minute, then retry."

exc = RateLimitedError("Too many requests.")
exc.hint                                            # 'Wait a minute, then retry.'
str(exc)                                            # 'Too many requests. (Wait a minute, then retry.)'
RateLimitedError("Too many requests.", hint="Upgrade your plan.").hint   # 'Upgrade your plan.'
```

Raised in a route, it produces a 429 with `{"error": {"message": "Too many requests.", "type": "RateLimitedError", "hint": "Wait a minute, then retry."}}`. In the CLI, it prints `error:` and `hint:` and exits 1.

For a hint that must be computed, override `__init__` and pass `hint=` to `super().__init__`, the way `DependencyMissingError` builds its install command.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| CLI prints a Python traceback for an operator error | Code raised a generic exception, or the command ran through Typer's `app()` instead of `kumaru.cli.main()`. `app()` re-raises exceptions. | Raise the right `KumaruError` subclass; run through the `kumaru` console script, which calls `main()`. |
| HTTP 500 with a plain-text `Internal Server Error` body | A non-Kumaru exception escaped a non-streaming route. | Read the traceback in the server log; raise a `KumaruError` subclass with a status and hint. |
| A config or registry mistake arrives as 503, not 400 | The backend loads in the background. `_load()` stores `str(exc)` in `load_error`, and `ensure_ready()` re-raises it as `BackendError`. | Read `error.message`, which holds the original message and hint. Fix the config, then `POST /v1/backend` or restart. |
| Streaming request returns HTTP 200 but no answer | The failure happened after the SSE response began, so it arrived as an `error` frame. | Check each frame for an `error` key before reading `choices`. |
| Client SDK retries several times before failing | 5xx statuses (503 while loading, 500) are retried by OpenAI-style SDKs. | Expected for 503 while loading. Make permanent failures 4xx classes. |
| No install command for a missing optional dependency | Code raised `BackendError` instead of `DependencyMissingError`. | Raise `DependencyMissingError(package, extra="...")`. |
| The hint's `pip install -e ".[<extra>]"` fails with `... does not appear to be a Python project` | It ran outside the Kumaru folder; `-e "."` installs the project in the current directory. | `Set-Location` to the folder that holds `pyproject.toml`, and use the venv's pip (`.\.venv\Scripts\python.exe -m pip ...`). |
| Error handling branches on message text | A distinct error type is missing. | Add or use a specific subclass and branch on type. |
| The hint appears twice, or in parentheses | `str(exc)` already appends the hint. | Format with `exc.message` and `exc.hint` separately. |

## Related files

- `src\kumaru\server\app.py` — `kumaru_error_handler` maps every `KumaruError` to JSON. `ensure_ready()` and `swap_backend()` raise the 503s.
- `src\kumaru\server\routes_chat.py` — raises `AuthError` and `RequestError`; turns errors during streaming into SSE `error` frames.
- `src\kumaru\cli.py` — `main()` prints `error:`/`hint:` and exits 1; Ctrl-C exits 130.
- `src\kumaru\core\config.py` — raises `ConfigError` for missing profiles, non-mapping YAML, and invalid values.
- `src\kumaru\core\registry.py` — raises `RegistryError` for duplicate and unknown names.
- `src\kumaru\agent\memory.py` — the `none` policy raises `ContextLengthExceededError`.
- `src\kumaru\agent\tools.py` — `Tool.safe_run` turns `ToolError` and other failures into text for the model.
- `src\kumaru\train\runner.py` — the training preflight; its PyTorch and bitsandbytes hints come from `install_command()`.
- `scripts\setup.ps1` — the `-Gpu`, `-Quant` and `-Gguf` switches the Windows hints name; see [setup.ps1.md](../../../scripts/setup.ps1.md).
- `pyproject.toml` — the optional extras: `torch`, `train`, `quant`, `gguf`, `bench`, `dev` and `all`.
