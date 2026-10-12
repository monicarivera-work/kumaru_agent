# `pyproject.toml`

> The package definition: Kumaru's name and version, what `pip` installs (a light base plus optional extras), the `kumaru` command, and the settings for pytest, ruff and mypy.

**Read this when:** you are adding a dependency, installing for a new job (training, benchmarks, GGUF), releasing a version, or pytest, ruff or mypy behaves differently from what you expect.

---

## What it does

`pip install -e ".[dev]"`, which is what `scripts\setup.ps1` runs, reads this file and:

1. installs the **base dependencies**: enough to chat through the server, the UI and the CLI;
2. installs each **extra** named in the brackets (here `dev`: the test tools);
3. links `src\kumaru` into the virtual environment **editable** (`-e`), so code edits take effect without reinstalling;
4. creates `.venv\Scripts\kumaru.exe`, which runs `kumaru.cli:main`.

The `[tool.*]` tables configure pytest, ruff and mypy, so the bare commands (`pytest`, `ruff check`, `mypy src`) need no flags.

## Why it exists

Every modern Python tool reads this one file (the standard is PEP 621), so it replaces `setup.py`, `requirements.txt`, `pytest.ini` and the linter config files.

The decision that matters is **a light base and heavy extras**:

- **The base installs from prebuilt wheels.** Anyone can install it and reach a working UI without a GPU or a compiler. It took 95 seconds in a fresh virtual environment with an empty pip cache, including the test tools.
- **PyTorch (about 2.5 GB) is opt-in.** The right wheel depends on your CUDA version, and pip cannot work that out. That is why `setup.ps1 -Gpu` installs it from PyTorch's own index first.
- **The recommended model needs no extra at all.** The `qwen36` profile runs on `llama-server`, a separate binary that `scripts\get-llama-server.ps1` downloads, not a Python package. The `huggingface_hub` base dependency downloads its model file.

## Mental model

```
pip install -e ".[train,bench]"
             │    │     └─ extra: benchmark datasets and IFEval's checkers
             │    └─ extra: PyTorch, transformers, peft (fine-tuning)
             └─ editable: link src\ instead of copying it
```

Extras add up, and `all` means every extra. Which ones you need depends on the backend your profile uses:

| Profile | Backend | Needs |
|---|---|---|
| `test` | `echo` | base |
| `qwen36`, `qwen25-7b-gguf` | `llamacpp_server` | base, plus the binary from `get-llama-server.ps1` |
| `vllm` | `openai_compat` | base, plus a running vLLM server |
| `default`, `local-gpu` | `transformers` | `torch` (via `setup.ps1 -Gpu`) |
| `low-vram` | `transformers`, 4-bit | `torch` and `quant` (`setup.ps1 -Gpu -Quant`) |
| `cpu-only` | `llamacpp` (in-process) | `gguf` |

`kumaru serve` with no `-c` uses `default`, so on a base install pass `-c qwen36` or `-c test`.

## What installs what

### Base dependencies

| Package | Why Kumaru needs it |
|---|---|
| `fastapi` | The HTTP server and its routes (`server\`) |
| `uvicorn[standard]` | Runs the server; `[standard]` adds the fast HTTP parser, WebSockets and file watching |
| `pydantic` | Validates every config file and every API request (`core\config.py`, `server\schemas.py`) |
| `PyYAML` | Reads the profiles in `configs\` |
| `httpx` | The HTTP client for the `llamacpp_server` and `openai_compat` backends |
| `typer` | The `kumaru` command line (`cli.py`) |
| `rich` | Tables and colour in the CLI |
| `huggingface_hub` | Downloads GGUF model files for `llamacpp_server` |

### Extras

| Extra | Adds | Install it when | How |
|---|---|---|---|
| `torch` | torch, transformers, accelerate, safetensors, sentencepiece, tokenizers, numpy | Running a `transformers` profile | `.\scripts\setup.ps1 -Gpu` |
| `train` | `torch`, plus peft, datasets, tensorboard | Fine-tuning (LoRA) or pretraining | `.\scripts\setup.ps1 -Gpu` |
| `quant` | bitsandbytes | 4-bit loading and QLoRA | `.\scripts\setup.ps1 -Gpu -Quant` |
| `gguf` | llama-cpp-python | The in-process `llamacpp` backend (not `qwen36`) | `.\scripts\setup.ps1 -Gguf` |
| `bench` | datasets, langdetect, nltk, regex | `kumaru bench` on the public benchmarks | `pip install -e ".[bench]"` |
| `dev` | pytest, pytest-asyncio, ruff, mypy | Running the tests and the linters | `setup.ps1` always installs it |
| `all` | Every extra above | A workstation that does everything | `pip install -e ".[all]"`, after `setup.ps1 -Gpu` |

Two cautions:

- **Never let plain pip install torch on Windows.** PyPI's Windows torch wheel is CPU-only, and once any torch is installed, pip keeps it rather than replace it. Always run `setup.ps1 -Gpu` before installing `torch`, `train` or `all`.
- **`gguf` compiles from source.** PyPI ships current llama-cpp-python releases as source only, so the install needs a C/C++ compiler. On a machine without one, `.[all]` fails too. Install the extras you need instead. [`llamacpp_backend.md`](src/kumaru/backends/llamacpp_backend.md) has the details.

## The tool settings

**pytest** (`[tool.pytest.ini_options]`)

| Setting | Effect |
|---|---|
| `testpaths = ["tests"]` | A bare `pytest` finds the suite |
| `asyncio_mode = "auto"` | pytest-asyncio runs every `async def` test without a decorator |
| `filterwarnings` | Hides third-party `DeprecationWarning` noise |

A test that needs an extra calls `pytest.importorskip("numpy")` (or the extra's package), so a base install skips it instead of failing. On a base install the suite reports 322 passed and 8 skipped. [`tests/README.md`](tests/README.md) covers the suite.

**ruff** (`[tool.ruff]`, `[tool.ruff.lint]`): the linter, and the one check every change must pass.

| Setting | Effect |
|---|---|
| `line-length = 100`, `target-version = "py310"` | Style target; code may use Python 3.10 syntax |
| `src = ["src", "tests"]` | Tells import sorting which imports are Kumaru's own |
| `select = ["E", "F", "I", "UP", "B", "SIM"]` | Style errors, pyflakes (unused imports, undefined names), import order, modern syntax, likely bugs (bugbear), simplifications |
| `ignore = ["E501", "B008"]` | Long lines are allowed. B008 (a function call as an argument default) must stay off: Typer's `typer.Option(...)` defaults in `cli.py` are exactly that |

`ruff format` is **not** used. It has never been applied, and it would rewrite 36 of the 60 Python files. Run it across the project only in a dedicated commit, so real changes aren't buried in a formatting diff.

**mypy** (`[tool.mypy]`, `[[tool.mypy.overrides]]`): advisory, not enforced. `verify.ps1` does not run it.

| Setting | Effect |
|---|---|
| `python_version = "3.10"` | Checks against the oldest Python Kumaru supports, so it flags standard-library features 3.10 lacks |
| `ignore_missing_imports = true` | A package that isn't installed (torch on a base install) is typed `Any` instead of being an error |
| `warn_unused_ignores = false` | The default, made explicit. A `# type: ignore` that one platform needs isn't reported as unused on another: `llamacpp_server.py` reads `Popen._handle`, which the type stubs declare only on Windows |
| Override for `numpy.*` | Skips numpy's own type stubs, so numpy values are typed `Any`. numpy 2.5 and later write those stubs in Python 3.12 syntax, which a 3.10 check cannot parse. Without the override, any install with numpy (the torch extra) stops on `numpy\__init__.pyi` with "errors prevented further checking" before checking any Kumaru code |

As of 0.2.0, `mypy src` reports 9 errors in 5 files, all in code from 0.1.0 (type annotations only, no runtime bugs). The first run builds `.mypy_cache\` and is slow with torch installed; later runs take about 5 seconds. Treat a *new* error as a signal.

## How to use it

```powershell
pip install -e ".[dev]"            # what setup.ps1 runs
pip install -e ".[bench]"          # add the benchmark extra
pip install -e ".[train,bench]"    # several at once (after setup.ps1 -Gpu)
pip show kumaru                     # installed version, and where it points

ruff check src tests scripts        # lint: must pass
pytest tests -q                     # tests
mypy src                            # type check: advisory, 9 known errors
```

Run these from the activated venv in the repo root.

## How to extend it

- **Adding a dependency.** Put it in the base only when the chat path needs it and it is a small wheel. Anything else goes in an extra and is **imported inside the function that uses it**: `transformers_backend.py` imports torch inside `load()`, for example. That way a base install never breaks at import time. Guard its tests with `pytest.importorskip`. Then re-run `pip install -e ".[...]"`: an editable install picks up code changes, but not changes to this file.
- **Releasing a version.** Bump `version` here **and** `__version__` in `src\kumaru\__init__.py`. Nothing checks that they match. pip reports this file's version. `kumaru info`, `/healthz` and each benchmark run's `run.json` report `__version__`. The version is not part of a benchmark run's fingerprint, so a bump never forces a re-run.
- **Pinning an environment.** The `>=` floors accept newer releases. A clean install in October 2026 got FastAPI 0.143, Starlette 1.7, huggingface_hub 2.2 and ruff 0.17, and the suite and `ruff check` passed. To reproduce an environment exactly:

  ```powershell
  pip freeze --exclude-editable > constraints.txt          # on the machine that works
  pip install -e ".[dev]" -c constraints.txt              # on the new one
  ```

  `--exclude-editable` matters because a constraints file may not contain Kumaru's own `-e` line.
- **Package data.** `[tool.setuptools.package-data]` lists `py.typed`, which does not exist yet, so nothing is shipped. Create an empty `src\kumaru\py.typed` when another package needs to type-check against Kumaru. Note that `configs\` and `ui\` live outside the package. A non-editable `pip install .` therefore has no profiles and no UI, which is why every guide uses `-e`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `requires a different Python: ... not in '>=3.10'` | Python older than 3.10 | Install 3.10 or newer, delete `.venv`, re-run setup |
| `kumaru : The term 'kumaru' is not recognized` | The venv is not activated | `.\.venv\Scripts\Activate.ps1`, or call `.\.venv\Scripts\kumaru.exe` |
| A dependency you added, or a version bump, does not appear | Editable installs pick up code, not this file | Re-run `pip install -e ".[dev]"` (plus your extras) |
| `[WinError 32] ... kumaru.exe ... used by another process` | `kumaru serve` is running | Stop it, then re-run |
| `ModuleNotFoundError: No module named 'torch'` | A `transformers` profile (including `default`) on a base install | `.\scripts\setup.ps1 -Gpu`, or use `-c qwen36` |
| `kumaru info` says `cuda: unavailable` on an NVIDIA machine | A CPU-only torch is installed, and pip will not replace it | `pip uninstall -y torch`, then `.\scripts\setup.ps1 -Gpu` |
| `pip install -e ".[all]"` fails building `llama-cpp-python` | No C/C++ compiler | Leave out `gguf`: `pip install -e ".[train,quant,bench,dev]"` |
| `pip install -e ".[quant]"` fails on `bitsandbytes` | No prebuilt wheel for your platform | Skip `quant`; set `load_in_4bit: false` and use bf16 |
| `No module named pytest` when running the tests | Installed with plain `pip install -e .` | `pip install -e ".[dev]"` |

## Related files

- [`scripts/setup.ps1`](scripts/setup.ps1.md): installs the base, `dev`, and the extras you ask for
- [`scripts/verify.ps1`](scripts/verify.ps1.md): runs the test suite, and needs `dev`
- [`src/kumaru/__init__.py`](src/kumaru/__init__.md): `__version__`
- [`tests/`](tests/README.md): the suite configured here
- [`.gitignore`](.gitignore.md): what stays out of git, including everything `pip` builds
- [`docs/QUICKSTART.md`](../QUICKSTART.md): the install, step by step
