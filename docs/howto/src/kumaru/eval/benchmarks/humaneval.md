# `src\kumaru\eval\benchmarks\humaneval.py`

> HumanEval: 164 Python functions to write from a signature and docstring. Each answer is graded pass/fail by running the dataset's hidden unit tests on your machine, so the benchmark is opt-in.

**Read this when:** you are about to pass `--allow-code-exec`, a HumanEval score looks wrong, or you need to know what model-written code can and can't do to your machine.

---

> **Without `--sandbox docker`, this is not a sandbox.** With `--allow-code-exec`, code a model wrote runs under your user account. The safeguards stop a model's accidents. They do not stop code written to do harm. Evaluate models you don't trust with `--sandbox docker` or in a VM ([Run it contained](#run-it-contained)).

## What it does

- **`HumanEval.load()`** reads `openai/openai_humaneval` (no config), split `test`, through `load_hf_rows` ([`base.md`](base.md)). That is 164 problems, and each becomes a `BenchItem`:
  - `id` is the dataset's `task_id`, e.g. `HumanEval/116`.
  - `prompt` is `PROMPT` with the problem's stub (imports, signature, docstring) inside a Python fence.
  - `meta` is `{"prompt": <stub>, "test": <hidden tests>, "entry_point": <function name>}`. Every hidden test defines `check(candidate)`.
  - There is no `answer` (records show `expected: null`) and no `group`. So `-n` draws a plain seeded sample of the 164 (`select_items`), and `--full` runs all of them.
- **`HumanEval.grade()`** takes three steps:
  1. `assemble_program` joins the guard, the stub, the model's code, the hidden tests and `check(<entry_point>)` into one script.
  2. `run_program` appends a finish marker to the script and runs it through `codeexec` ([`codeexec.md`](codeexec.md)): a fresh Python process in an empty temporary folder, for at most `timeout_s` = 15 seconds. With `--sandbox docker`, it runs in a throwaway container instead.
  3. It returns `Grade(correct=status == "passed", extracted=status, detail={"status": status, "stderr": ...})`. `passed` needs exit code 0 **and** the finish marker in stdout ([Mental model](#mental-model)). Exit code 0 without the marker is `failed`, with `exited with code 0 before the tests finished (an early sys.exit?)` as the detail. Any other exit is `failed`, and the last 800 characters of stderr are kept. In the sandbox, a container its memory limit killed is `failed` with `killed: out of memory?`. Running past the limit is `timeout`. The `stderr` key appears only when there is some.

  The score is pass@1 from one sampled answer per problem: the share of problems whose only answer passed.
- **Three class settings change how the runner treats it:**
  - `needs_code_execution = True`. The runner refuses the benchmark without `--allow-code-exec`. The refusal happens while benchmark names are resolved, before any dataset download, model load or file write.
  - `blocking_grader = True`. The runner grades in a worker thread (`asyncio.to_thread`), so a slow program doesn't stall the other questions. Grading still holds its `-j` slot, and `codeexec` runs at most `KUMARU_CODE_WORKERS` programs at once.
  - `generation_overrides()` returns `{"presence_penalty": 0.0, "repetition_penalty": 1.0}`, for this benchmark only.
- **`--sandbox docker`** reaches it through `configure(sandbox=...)`. Then `preflight()` checks that Docker has the sandbox image, before the model loads, and `fingerprint_extra()` adds the sandbox and the image's ID to the fingerprint. So a run in the sandbox never resumes one made outside it, or with another image. A local run adds nothing, so runs made before the option still resume.

This is the exact prompt, the module's `PROMPT`. It is part of the measurement. `{prompt}` is the stub, and every stub ends with a newline, so the closing fence lands on its own line:

```python
PROMPT = (
    "Complete the following Python function. Return the complete function, including "
    "its signature and any imports it needs, in a single ```python code block.\n\n"
    "```python\n{prompt}```"
)
```

## Why it exists

- **It measures working code, not plausible code.** Hidden unit tests decide the verdict. Wrong code fails however confident it looks, and wording plays no part in the grade.
- **It covers what the other benchmarks don't.** GSM8K tests arithmetic chains, MMLU-Pro tests knowledge and IFEval tests format compliance. HumanEval tests whether a model can turn a specification into a correct program, which is most of what a coding assistant does.
- **One sample per problem matches use.** In chat you get one answer, so pass@1 is the number that matters. Answers are sampled at the profile's temperature with a per-question seed; `-t 0` makes them greedy. Kumaru does not compute Chen et al.'s pass@k for k > 1, which needs many samples per problem.
- **Contamination.** The set has been public since 2021, and newer models may have seen it in training. See the caveats in [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md). Compare Kumaru runs with each other, not with model cards.

## Mental model

```
stub -> PROMPT -> model writes code -> extract_code -> assemble_program -> python -I program.py (15 s) -> exit code 0 and the finish marker?
```

Each answer becomes one `program.py` in a fresh temporary directory, laid out like this:

```
<GUARD>               15 lines that set destructive os / shutil / subprocess functions to None

<stub>                from the dataset: imports, helpers, def + docstring

<the model's code>    a whole function, or a bare body that continues the stub's def

<hidden test>         def check(candidate): assert candidate(...) == ...

check(<entry_point>)

<finish marker>       added by run_program: os.write(1, b"\nkumaru-finished-<16 random hex digits>\n")
```

- **Which code is used.** `extract_code` ([`extract.md`](extract.md)) prefers fenced blocks tagged `python`, `py` or `python3`, or untagged. Among those it takes the longest block that defines the entry point, otherwise the longest block. An unclosed fence yields everything after it. A reply with no fence is used whole, prose included, and that raises a `SyntaxError`. Only the reply's content runs; code in `reasoning` never does.
- **Two reply shapes work** (`assemble_program`):
  - The code defines `def <entry_point>(` (or `async def`). It goes after the stub, so the model's definition replaces the stub's docstring-only one. The stub's imports and helper functions still apply.
  - Otherwise the code is a bare body that continues the stub's open `def`. If its first non-blank line is not indented, the whole body is indented by 4 spaces.
- **Everything in the block runs.** Example `print` calls or `assert`s the model adds at top level run before `check()`. Stdout is captured only to look for the finish marker; what the code prints is otherwise ignored. A failing `assert` of the model's own fails the item.
- **The verdict needs exit code 0 and the finish marker.** The marker, `kumaru-finished-` plus 16 random hex digits, is new for every program, and only the script's last line prints it. So a pass proves the script ran past `check()`.
  - Code that exits with 0 before `check()` runs, via `sys.exit(0)`, `os._exit(0)` or `raise SystemExit`, is `failed`: `exited with code 0 before the tests finished (an early sys.exit?)`. Such code used to be graded `passed`. (`exit(0)` and `quit()` already failed, because the guard sets those builtins to `None`.)
  - The marker is written straight to file descriptor 1 with `os.write`, so code that replaces `sys.stdout` or `print` can't swallow it, and being random it can't be printed by chance. Code that reads its own script file could still copy it: like the guard, the marker stops accidents, not cheating.
  - Re-grading all 200 saved answers from the two Qwen2.5-7B runs with this rule changed no verdict, so no published score moved.
- **The program sees the standard library and nothing from your venv.** The child is the base interpreter (`sys._base_executable`), not the venv's, run with `-I`: isolated mode, which ignores `PYTHON*` variables and user site-packages. In the sandbox it is the image's `python3` (`DOCKER_PYTHON`), with nothing extra installed. The venv's packages are invisible; only packages installed into the base Python itself can be imported. The reference solutions need only the standard library. A reply that imports `numpy` or `sympy` fails, and that is scored as the model's error by design: the task is a self-contained function. Real example: `HumanEval/39` failed in two runs with `ModuleNotFoundError: No module named 'sympy'`.
- **Budget.** Every benchmark gets 8,192 answer tokens by default (`Benchmark.answer_tokens`), plus the thinking budget when thinking is on. HumanEval answers are short (206 tokens per answer in the run below), so the cap binds only when something is wrong: a repetition loop, or thinking that never ends.
  - A cut reply leaves an unclosed fence. The partial code is graded as it stands and usually fails.
  - If thinking uses the whole budget, `output` is empty. The stub alone then returns `None` and fails its tests.
  - `n_truncated`, the **cut off** column of the report, is how you spot a binding cap.
- **Sampling.** Presence and repetition penalties are off for this benchmark, whatever the profile says. For example, `qwen36.yaml` sets `presence_penalty: 1.5` and `qwen25-7b-gguf.yaml` sets `repetition_penalty: 1.05`. Code repeats tokens legitimately (indentation, names, brackets), so penalties push the model toward bugs. Everything else comes from the profile. The overrides are part of the benchmark's fingerprint and appear in its `summary.json` as `"repetition_penalty": 1.0, "presence_penalty": 0.0`.

### What protects you, and what doesn't

| Safeguard (`codeexec.run`, `GUARD`) | Stops | Does not stop |
|---|---|---|
| Opt-in `--allow-code-exec` | Model code running without your decision. | Anything, once you have given it. |
| A separate process: the base interpreter with `-I -X utf8`, stdin from `DEVNULL`, no console window | A crash or hang inside Kumaru. `PYTHON*` variables and user site-packages leaking in. | — |
| 15 s wall-clock limit (`timeout_s`), then everything the program started is killed: a job object on Windows, a process group elsewhere | An endless loop. Processes the program started, by any means, outliving it: they die when the program ends, timed out or not. | On Linux and macOS, a process that starts its own session leaves the group and survives. |
| A fresh, empty working directory, `%TEMP%\kumaru-code-*`, deleted afterwards; `TEMP`, `TMP` and `TMPDIR` point into it | Relative paths and temporary files landing in your repo or your temp folder. | Absolute paths: `open(r"C:\Users\you\notes.txt", "w")` works. |
| Environment reduced to `codeexec.ENV_BASE`: `PATH`, `PATHEXT`, `SYSTEMROOT`, `SYSTEMDRIVE`, `WINDIR`, `COMSPEC`, `HOME`, `USERPROFILE`, `LANG`, `LC_ALL`, `NUMBER_OF_PROCESSORS`, `PROCESSOR_ARCHITECTURE`, `OS` | API keys and tokens held in environment variables (`HF_TOKEN`, `OPENAI_API_KEY`, ...). | Secrets in files: `~\.ssh`, `~\.cache\huggingface\token`, browser profiles, cloud CLI logins. |
| A 4 GB memory cap on Windows (`KUMARU_CODE_MEMORY_MB`), for the program and everything it starts | A runaway allocation paging the machine to a standstill: allocations past the cap fail, and so does the answer. | Linux and macOS have no cap outside the sandbox. |
| Only the last 64 KB of each output stream kept | A program that prints gigabytes filling Kumaru's memory. | — |
| `GUARD` sets these to `None`: up to 29 `os` functions, those that exist on the platform (including `remove`, `unlink`, `rmdir`, `rename`, `replace`, `chmod`, `system`, `popen`, `kill`, `chdir`, `execv`, `spawnv`), `shutil.rmtree`/`move`/`chown`, `subprocess.Popen` and the `exit`/`quit` builtins. It also makes `ipdb`, `joblib`, `resource`, `psutil` and `tkinter` unimportable. | A model's accidental `shutil.rmtree(".")` or `subprocess.run(...)`. The call raises `TypeError: 'NoneType' object is not callable` and the item fails. | Overwriting files with `open(..., "w")`. Network access (`socket`, `urllib`). Deliberate bypasses: `import nt` then `nt.remove(...)` (`posix` outside Windows), `importlib.reload(os)`, `nt.spawnv(...)`, `ctypes`. |
| `--sandbox docker`: each program in a throwaway container with no network, a read-only root, user `nobody`, no capabilities, 2 CPUs, 2 GB of memory and 512 processes | Everything above that the local safeguards don't stop: your files and your network are out of reach, and nothing outlives the container. | A container escape through a kernel or Docker flaw. The image is [not yet built or tested](../../../../docker/sandbox/Dockerfile.md). |

The plain risk, without `--sandbox docker`: each answer runs with your account's rights. It can read any file you can read and send it anywhere. It can overwrite any file you can write. The `nt.remove`, `importlib.reload(os)` and `nt.spawnv` bypasses were checked to work. A process started with `nt.spawnv` used to outlive the timeout; now `codeexec` kills it with the rest of the program. The guard makes harm unlikely by accident, not impossible on purpose.

The known gap the module docstring used to name, a timeout that killed only the program's own process, is closed. What remains is by design: the isolated base interpreter (`-I`) sees only the standard library, which is all HumanEval needs.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `HumanEval` | `HumanEval()` | The benchmark, registered lazily as `humaneval`. `name = "humaneval"`. `description = "Python function synthesis graded by hidden unit tests (pass@1)."` `source = "openai/openai_humaneval (164 problems) - Chen et al. 2021"`. Class attributes: `needs_code_execution = True`, `blocking_grader = True`, `timeout_s = 15.0` (no CLI flag), `sandbox = "none"`. |
| `HumanEval.configure` | `configure(self, *, sandbox: str = "none") -> None` | Called by the runner with `--sandbox`. Raises `KumaruError` (`Unknown sandbox '...'`) for anything but `codeexec.SANDBOXES`. |
| `HumanEval.preflight` | `preflight(self) -> None` | In the sandbox: `codeexec.check_docker()`, which raises `ToolchainMissing` without Docker or the image, and keeps the image's ID. Locally: nothing. |
| `HumanEval.fingerprint_extra` | `fingerprint_extra(self) -> dict[str, Any]` | `{}` locally; `{"sandbox": "docker", "image": <image ID>}` in the sandbox. |
| `HumanEval.load` | `load(self) -> list[BenchItem]` | All 164 problems, in dataset order. Needs `datasets` (the `bench` extra) and, the first time, the internet. |
| `HumanEval.generation_overrides` | `generation_overrides(self) -> dict[str, Any]` | `{"presence_penalty": 0.0, "repetition_penalty": 1.0}`. |
| `HumanEval.grade` | `grade(self, item: BenchItem, output: str) -> Grade` | Assembles the program, runs it and returns the verdict. Blocks for up to `timeout_s`, plus waiting for a free `codeexec` slot, clean-up, and in the sandbox starting a container. |
| `assemble_program` | `assemble_program(prompt: str, completion: str, test: str, entry_point: str) -> str` | The script that will run, minus the finish-marker line `run_program` appends. Pure, so use it to see what was executed. |
| `run_program` | `run_program(program: str, timeout_s: float, *, sandbox: str = "none") -> tuple[str, str]` | Appends the finish marker and runs the script with `codeexec.run`, locally or in the sandbox. Returns `("passed", "")` for exit code 0 with the marker, `("failed", "exited with code 0 before the tests finished (an early sys.exit?)")` for exit code 0 without it, `("failed", <codeexec's note, if any, then the last 800 characters of stderr>)` for any other ending, or `("timeout", "")`. |
| `GUARD` | `str` | The 15-line prelude added to the top of every program. |
| `PROMPT` | `str` with one `{prompt}` field | The user message template. |
| `DOCKER_PYTHON` | `"python3"` | The sandbox image's Python ([`Dockerfile.md`](../../../../docker/sandbox/Dockerfile.md)). |

## How to use it

### Run it contained

Strongest first:

1. **A disposable VM or container that holds no credentials**, such as Windows Sandbox, a Hyper-V VM or a Docker container. To keep the GPU on the host, run the model server on the host, and run `kumaru bench run` inside the box with a profile whose `backend.base_url` points at that server.
   - With `base_url` set, the `llamacpp_server` backend attaches instead of spawning ([`llamacpp_server.md`](../../backends/llamacpp_server.md)). The `openai_compat` backend never spawns.
   - Let the box reach that one port and nothing else.
2. **`--sandbox docker`**: Kumaru and the model stay on the host, and each program runs in its own throwaway container with no network (the last row of the table above). Build the image once with `docker build -t kumaru-sandbox -f docker/sandbox/Dockerfile .`. It is [not yet built or tested](../../../../docker/sandbox/Dockerfile.md): run `python scripts\check_graders.py humaneval --sandbox docker` before trusting it.
3. **A separate low-privilege Windows account** with no SSH keys, tokens or cloud logins, and outbound traffic blocked by the firewall.
4. **Your own account**, only for models from publishers you trust, and only after reading the table above.

### Run it

```powershell
kumaru bench run humaneval -c qwen36 --allow-code-exec                    # 100 seeded problems
kumaru bench run humaneval -c qwen36 --allow-code-exec --full             # all 164
kumaru bench run humaneval -c qwen36 --allow-code-exec --sandbox docker   # each program in a container
kumaru bench run golden gsm8k humaneval -c qwen36 --allow-code-exec       # several benchmarks in one run
kumaru bench report runs\bench\<run>\humaneval
```

Without the flag, the command exits with code 1 and writes nothing (with `--sandbox docker`, the message ends `model-written code.` instead):

```
error: 'humaneval' grades answers by running model-written code on this machine.
hint: Add --allow-code-exec to accept that (docs/BENCHMARKS.md explains the safeguards and their limits), or leave it out of the list.
```

A real report (Qwen2.5-7B, illustrative only; results are in [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md)):

```
│ humaneval │ 100 │       81.0 │ 72.2-87.5 │      0 │       0 │           206 │        55.1 │
humaneval  tests: passed 81, failed 19
```

For HumanEval the line under the table is `tests:`, and it counts test outcomes rather than the extraction methods other benchmarks show under `answers found by:` ([`runner.md`](runner.md#report)). A `timeout` count appears when programs ran out of time, and `error N` when grading itself failed.

**How precise is it?** The 95% interval around 81% at n = 100 is 72.2–87.5. `--full` still leaves about ±6 points near 80%, because only 164 problems exist: 133/164 gives 74.4–86.4. To tell two models apart, use the paired `kumaru bench compare` ([`stats.md`](stats.md), [`runner.md`](runner.md#compare-two-runs)). For illustration, two runs of the same model and profile scored 80.0 and 81.0: B − A = +1.0 points, 95% CI −3.0 to +6.0, "no clear difference". Even so, 5 of the 100 problems flipped (`-e 1`, trimmed):

```
humaneval: only B right (3)
  HumanEval/11  Complete the following Python function. Return the complete function, including its signature and a…
    expected -  ·  A: failed  ·  B: passed

humaneval: only A right (2)
  HumanEval/116  Complete the following Python function. Return the complete function, including its signature and a…
    expected -  ·  A: passed  ·  B: failed
```

### Read why code failed

```powershell
Get-Content runs\bench\<run>\humaneval\items.jsonl | ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { -not $_.correct } |
  Select-Object id, extracted, finish_reason, @{n='last error line'; e={ ($_.detail.stderr -split "`n" | Where-Object { $_ })[-1] }}
```

```
id           extracted finish_reason last error line
--           --------- ------------- ---------------
HumanEval/39 failed    stop          ModuleNotFoundError: No module named 'sympy'
HumanEval/65 failed    stop          AssertionError
HumanEval/77 failed    stop          TypeError: type complex doesn't define __round__ method
```

How to read the last error line:

| Last error line | Meaning |
|---|---|
| `AssertionError` | A hidden test failed. The stderr tail shows which assert. |
| `ModuleNotFoundError` | The code imported something outside the standard library. |
| `SyntaxError` or `IndentationError` | No code fence, or a cut-off reply (check `finish_reason`). |
| `TypeError: 'NoneType' object is not callable` | The code called something the guard disabled. |
| `exited with code 0 before the tests finished (an early sys.exit?)` | The code ended the program before `check()` ran, e.g. a top-level `sys.exit(0)`. |
| `MemoryError` | The program hit the 4 GB memory cap, usually a loop that keeps allocating. In the sandbox the container is killed instead: `killed: out of memory?`. |
| `timeout` status (no stderr) | An endless loop, or a very slow solution. |

Read the failing assert before you blame the model. `HumanEval/116`'s docstring example says `sort_array([-2, -3, -4, -5, -6]) == [-6, -5, -4, -3, -2]`, but its hidden test expects `[-4, -2, -6, -5, -3]`. In the comparison above, run B's reply followed the docstring and failed; run A's passed.

To see the exact script that ran, rebuild it. With both variables below set, `load()` reads only the local dataset cache:

```python
# $env:HF_HUB_OFFLINE='1'; $env:HF_DATASETS_OFFLINE='1'  (in the shell first)
from pathlib import Path
from kumaru.eval.benchmarks.humaneval import HumanEval, assemble_program, run_program
from kumaru.eval.benchmarks.runner import load_results

summary, records = load_results(Path(r"runs\bench\<run>\humaneval"))
item = {i.id: i for i in HumanEval().load()}["HumanEval/116"]
program = assemble_program(item.meta["prompt"], records[item.id].output, item.meta["test"], item.meta["entry_point"])
print(program)                       # read it before you run it (run_program adds the finish-marker line)
run_program(program, timeout_s=15)   # ('failed', '...AssertionError\n'): the recorded verdict
```

## How to extend it

- **What a resumed run notices** (`-o <run dir>`):
  - Change `PROMPT` and every question is re-asked, because each record's `key` covers the messages sent.
  - Change `generation_overrides`, the sandbox or the sandbox image and the resume is refused, because they are in the fingerprint.
  - Change `GUARD`, `timeout_s` or the pass rule in `run_program`, and the old verdicts are silently kept: the fingerprint covers settings, not code, and none of these is in the fingerprint or the key. After changing any of them, start a fresh run.
- **A different time limit:** subclass `HumanEval`, set `timeout_s` and give the subclass a new `name` (it's a different measurement). Then register it ([`__init__.md`](__init__.md#add-a-benchmark-end-to-end)). There is no CLI flag.
- **Another kind of isolation:** add it to `codeexec` ([`codeexec.md`](codeexec.md)) as a new sandbox, next to `none` and `docker`, so every code benchmark gets it. Keep the finish-marker check, or an early `sys.exit(0)` passes again.
- **Another dataset in the same format** (fields `task_id`, `prompt`, `test` defining `check(candidate)`, `entry_point`): override `name`, `description`, `source` and `load`, and reuse everything else.
- **Any new benchmark that runs code** must set `needs_code_execution = True` so the consent gate applies. Set `blocking_grader = True` if grading waits on a process.

Pinned behaviour:
- `tests\test_bench_graders.py::TestHumanEval`: both reply shapes, the indentation fix, pass, fail with the traceback tail, timeout, a clean exit before the tests (`test_a_clean_exit_before_the_tests_is_not_a_pass`, for `sys.exit(0)`, `os._exit(0)` and `raise SystemExit`), the image's Python and the out-of-memory note in the sandbox (Docker faked), the guard blocking `os.remove`, environment secrets withheld, and the `Grade` fields.
- `tests\test_bench_codeexec.py`: what `codeexec` guarantees for every code benchmark ([`codeexec.md`](codeexec.md)).
- `tests\test_bench_runner.py::test_code_execution_benchmarks_need_explicit_consent`.
- `tests\test_cli.py::test_errors_become_a_message_a_hint_and_exit_code_1`.

`scripts\check_graders.py humaneval` also runs every reference solution through `HumanEval.grade` in three shapes (whole function, body only, unindented body), and checks a wrong solution, a hang and the guard; add `--sandbox docker` to check the sandbox. The "Are the graders right?" section of [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md) reports the result.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `'humaneval' grades answers by running model-written code on this machine.` | `--allow-code-exec` is missing. | Read this page, then add the flag, or drop `humaneval` from the list. |
| `Optional dependency 'datasets' is not installed.` | The `bench` extra is missing. | `pip install -e ".[bench]"` |
| `Could not load dataset 'openai/openai_humaneval' (split 'test'): ...` | First run without internet. | Connect once. Later runs read the Hugging Face cache. |
| `ModuleNotFoundError` for `numpy`, `sympy`, ... | The child is the base interpreter in isolated mode, so venv packages are invisible. | Expected: the task asks for a self-contained function. Don't install packages into the base Python to "fix" it; that changes the benchmark. |
| `SyntaxError` at the first line of the reply | No code fence, so the whole reply, prose included, was run. | A format failure. Read `output`, and check the chat template. |
| `timeout` on code that looks right | An endless loop, or a slow solution under load. The 15 s is wall-clock, and programs run next to the model and each other. | Read `output`. If the code is slow rather than stuck, lower `KUMARU_CODE_WORKERS` or `-j`, or subclass with a larger `timeout_s`. |
| `kumaru-code-*` folders left in `%TEMP%` | Cleanup failed because a file was still locked (a virus scanner, say). Cleanup errors are ignored by design. | Delete the folders once no grader is running. |
| `--sandbox docker needs Docker, which is not installed.` or `Docker has no image 'kumaru-sandbox', or the Docker daemon is not running` | No Docker, Docker Desktop not started, or the image not built. Checked before the model loads. | Start Docker, then `docker build -t kumaru-sandbox -f docker/sandbox/Dockerfile .` ([`Dockerfile.md`](../../../../docker/sandbox/Dockerfile.md)). |
| `failed` with `killed: out of memory?` (sandbox) | The container's 2 GB limit killed the program. | Usually a runaway loop. Read `output`. |
| `cut off` > 0, or an empty `output` | The 8,192-token budget was hit, usually by a loop or by thinking that never ended. Cut code is graded as it stands and usually fails. | Read `output` and `reasoning`. Raise `--max-tokens` or `--thinking-budget` only if the work was real. |
| `failed` with `exited with code 0 before the tests finished (an early sys.exit?)` | The code exited with 0 before `check()` ran (`sys.exit(0)`, `os._exit(0)`, `raise SystemExit`), so the finish marker never printed. | Expected: the hidden tests never ran, so it can't pass. Read `output`. |
| `errors` > 0 with `grader failed: ...` | Grading itself raised, e.g. the temporary directory could not be created. Five failed questions in a row stop the benchmark: `Stopped humaneval: 5 questions failed in a row. Last error: grader failed: ...`, and the hint calls an error starting `grader failed` the benchmark's own bug. | Fix the cause, then resume with `-o <run dir>`; errored questions are asked again. |
| Far from the model card's score | A different protocol: prompt, sampling (penalties off here), extraction, available imports. | Compare Kumaru runs with each other. |

## Related files

- `src\kumaru\eval\benchmarks\codeexec.py`: runs each program, locally or in the sandbox ([`codeexec.md`](codeexec.md)).
- `src\kumaru\eval\benchmarks\multipl_e.py`: the same problems in eight languages, graded the same way ([`multipl_e.md`](multipl_e.md)).
- `docker\sandbox\Dockerfile`: the sandbox image ([`Dockerfile.md`](../../../../docker/sandbox/Dockerfile.md)).
- `src\kumaru\eval\benchmarks\extract.py`: `extract_code` ([`extract.md`](extract.md)).
- `src\kumaru\eval\benchmarks\base.py`: the `Benchmark` flags, `load_hf_rows`, `select_items` and the 8,192 default ([`base.md`](base.md)).
- `src\kumaru\eval\benchmarks\runner.py`: the consent check, threaded grading, budgets, `report` and `compare` ([`runner.md`](runner.md)).
- `src\kumaru\core\registry.py`: `BENCHMARKS.register_lazy("humaneval", ...)`.
- `scripts\check_graders.py`: runs the reference solutions through this grader ([`check_graders.py.md`](../../../../scripts/check_graders.py.md)).
- `tests\test_bench_graders.py`, `tests\test_bench_runner.py`, `tests\test_cli.py`: pinned behaviour.
- [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md): results, grader validation and caveats.
