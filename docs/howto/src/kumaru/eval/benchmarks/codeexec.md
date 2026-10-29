# `src\kumaru\eval\benchmarks\codeexec.py`

> Runs one model-written program, as a list of steps such as compile then run, in a fresh temporary folder with a time limit per step, a memory cap and a scrubbed environment. It kills everything the program started and reports how it ended. Every code benchmark (HumanEval, MultiPL-E) grades through it.

**Read this when:** you are about to grade with `--allow-code-exec`, a code benchmark reports `timeout`, `compile_error` or a toolchain error, you want `--sandbox docker`, or you are adding a benchmark that runs code.

---

> **Without `--sandbox docker`, this is not a sandbox.** Model-written programs run as you, can read your files and can reach the network. The protections below stop a model's accidents, not code written to do harm. `--sandbox docker` runs each program in a throwaway container with no network ([Run it in Docker](#run-it-in-docker)).

## What it does

- **`run(program, *, sandbox="none")`** runs a `Program` (its files plus its `Step`s) and returns an `Outcome`. It waits for a free slot first: at most `max_workers()` programs run at once in the whole process, however many answers the runner grades in parallel.
- **Locally** (`sandbox="none"`, `_run_local`):
  1. It makes a folder `kumaru-code-*` in the system temp folder and writes the program's files into it as UTF-8 bytes, with no newline translation.
  2. It builds the environment (`child_env`): only the `ENV_BASE` variables and the program's `keep_env` pass through from yours, then `TEMP`, `TMP` and `TMPDIR` point at the program's folder, then the program's `env` is applied. API keys and tokens never reach the program.
  3. It runs each step in order, in that folder, with stdin closed and no console window, for at most the step's `timeout_s`. When the step ends or times out, **everything it started is killed**, not only its own process (below).
  4. It deletes the folder (`_remove`, three tries, clearing read-only bits in between).
- **The verdict:**

  | How it ended | `status` | `stage` | `returncode` |
  |---|---|---|---|
  | Every step exited with 0 and the last printed the marker | `passed` | the last step's | 0 |
  | Every step exited with 0, but no marker | `failed`, with `note` = `EARLY_EXIT_NOTE` | the last step's | 0 |
  | A `compile` step exited non-zero | `compile_error` | `compile` | its exit code |
  | A `run` step exited non-zero | `failed` | `run` | its exit code |
  | A step ran out of time | `timeout` | that step's | `None` |
  | A `setup` step failed or timed out | raises `KumaruError` (`Sandbox setup step ... failed`): the sandbox is broken, not the answer | | |

  The first step that doesn't exit with 0 ends the program; later steps don't run. `stdout` and `stderr` are the last 64 KB (`TAIL_BYTES`) of the step the program ended in.
- **In Docker** (`sandbox="docker"`, `_run_docker`), each program runs in a new container of the sandbox image ([`Dockerfile.md`](../../../../docker/sandbox/Dockerfile.md)). Its files arrive on stdin as a tar archive; a shell script (`docker_script`) runs each step under `timeout`, then prints a record per step, which `parse_docker_output` turns into the same `Outcome`s as above.

## Why it exists

- **One set of protections for every code benchmark.** HumanEval used to run its programs itself, and a timeout killed only the program's own process: a process it had started kept running. Compiled languages make that worse, because compilers start helpers (`dotnet build` starts MSBuild nodes, `go test` starts the compiler and linker). Putting every step of every benchmark through one function means one place to get killing, cleanup, limits and the pass rule right.
- **A pass must prove the tests ran.** A program that exits with 0 before its tests (`sys.exit(0)`, `process.exit(0)`, `std::exit(0)`, `os.Exit(0)`) would otherwise pass. Each program gets a random marker (`new_marker`) to print after its last test, and only exit code 0 *with* the marker is `passed`.
- **A compile error is not a wrong answer at run time.** Separating `compile_error` from `failed` shows whether a model can't write the language's syntax and types, or writes code that compiles and computes the wrong thing.
- **Docker is optional, so it can't be the only defence.** On a machine without Docker, local runs still get the limits above.

## Mental model

```
Program{files, steps, marker}
   -> wait for a slot (max_workers)
   -> kumaru-code-XXXX\  (also TEMP/TMP/TMPDIR)
   -> step 1 (compile, 120 s) -> kill its tree -> exit 0? -> step 2 (run, 15 s) -> kill its tree
   -> exit 0 and the marker in stdout? -> Outcome
   -> delete the folder
```

### How a step's processes die

| Platform | How the step runs | What is killed, and when |
|---|---|---|
| Windows | The process is created suspended, put in a new **job object**, then resumed (`NtResumeProcess`), so it can't start a child before it is contained. Every process it starts joins the job and can't leave it. | `TerminateJobObject` kills the whole job when the step ends or times out, then waits up to 5 s until no process is left. The job also has `KILL_ON_JOB_CLOSE`, so the job dies with Kumaru. |
| Windows, if a job can't be created or joined | Nothing runs: `run` raises `KumaruError` (see below). | The process, if it was created, is killed while still suspended. |
| Linux, macOS | The process leads a new session (`start_new_session`), so it and its children share a process group. | `killpg(SIGKILL)` on the group. A child that starts its own session escapes it; Docker doesn't have that gap. |

Without a job nothing would cap the program's memory, and a process it started could outlive it while holding its output pipes open, keeping the step's readers waiting past the time limit. So a program Kumaru can't contain is not run. The error is not a wrong answer: the runner records it against the question ("grader failed: KumaruError: Could not create the Windows job object …"), a resumed run asks the question again, and five such errors in a row stop the run. If it keeps happening (the machine is out of memory or handles), grade with `--sandbox docker`.

The job object also carries three limits:

- **Memory:** the job may commit at most `KUMARU_CODE_MEMORY_MB` (default 4096) across all its processes. Past that, allocations fail, so a runaway program crashes (a failure) instead of paging the machine to a standstill. `0` turns the cap off. The cap covers compile steps too, so don't set it far below the default.
- **Processes:** at most `MAX_PROCESSES` (512, as Docker's `--pids-limit`) may run at once. Past that, starting another fails with `ERROR_NOT_ENOUGH_QUOTA` (1816), so a fork bomb stops there instead of filling the machine.
- **No crash dialogs** (`DIE_ON_UNHANDLED_EXCEPTION`): a crashing program exits at once instead of opening Windows Error Reporting, which would hold it until the timeout.

### What protects you, and what doesn't

| Protection | Local (`none`) | Docker (`docker`) |
|---|---|---|
| Opt-in `--allow-code-exec` | yes | yes |
| Time limit per step, everything started killed | yes (job object or process group) | yes (`timeout -k 2` per step, the container removed at the end) |
| Memory cap | Windows: 4 GB per step (`KUMARU_CODE_MEMORY_MB`); Linux and macOS: none | 2 GB per container, no swap |
| CPU and process limits | Windows: 512 processes at once per step; no CPU limit (the runner's `-j` and `max_workers` bound how many programs run) | 2 CPUs, 512 processes |
| Environment variables withheld (API keys, tokens) | yes | yes: the container gets only the variables the language needs |
| Your files | readable and writable with your rights; relative paths land in the temp folder | invisible: the container sees only its image and its own files; the root is read-only, and only `/work` and `/tmp` (in memory) are writable |
| Network | open | none (`--network none`) |
| User | you | `nobody` (65534), all capabilities dropped, `no-new-privileges` |

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `run` | `run(program: Program, *, sandbox: str = "none") -> Outcome` | Run one program. Raises `ValueError` for an unknown sandbox or a program without steps, `ToolchainMissing` when a step's executable can't start, and `KumaruError` when a setup step fails, Docker misbehaves, or (on Windows) a step's job object can't be created or joined. |
| `Program` | `Program(files: dict[str, str], steps: tuple[Step, ...], marker: str, env: dict[str, str] = {}, keep_env: frozenset[str] = frozenset())` | Relative path → text; the steps; the marker the last step must print; variables set for every step; extra variables passed through from your environment. |
| `Step` | `Step(argv: tuple[str, ...], timeout_s: float, stage: Stage = "run")` | One command, run in the program's folder. A relative `argv[0]` that names a file in the folder (`./program.exe`, which an earlier step built) is run from there. `stage` is `setup`, `compile` or `run`. |
| `Outcome` | `Outcome(status, stage, returncode, stdout, stderr, elapsed_s, note="")`, `.passed` | How the program ended (table above). `returncode` is `None` after a timeout. Docker keeps 4 KB of each stream, not 64 KB. |
| `Status`, `Stage` | `Literal["passed", "failed", "timeout", "compile_error"]`, `Literal["setup", "compile", "run"]` | |
| `ToolchainMissing` | `KumaruError` subclass, `http_status = 400` | A compiler, runtime or Docker image isn't available. The runner turns it into a skipped language inside a suite ([`runner.md`](runner.md)). Its `why` property gives the message and then the hint as two sentences, for lists of what did not run; `str(exc)` would wrap the hint in parentheses, and hints such as Java's contain their own. |
| `new_marker` | `new_marker() -> str` | `kumaru-finished-` plus 16 random hex digits. |
| `find_program` | `find_program(name: str) -> str \| None` | The absolute path of `name` in a `PATH` folder, or `None`. Every toolchain and `docker` is found with it (see [Finding programs](#finding-programs)). |
| `max_workers`, `set_max_workers` | `max_workers() -> int`, `set_max_workers(n: int) -> None` | Programs at once: `KUMARU_CODE_WORKERS`, else half the CPU cores, between 1 and 4. `set_max_workers` resizes the pool; call it before grading starts. |
| `child_env` | `child_env(program: Program, workdir: Path) -> dict[str, str]` | The environment a local step sees. |
| `memory_limit_mb` | `memory_limit_mb() -> int` | `KUMARU_CODE_MEMORY_MB`, else 4096. |
| `fingerprint_limits` | `fingerprint_limits(sandbox: str) -> dict[str, int]` | What a run's fingerprint records of these limits: `{"memory_mb": n}` for a local run on Windows whose cap isn't the default (`0` = no cap), else `{}`. A cap can change a verdict, so a run at another cap is another run; runs at the default keep the fingerprints they were made with. Docker's cap is fixed (2 GB), and only Windows reads the variable. |
| `docker_image` | `docker_image() -> str` | `KUMARU_SANDBOX_IMAGE`, else `kumaru-sandbox`. |
| `check_docker` | `check_docker(image: str \| None = None) -> str` | The image's ID. Raises `ToolchainMissing`, with a hint that gives the `docker build` command, when Docker isn't installed (not in a `PATH` folder), isn't running or lacks the image. When the lookup by name fails, it first starts a container from the image (`/bin/true`, no network) and looks again: Docker Desktop's Resource Saver, which pauses the engine after 5 minutes without a container, answers lookups by name with `No such image` until a container starts. |
| `docker_command` | `docker_command(docker: str, image: str, name: str, env: dict[str, str], script: str) -> list[str]` | The `docker run` argv, starting with `docker`, the docker program's absolute path from `find_program` (limits listed in [Run it in Docker](#run-it-in-docker)). |
| `docker_script` | `docker_script(program: Program, token: str) -> str` | The `/bin/sh` script a container runs. |
| `parse_docker_output` | `parse_docker_output(stdout: str, token: str, program: Program) -> Outcome` | The script's records back into an `Outcome`. A step that exits with 124 or 137 after at least 95% of its limit timed out. |
| Constants | `SANDBOXES = ("none", "docker")`, `TAIL_BYTES = 65536`, `DEFAULT_MEMORY_MB = 4096`, `MAX_PROCESSES = 512`, `DEFAULT_IMAGE = "kumaru-sandbox"`, `DOCKERFILE = "docker/sandbox/Dockerfile"`, `EARLY_EXIT_NOTE`, `ENV_BASE` | `ENV_BASE` is `PATH`, `PATHEXT`, `SYSTEMROOT`, `SYSTEMDRIVE`, `WINDIR`, `COMSPEC`, `HOME`, `USERPROFILE`, `LANG`, `LC_ALL`, `NUMBER_OF_PROCESSORS`, `PROCESSOR_ARCHITECTURE` and `OS`. |

### Finding programs

`find_program(name)` is `shutil.which` without the current folder. On Windows, `which` looks in the current folder before `PATH` (unless `NoDefaultCurrentDirectoryInExePath` is set), so a `node.cmd` left in the folder Kumaru was started from would run, with your whole environment, as soon as Kumaru asked Node.js for its version. `find_program`:

- walks only the absolute folders in `PATH`, in order. It skips empty and relative entries (`.`, `bin`), which also mean the current folder; on Windows it also skips `\bin` and `C:bin`, which depend on the current drive or on drive C:'s current folder. Quotes around an entry are removed on Windows, as `cmd` does;
- on Windows, tries `name` plus each `PATHEXT` extension (default `.COM;.EXE;.BAT;.CMD`), or `name` alone if it already ends with one; never an extensionless file, which Windows can't run;
- on Linux and macOS, wants a file you may execute;
- returns that absolute path. The toolchains and Docker run by it, never by a bare name.

## How to use it

### Run a program yourself

```python
from kumaru.eval.benchmarks import codeexec

marker = codeexec.new_marker()
program = codeexec.Program(
    {"program.py": f"assert 1 + 1 == 2\nprint({marker!r})\n"},
    (codeexec.Step(("python", "program.py"), timeout_s=15),),
    marker,
)
print(codeexec.run(program))   # Outcome(status='passed', stage='run', returncode=0, ...)
```

A compiled language has a `compile` step first; see `build_program` in [`multipl_e.md`](multipl_e.md) for the steps each language uses.

### Tune the limits

| Variable | Default | Effect |
|---|---|---|
| `KUMARU_CODE_WORKERS` | half the CPU cores, 1 to 4 | Programs that run at once. Lower it when the CPU is shared (a model on the CPU, a game): a compiler under load can run out of time. |
| `KUMARU_CODE_MEMORY_MB` | 4096 | Windows only: memory per step, all its processes together. `0` = no cap. Any other value than 4096 is part of the run's fingerprint, so a run resumes only at the cap it was started with. |
| `KUMARU_SANDBOX_IMAGE` | `kumaru-sandbox` | The image `--sandbox docker` uses. |

The time limits belong to each benchmark (HumanEval: 15 s; MultiPL-E: 120 s to compile, 15 s to run), not to this module.

### Run it in Docker

Build the image once ([`Dockerfile.md`](../../../../docker/sandbox/Dockerfile.md)), then add `--sandbox docker`:

```powershell
docker build -t kumaru-sandbox -f docker/sandbox/Dockerfile .
kumaru bench run humaneval multipl_e -c qwen36 --allow-code-exec --sandbox docker
```

Each program then runs as (`docker` being the absolute path `find_program` found):

```
docker run --rm -i --pull never --name kumaru-<random> --network none --read-only
  --tmpfs /work:rw,exec,nosuid,nodev,size=512m,mode=1777 --tmpfs /tmp:rw,exec,nosuid,nodev,size=1g,mode=1777
  --memory 2g --memory-swap 2g --cpus 2 --pids-limit 512 --ulimit core=1 --cap-drop ALL --security-opt no-new-privileges
  --user 65534:65534 --workdir /work --env HOME=/tmp/home --env TMPDIR=/tmp --env DOTNET_CLI_HOME=/tmp/home
  [--env <the language's variables>] --entrypoint /bin/sh kumaru-sandbox -c <script>
```

The script unpacks the files from stdin, runs each step as `timeout -k 2 <limit> <command>`, and after each prints a record fenced by a random token: the step number, its exit code, its milliseconds, whether the marker appeared, and the last 4 KB of each stream. Kumaru allows the container the sum of every step's limit plus 2 s each, plus 30 s to start; past that it runs `docker rm -f` and records a `timeout`.

`--ulimit core=1` turns off core dumps. Without it, a program that crashes (a segfault or an abort) leaves one, and Docker Desktop's WSL VM sends it to Windows, into `%TEMP%\wsl-crashes`: 1.1 GB for one Node.js that aborted. Linux skips that kind of dump only when the limit is exactly 1, so `core=0` doesn't stop it.

> The image was built and checked on 2026-10-23 ([`Dockerfile.md`](../../../../docker/sandbox/Dockerfile.md)): graded in it, HumanEval's and MultiPL-E's verdicts are reproduced as closely as on the host. After rebuilding it, run `python scripts\check_graders.py humaneval multipl_e --sandbox docker` again before trusting results from it.

## How to extend it

- **A benchmark that runs code** builds a `Program` and calls `run`. Then:
  - set `needs_code_execution = True` and `blocking_grader = True` on the benchmark;
  - put the marker last in the program, after the tests, written straight to the output stream (not through a buffer the program could replace), and flush it;
  - accept `sandbox` in `configure` and pass it to `run`; inside Docker, executables are the image's (plain names on its `PATH`), not paths on this machine.
- **Another isolation** (a VM, a remote runner) would be a third value in `SANDBOXES`, in the `Literal` of `BenchSettings.sandbox` (`runner.py`) and in the `cast` of `--sandbox` in `kumaru bench run` (`cli.py`), plus a `_run_<name>` that returns the same `Outcome`s. Keep the pass rule (exit 0 and the marker) and the per-stage statuses.
- **A setup step** (`stage="setup"`) prepares the folder, e.g. the Go cache copy in Docker. Its failure raises instead of failing the answer, because the answer never ran.

Pinned behaviour: `tests\test_bench_codeexec.py` (the pass rule, stages, a timeout that kills a grandchild process, secrets withheld, the folder removed, the output tail, the memory and process caps, a program not started when its job can't be made or joined, the memory cap in the fingerprint, the worker limit, `find_program` passing over decoys in the current folder and in relative `PATH` entries, and the Docker command, script, record parsing and failure handling, and waking a paused Docker Desktop, with Docker faked).

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `timeout` on code that looks right | An endless loop, a slow solution, or a shared CPU: a model or a game next to the graders, or more programs at once than the cores can take. | Read the answer. If it is slow rather than stuck, lower `KUMARU_CODE_WORKERS` (or `-j`). A `timeout` at the `compile` stage (limit 120 s) means heavy contention. |
| `failed` with `exited with code 0 before the tests finished (an early exit?)` | The program exited with 0 before its last test, so the marker never printed. | Expected: the tests didn't run. Read the answer. |
| `failed` with an out-of-memory error (`MemoryError`, `std::bad_alloc`, `OutOfMemoryError`, `memory allocation of N bytes failed`) | The program hit the memory cap: usually a runaway loop that keeps allocating. | Expected for such loops. If a compile step hits it, raise `KUMARU_CODE_MEMORY_MB`. |
| `failed` with `[WinError 1816]` or `Not enough quota` | The program had 512 processes running at once (`MAX_PROCESSES`): a fork bomb. | Expected. |
| `grader failed: KumaruError: Could not create the Windows job object …` (or `Could not start … inside its Windows job object`) | Windows refused a job object, or refused to put the program in it or resume it there, usually for lack of memory or handles. The program did not run. | Free memory and resume the run (`--out`): the question is asked again. If it persists, grade with `--sandbox docker`. |
| A toolchain is "not on PATH", but `where node` finds it | `where` and `shutil.which` also search the current folder and relative `PATH` entries; Kumaru doesn't. | Put the toolchain's folder in `PATH` as an absolute path. |
| `Could not start <path>: [WinError 2] ...` | A step's executable doesn't exist (removed, or a wrong path in a toolchain). | Reinstall the toolchain, or check `PATH`. |
| `Sandbox setup step 'cp -R /opt/gocache /tmp/gocache' failed ...` | The image lacks what a setup step copies. | Rebuild the image from `docker/sandbox/Dockerfile`. |
| `--sandbox docker needs Docker, which is not installed.` / `Docker has no image 'kumaru-sandbox', or the Docker daemon is not running` | No Docker, Docker stopped, or the image isn't built. | Follow the hint: start Docker, then `docker build -t kumaru-sandbox -f docker/sandbox/Dockerfile .` |
| `The sandbox container failed (docker run exited with 125)` | Docker refused to start the container (bad option, no space, daemon error). | Read the message; check `docker info`. |
| `The sandbox container printed no results` / `stopped after step N of M` | The script inside the container broke: the image lacks `/bin/sh`, `tar` or `timeout`, or a step killed the shell. | Rebuild the image; run `python scripts\check_graders.py humaneval --sandbox docker` to check it. |
| `failed` (or `compile_error`) with `killed: out of memory?` | A container hit its 2 GB limit, which killed the whole container. | Expected for runaway programs. |
| `timeout` with `the container did not finish in time` | The container outlived every step's limit plus 30 s (a slow Docker start, or Docker hung). Kumaru removed it with `docker rm -f`. | Check `docker ps`; restart Docker if it is stuck. |
| `kumaru-code-*` folders left in `%TEMP%` | A file was still locked when cleanup ran (a virus scanner, a process outside the job). | Delete them once no grader is running. |

## Related files

- `src\kumaru\eval\benchmarks\humaneval.py`: runs its Python programs through `run` ([`humaneval.md`](humaneval.md)).
- `src\kumaru\eval\benchmarks\multipl_e.py`: builds each language's `Program` ([`multipl_e.md`](multipl_e.md)).
- `docker\sandbox\Dockerfile`: the sandbox image ([`Dockerfile.md`](../../../../docker/sandbox/Dockerfile.md)).
- `src\kumaru\eval\benchmarks\runner.py`: the consent gate, `--sandbox`, and skipping a language whose toolchain is missing ([`runner.md`](runner.md)).
- `tests\test_bench_codeexec.py`: pinned behaviour.
