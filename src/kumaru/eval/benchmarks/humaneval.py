"""HumanEval — write a correct Python function from its docstring.

164 hand-written programming problems. Each gives a signature and docstring;
the model writes the function; hidden unit tests decide pass or fail. The
headline metric is pass@1: the share of problems solved on the first attempt.

Running the tests means running code a model wrote, on your machine. So it is
opt-in (`--allow-code-exec`), and each program runs through `codeexec`: a
separate Python process in an empty temporary directory, with a time limit,
a memory cap, a scrubbed environment, and everything it starts killed when it
ends. A guard also disables the destructive parts of `os`, `shutil` and
`subprocess` (the same idea as the reference harness's `reliability_guard`).
That stops *accidents* — a model's `shutil.rmtree` — not a determined
adversary. It is not a sandbox; use `--sandbox docker` (a throwaway container
with no network) or a VM to evaluate models you do not trust. The isolated
interpreter (`-I`, the base Python rather than the venv) sees only the
standard library, which is all HumanEval needs.

A program passes only if it exits cleanly *and* prints a random marker placed
after the tests: model code that calls `sys.exit(0)` before them would
otherwise exit cleanly and pass untested.

Source: Chen et al., 2021, "Evaluating Large Language Models Trained on Code";
dataset `openai/openai_humaneval`, split `test`.
"""

from __future__ import annotations

import sys
import textwrap
from typing import Any

from kumaru.core.errors import KumaruError
from kumaru.eval.benchmarks import codeexec
from kumaru.eval.benchmarks.base import BenchItem, Benchmark, Grade, load_hf_rows
from kumaru.eval.benchmarks.extract import defines_python, extract_code

PROMPT = (
    "Complete the following Python function. Return the complete function, including "
    "its signature and any imports it needs, in a single ```python code block.\n\n"
    "```python\n{prompt}```"
)

# Prepended to every program. Blocks accidents, not adversaries (see module doc).
GUARD = """\
import builtins as _kumaru_builtins, os as _kumaru_os, shutil as _kumaru_shutil
import subprocess as _kumaru_subprocess, sys as _kumaru_sys
for _kumaru_name in (
    "kill", "killpg", "system", "putenv", "remove", "removedirs", "rmdir", "unlink",
    "rename", "renames", "replace", "truncate", "chmod", "fchmod", "chown", "fchown",
    "lchown", "chroot", "fork", "forkpty", "setuid", "chdir", "fchdir", "startfile",
    "popen", "execv", "execve", "spawnl", "spawnv",
):
    if hasattr(_kumaru_os, _kumaru_name):
        setattr(_kumaru_os, _kumaru_name, None)
_kumaru_shutil.rmtree = _kumaru_shutil.move = _kumaru_shutil.chown = None
_kumaru_subprocess.Popen = None
_kumaru_builtins.exit = _kumaru_builtins.quit = None
for _kumaru_module in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
    _kumaru_sys.modules[_kumaru_module] = None
"""

# What the sandbox image calls its Python (docker/sandbox/Dockerfile).
DOCKER_PYTHON = "python3"


def assemble_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    """Join the problem, the model's code and the hidden tests into one script."""
    code = extract_code(completion, entry_point)
    if defines_python(code, entry_point):
        # The prompt still contributes imports and helper functions; its stub
        # (a docstring-only body, which is valid Python) is then overridden.
        body = prompt + "\n\n" + code
    else:
        # A bare function body continues the prompt's open function. Models
        # sometimes drop the body's indentation; shift it back under the def.
        first = next((line for line in code.splitlines() if line.strip()), "")
        if first and not first[:1].isspace():
            code = textwrap.indent(code, "    ")
        body = prompt + code
    return f"{GUARD}\n{body}\n\n{test}\n\ncheck({entry_point})\n"


def run_program(program: str, timeout_s: float, *, sandbox: str = "none") -> tuple[str, str]:
    """Run a script in a fresh interpreter. Returns `(status, stderr_tail)`.

    "passed" needs exit code 0 *and* a fresh random marker that only the end of
    the script prints, written straight to file descriptor 1 so a replaced
    `sys.stdout` or `print` cannot swallow it.
    """
    marker = codeexec.new_marker()
    finish = f"\n\nimport os as _kumaru_finish\n_kumaru_finish.write(1, b'\\n{marker}\\n')\n"
    # The base interpreter, not the venv launcher: on Windows the launcher is a
    # parent process, and the real interpreter would be its child.
    python = getattr(sys, "_base_executable", None) or sys.executable
    if sandbox == "docker":
        python = DOCKER_PYTHON
    step = codeexec.Step((python, "-I", "-X", "utf8", "program.py"), timeout_s)
    outcome = codeexec.run(
        codeexec.Program({"program.py": program + finish}, (step,), marker), sandbox=sandbox
    )
    if outcome.passed:
        return "passed", ""
    if outcome.status == "timeout":
        return "timeout", ""
    if outcome.note == codeexec.EARLY_EXIT_NOTE:
        return "failed", "exited with code 0 before the tests finished (an early sys.exit?)"
    # Any other note says what stopped it, e.g. the sandbox's memory limit.
    return "failed", "\n".join(part for part in (outcome.note, outcome.stderr[-800:]) if part)


class HumanEval(Benchmark):
    name = "humaneval"
    description = "Python function synthesis graded by hidden unit tests (pass@1)."
    source = "openai/openai_humaneval (164 problems) - Chen et al. 2021"
    needs_code_execution = True
    blocking_grader = True
    timeout_s = 15.0
    sandbox = "none"
    _image = ""

    def configure(self, *, sandbox: str = "none") -> None:
        if sandbox not in codeexec.SANDBOXES:
            raise KumaruError(
                f"Unknown sandbox '{sandbox}'.", hint=f"Use one of: {', '.join(codeexec.SANDBOXES)}."
            )
        self.sandbox = sandbox

    def preflight(self) -> None:
        if self.sandbox == "docker":
            self._image = codeexec.check_docker()

    def fingerprint_extra(self) -> dict[str, Any]:
        # Nothing for local runs, so runs made before the sandbox option still resume.
        if self.sandbox == "none":
            return {}
        return {"sandbox": self.sandbox, "image": self._image}

    def load(self) -> list[BenchItem]:
        return [
            BenchItem(
                id=row["task_id"],
                prompt=PROMPT.format(prompt=row["prompt"]),
                meta={"prompt": row["prompt"], "test": row["test"], "entry_point": row["entry_point"]},
            )
            for row in load_hf_rows("openai/openai_humaneval", None, "test")
        ]

    def generation_overrides(self) -> dict[str, Any]:
        # Code repeats tokens legitimately (indentation, names, brackets), so
        # repetition and presence penalties only push the model toward bugs.
        return {"presence_penalty": 0.0, "repetition_penalty": 1.0}

    def grade(self, item: BenchItem, output: str) -> Grade:
        meta = item.meta
        program = assemble_program(meta["prompt"], output, meta["test"], meta["entry_point"])
        status, stderr = run_program(program, self.timeout_s, sandbox=self.sandbox)
        detail: dict[str, Any] = {"status": status}
        if stderr:
            detail["stderr"] = stderr
        return Grade(correct=status == "passed", extracted=status, detail=detail)
