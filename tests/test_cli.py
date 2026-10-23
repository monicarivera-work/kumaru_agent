"""Protect the command line: flag handling, one-shot chat, config, and the bench workflow.

Every command runs in-process against the echo backend, so no model, GPU or
network is needed.
"""

import json
import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kumaru.cli import _build_overrides, app, main
from kumaru.eval.benchmarks import runner
from kumaru.eval.benchmarks.golden import GoldenSet

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
WIDE = {"COLUMNS": "200"}  # rich wraps at the terminal width; keep table rows on one line
ECHO_THOUGHT = (
    "The user asked: hello. Echo cannot reason; this text only exercises the reasoning channel."
)
MINI_SET = [
    {"id": "say", "prompt": "Say kumaru.", "grader": "contains", "answer": "kumaru", "tags": ["easy"]},
    {
        "id": "four",
        "prompt": 'What is 2 + 2? Finish with "Answer: <number>".',
        "grader": "numeric",
        "answer": 4,
        "tags": ["math"],
    },
    # Echo's reply reports "max_tokens=N". A thinking budget of 3 raises N by 3
    # over the answer budget, so only the thinking run gets this right: a known flip.
    {
        "id": "budget",
        "prompt": "Report your output cap.",
        "grader": "contains",
        "answer": f"max_tokens={GoldenSet.answer_tokens + 3}",
        "tags": ["easy"],
    },
]


@pytest.fixture
def echo_profile(scratch_dir: Path) -> Path:
    path = scratch_dir / "echo.yaml"
    path.write_text(
        "backend:\n  name: echo\n  model: echo\n  extra:\n    delay_s: 0\n"
        "generation:\n  max_tokens: 64\n  temperature: 0.0\n",
        encoding="utf-8",
    )
    return path


def run_cli(*args: str) -> str:
    """Run `kumaru <args>` in-process; return its output without colour codes."""
    result = CliRunner().invoke(app, list(args), env=WIDE)
    if result.exit_code != 0:
        raise AssertionError(f"exit code {result.exit_code}:\n{result.output}") from result.exception
    return ANSI.sub("", result.output)


def test_build_overrides_nests_flags_and_drops_only_unset_ones() -> None:
    overrides = _build_overrides(
        model="m", context=4096, temperature=0.0, thinking=False, thinking_budget=0,
        max_tokens=None, not_a_flag="x",
    )  # fmt: skip
    assert overrides == {
        "backend": {"model": "m", "max_context_length": 4096},
        # False and 0 are real choices; only None means "not given".
        "generation": {"temperature": 0.0, "thinking": False, "thinking_budget": 0},
    }


def test_config_shows_that_a_flag_beats_the_profile(echo_profile: Path) -> None:
    text = run_cli("config", "-c", str(echo_profile), "-m", "flag-model")
    assert "model: flag-model" in text
    assert "delay_s: 0" in text  # the rest of the profile still applies


def test_chat_answers_one_question_and_exits(echo_profile: Path) -> None:
    text = run_cli("chat", "-c", str(echo_profile), "-q", "hello")
    assert "You said: hello" in text
    assert "thinking…" not in text


def test_chat_think_prints_the_thought_before_the_answer(echo_profile: Path) -> None:
    text = run_cli("chat", "-c", str(echo_profile), "-q", "hello", "--think")
    assert "thinking…" in text
    assert ECHO_THOUGHT in text
    assert text.index(ECHO_THOUGHT) < text.index("You said: hello")


def test_chat_flags_beat_the_profiles_thinking_preset(scratch_dir: Path) -> None:
    profile = scratch_dir / "thinker.yaml"
    profile.write_text(
        "backend:\n  name: echo\n  model: echo\n  extra:\n    delay_s: 0\n"
        "generation:\n  max_tokens: 64\n  temperature: 0.0\n"
        "thinking_mode:\n  max_tokens: 4000\n  temperature: 1.0\n",
        encoding="utf-8",
    )
    # Echo's reply reports the parameters it was given.
    preset = run_cli("chat", "-c", str(profile), "-q", "hello", "--think")
    assert "max_tokens=4000, temperature=1.0." in preset
    flags = run_cli("chat", "-c", str(profile), "-q", "hello", "--think", "-t", "0.2", "--max-tokens", "50")
    assert "max_tokens=50, temperature=0.2." in flags


def test_bench_list_names_every_benchmark() -> None:
    text = run_cli("bench", "list")
    for name in ("golden", "gsm8k", "mmlu_pro", "ifeval", "humaneval", "multipl_e_py", "multipl_e_rs"):
        assert name in text
    assert "multipl_e " in text  # the suite of all eight languages


def test_bench_run_report_and_compare(
    echo_profile: Path, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "git_commit", lambda: None)  # no subprocesses in tests
    monkeypatch.setattr(runner, "environment", lambda: {"python": "test"})
    golden = scratch_dir / "mini.jsonl"
    golden.write_text("\n".join(json.dumps(row) for row in MINI_SET) + "\n", encoding="utf-8")

    def bench(name: str, *flags: str) -> Path:
        out = scratch_dir / name
        text = run_cli(
            "bench", "run", str(golden), "-c", str(echo_profile), "-o", str(out),
            "-j", "1", "--seed", "7", *flags,
        )  # fmt: skip
        assert "golden-mini" in text  # a custom set is named after its file
        return out

    a = bench("a", "--no-think")
    b = bench("b", "--think", "--thinking-budget", "3")

    run_json = json.loads((b / "run.json").read_text(encoding="utf-8"))
    assert (run_json["label"], run_json["benchmarks"]) == ("echo-think3", ["golden-mini"])
    summary = json.loads((b / "golden-mini" / "summary.json").read_text(encoding="utf-8"))
    assert (summary["n"], summary["n_correct"], summary["thinking_budget"]) == (3, 2, 3)
    assert len((b / "golden-mini" / "items.jsonl").read_text(encoding="utf-8").splitlines()) == 3

    report = run_cli("bench", "report", str(b))
    assert "think (budget 3)" in report
    assert "66.7" in report
    assert "easy 100.0 (n=2)" in report and "math 0.0 (n=1)" in report

    compare = run_cli("bench", "compare", str(a), str(b), "--examples", "3")
    assert "A = echo-nothink   B = echo-think3" in compare
    assert "golden-mini: only B right (1)" in compare
    assert "golden-mini/budget" in compare
    assert "only A right" not in compare


def test_errors_become_a_message_a_hint_and_exit_code_1(
    echo_profile: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # HumanEval runs model-written code, so it must be opted into explicitly.
    monkeypatch.setattr(sys, "argv", ["kumaru", "bench", "run", "humaneval", "-c", str(echo_profile)])
    monkeypatch.setenv("COLUMNS", "200")
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
    out = ANSI.sub("", capsys.readouterr().out)
    assert "error: 'humaneval' grades answers by running model-written code" in out
    assert "hint: Add --allow-code-exec" in out


def test_an_unknown_sandbox_is_refused(
    echo_profile: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["kumaru", "bench", "run", "multipl_e", "--allow-code-exec", "--sandbox", "vm", "-c", str(echo_profile)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("COLUMNS", "200")
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
    out = ANSI.sub("", capsys.readouterr().out)
    assert "error: Unknown sandbox 'vm'." in out
    assert "hint: Use one of: none, docker." in out
