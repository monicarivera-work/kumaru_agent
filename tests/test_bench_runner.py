"""Protect the benchmark runner end to end, on the offline echo backend.

What must never regress: the files a run leaves behind, resuming exactly where
a run stopped, refusing to mix different settings into one run, surviving bad
requests and bad graders, and the paired comparison of two runs.
"""

import asyncio
import dataclasses
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from kumaru.agent import prompt
from kumaru.agent.doc_index import build_index
from kumaru.backends.echo import EchoBackend
from kumaru.core.config import KumaruConfig
from kumaru.core.errors import ConfigError, KumaruError
from kumaru.core.types import GenerationParams
from kumaru.eval.benchmarks import runner
from kumaru.eval.benchmarks.golden import GoldenSet
from kumaru.eval.benchmarks.runner import (
    MAX_CONSECUTIVE_ERRORS,
    BenchSettings,
    BenchSummary,
    Comparison,
    ItemRecord,
    _redact,
    compare_runs,
    default_concurrency,
    default_label,
    item_seed,
    read_records,
    render_comparisons,
    render_flips,
    render_summaries,
    run_benchmarks,
    safe_name,
    verdict,
)

BUDGET = GoldenSet.answer_tokens

GOLDEN_ROWS = [
    {"id": "hello", "prompt": "hello", "answer": "You said: hello", "tags": ["chat"]},
    {"id": "math", "prompt": "two plus two", "answer": "four", "tags": ["math"]},
    # Echo states its output budget, so this passes only at the no-think budget.
    {"id": "budget", "prompt": "budget probe", "answer": rf"max_tokens={BUDGET}\b", "grader": "regex", "tags": ["chat"]},
    # Echo's thought begins "The user asked"; a grader must only ever see the answer.
    {"id": "private", "prompt": "thinking probe", "answer": "The user asked", "grader": "not_contains", "tags": ["chat"]},
]


@pytest.fixture(autouse=True)
def offline_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """No git or nvidia-smi subprocesses: faster, and identical on every machine."""
    monkeypatch.setattr(runner, "git_commit", lambda: "abc123")
    monkeypatch.setattr(runner, "environment", lambda: {"python": "test"})


def write_golden(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def golden(scratch_dir: Path) -> str:
    return str(write_golden(scratch_dir / "mini.jsonl", GOLDEN_ROWS))


async def bench(
    cfg: KumaruConfig, specs: list[str], out: Path, **settings
) -> tuple[Path, list[BenchSummary]]:
    return await run_benchmarks(
        cfg,
        specs,
        BenchSettings(out_dir=out, **settings),
        console=Console(file=io.StringIO()),
        show_progress=False,
    )


def records_of(out: Path, bench_name: str = "golden-mini") -> dict[str, ItemRecord]:
    return read_records(out / bench_name / "items.jsonl")


# -- a run and the files it leaves ----------------------------------------------


async def test_a_run_saves_every_answer_a_summary_and_its_provenance(
    echo_config: KumaruConfig, golden: str, scratch_dir: Path
) -> None:
    echo_config.server.api_key = "s3cret-key"
    out = scratch_dir / "run"
    out_dir, [summary] = await bench(echo_config, [golden, golden], out)  # duplicates collapse

    assert out_dir == out
    assert (summary.benchmark, summary.label, summary.model) == ("golden-mini", "echo-default", "echo")
    assert (summary.n, summary.n_planned, summary.n_correct, summary.complete) == (4, 4, 3, True)
    assert summary.by_group == {"chat": {"n": 3, "accuracy": 1.0}, "math": {"n": 1, "accuracy": 0.0}}
    assert summary.params["max_tokens"] == BUDGET
    assert summary.git_commit == "abc123"

    records = records_of(out)
    assert list(records) == [f"golden-mini/{row['id']}" for row in GOLDEN_ROWS]
    hello = records["golden-mini/hello"]
    assert hello.correct and hello.error is None and hello.reasoning is None
    assert hello.seed == item_seed(1234, "golden-mini/hello")
    assert f"max_tokens={BUDGET}" in hello.output
    assert json.loads((out / "golden-mini" / "summary.json").read_text(encoding="utf-8"))["n_correct"] == 3

    text = (out / "run.json").read_text(encoding="utf-8")
    run_json = json.loads(text)
    assert (run_json["label"], run_json["benchmarks"]) == ("echo-default", ["golden-mini"])
    assert "s3cret-key" not in text
    assert run_json["config"]["server"]["api_key"] == "<redacted>"
    # Counts are settings, not secrets: they must stay readable.
    assert run_json["config"]["generation"]["max_tokens"] == 32
    assert run_json["config"]["agent"]["max_history_tokens"] == echo_config.agent.max_history_tokens


async def test_thinking_is_recorded_but_never_graded(
    echo_config: KumaruConfig, golden: str, scratch_dir: Path
) -> None:
    out, [summary] = await bench(echo_config, [golden], scratch_dir / "think", thinking=True, thinking_budget=5)

    assert summary.label == "echo-think5"
    assert (summary.thinking, summary.thinking_budget) == (True, 5)
    # Thinking adds its budget on top of the answer budget.
    assert summary.params["max_tokens"] == BUDGET + 5
    records = records_of(out)
    private = records["golden-mini/private"]
    assert private.reasoning.startswith("The user asked")
    assert private.correct
    assert not records["golden-mini/budget"].correct
    assert summary.n_correct == 2


# -- resuming -------------------------------------------------------------------------


async def test_rerunning_into_the_same_directory_asks_only_what_is_missing(
    echo_config: KumaruConfig, golden: str, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = scratch_dir / "resume"
    await bench(echo_config, [golden], out)
    items = out / "golden-mini" / "items.jsonl"
    lines = items.read_text(encoding="utf-8").splitlines(keepends=True)
    # As after a hard kill: two answers saved, the third torn mid-line, no newline.
    items.write_text("".join(lines[:2]) + lines[2][:25], encoding="utf-8")

    asked: list[str] = []
    original = EchoBackend.generate

    async def counting(self, messages, params):
        asked.append(messages[-1].content)
        return await original(self, messages, params)

    monkeypatch.setattr(EchoBackend, "generate", counting)
    _, [summary] = await bench(echo_config, [golden], out)

    assert asked == ["budget probe", "thinking probe"]
    assert summary.complete and summary.n_correct == 3
    # The new answers start on a fresh line instead of being glued to the torn one.
    assert len(records_of(out)) == 4
    assert len(list(out.glob("run.resumed-*.json"))) == 1


async def test_resuming_with_different_settings_is_refused_before_anything_is_written(
    echo_config: KumaruConfig, golden: str, scratch_dir: Path
) -> None:
    out = scratch_dir / "refuse"
    await bench(echo_config, [golden], out)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}

    with pytest.raises(ConfigError, match="different settings"):
        await bench(echo_config, [golden], out, temperature=0.5)

    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before


# -- failures -----------------------------------------------------------------------


async def test_repeated_request_failures_stop_the_run_and_a_rerun_retries_them(
    echo_config: KumaruConfig, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [{"id": str(i), "prompt": f"question {i}", "answer": "You said"} for i in range(12)]
    spec = str(write_golden(scratch_dir / "many.jsonl", rows))
    out = scratch_dir / "fail"
    original = EchoBackend.generate
    server = {"down": True}

    async def flaky(self, messages, params):
        await asyncio.sleep(0)
        if server["down"]:
            raise ConnectionError("server went away")
        return await original(self, messages, params)

    monkeypatch.setattr(EchoBackend, "generate", flaky)
    with pytest.raises(KumaruError, match="questions failed in a row"):
        await bench(echo_config, [spec], out)

    failed = records_of(out, "golden-many")
    # Stopped early; questions already in flight may finish, so not exactly five.
    assert MAX_CONSECUTIVE_ERRORS <= len(failed) < len(rows)
    assert {r.error for r in failed.values()} == {"ConnectionError: server went away"}
    partial = json.loads((out / "golden-many" / "summary.json").read_text(encoding="utf-8"))
    assert (partial["n_errors"], partial["complete"]) == (len(failed), False)

    server["down"] = False
    _, [summary] = await bench(echo_config, [spec], out)
    assert (summary.n, summary.n_errors, summary.n_correct) == (12, 0, 12)
    assert all(r.error is None for r in records_of(out, "golden-many").values())


async def test_a_grader_crash_is_recorded_against_the_question_not_raised(
    echo_config: KumaruConfig, golden: str, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_grader(self, item, output):
        raise ValueError("bad pattern")

    monkeypatch.setattr(GoldenSet, "grade", broken_grader)
    out, [summary] = await bench(echo_config, [golden], scratch_dir / "grader")

    records = records_of(out)
    assert {r.error for r in records.values()} == {"grader failed: ValueError: bad pattern"}
    assert all(r.output for r in records.values())  # the reply is kept for inspection
    assert (summary.n_errors, summary.n_correct, summary.extraction) == (4, 0, {"error": 4})


async def test_a_program_kumaru_cannot_contain_is_an_error_not_a_wrong_answer(
    echo_config: KumaruConfig, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kumaru.eval.benchmarks import codeexec, humaneval

    class NoJob:
        def __init__(self) -> None:
            raise OSError(1455, "The paging file is too small for this operation to complete")

    def popen(*args, **kwargs):
        raise AssertionError("started outside a job")

    rows = [{"task_id": f"HumanEval/{n}", "entry_point": "add", "prompt": "def add(a, b):\n", "test": ""} for n in range(2)]
    monkeypatch.setattr(humaneval, "load_hf_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(codeexec, "_WINDOWS", True)
    monkeypatch.setattr(codeexec, "_Job", NoJob)
    monkeypatch.setattr(codeexec.subprocess, "Popen", popen)
    out, [summary] = await bench(echo_config, ["humaneval"], scratch_dir / "code", allow_code_exec=True)

    errors = {r.error for r in records_of(out, "humaneval").values()}
    assert len(errors) == 1
    assert errors.pop().startswith("grader failed: KumaruError: Could not create the Windows job object")
    assert (summary.n, summary.n_errors, summary.n_correct) == (2, 2, 0)  # retried on resume


async def test_a_grader_failing_every_question_stops_the_run_and_says_so(
    echo_config: KumaruConfig, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_grader(self, item, output):
        raise ValueError("bad pattern")

    monkeypatch.setattr(GoldenSet, "grade", broken_grader)
    rows = [{"id": str(i), "prompt": f"question {i}", "answer": "x"} for i in range(MAX_CONSECUTIVE_ERRORS + 2)]
    spec = str(write_golden(scratch_dir / "graded.jsonl", rows))
    with pytest.raises(KumaruError, match="questions failed in a row") as caught:
        await bench(echo_config, [spec], scratch_dir / "streak")
    assert "grader failed" in str(caught.value.hint)


async def test_code_execution_benchmarks_need_explicit_consent(
    echo_config: KumaruConfig, scratch_dir: Path
) -> None:
    # Refused while resolving names: no download, no model load, nothing written.
    with pytest.raises(ConfigError, match="model-written code"):
        await bench(echo_config, ["humaneval"], scratch_dir / "code")
    # A suite is named as the user wrote it; "on this machine" unless sandboxed.
    with pytest.raises(ConfigError, match="'multipl_e' grades answers by running model-written code on this machine"):
        await bench(echo_config, ["multipl_e"], scratch_dir / "code")
    with pytest.raises(ConfigError, match=r"'humaneval' grades answers by running model-written code\. "):
        await bench(echo_config, ["humaneval"], scratch_dir / "code", sandbox="docker")
    with pytest.raises(ConfigError, match="No benchmarks given"):
        await bench(echo_config, [], scratch_dir / "none")
    assert not (scratch_dir / "code").exists()


async def test_a_suite_skips_languages_without_a_toolchain_and_says_so(
    echo_config: KumaruConfig, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kumaru.eval.benchmarks import codeexec, multipl_e
    from kumaru.eval.benchmarks.codeexec import Outcome, ToolchainMissing

    absent = [lang for lang in multipl_e.LANGUAGES if lang != "py"]

    def not_installed(lang: str):
        def detect():
            raise ToolchainMissing(f"No {lang} here.", hint=f"Install {lang} (see its site).")

        return detect

    for lang in absent:
        monkeypatch.setitem(multipl_e._DETECT, lang, not_installed(lang))
    rows = [
        {"task_id": f"HumanEval/{n}", "entry_point": f"f{n}", "prompt": f"def f{n}():\n", "test": "\n"}
        for n in range(164)
    ]
    monkeypatch.setattr(multipl_e, "load_hf_rows", lambda *args, **kwargs: rows)

    def run(program, *, sandbox="none"):  # no subprocesses: only the canary's right answer passes
        passed = "return x + y" in "".join(program.files.values())
        return Outcome("passed" if passed else "failed", "run", 0 if passed else 1, "", "", 0.1)

    monkeypatch.setattr(codeexec, "run", run)
    out, console = scratch_dir / "suite", io.StringIO()
    suite = runner.BenchRunner(
        echo_config,
        BenchSettings(out_dir=out, limit=2, allow_code_exec=True),
        console=Console(file=console, width=200),
        show_progress=False,
    )
    [summary] = await suite.run(["multipl-e"])

    assert (summary.benchmark, summary.n) == ("multipl_e_py", 2)
    expected = {f"multipl_e_{lang}": f"No {lang} here. Install {lang} (see its site)." for lang in absent}
    assert suite.skipped == expected
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert (run_json["benchmarks"], run_json["skipped"]) == (["multipl_e_py"], expected)
    assert "skipped multipl_e_rs: No rs here. Install rs (see its site)." in console.getvalue()

    # Asked for by name, a missing toolchain is an error, not a skip.
    with pytest.raises(ToolchainMissing, match="No rs here"):
        await bench(echo_config, ["multipl_e_rs"], scratch_dir / "rs", allow_code_exec=True)
    # Another compiler version is another run.
    real = multipl_e._DETECT["py"]
    monkeypatch.setitem(multipl_e._DETECT, "py", lambda: dataclasses.replace(real(), version="python 0.0"))
    with pytest.raises(ConfigError, match="different settings"):
        await bench(echo_config, ["multipl_e_py"], out, allow_code_exec=True, limit=2)


async def test_two_golden_files_with_the_same_name_are_refused(
    echo_config: KumaruConfig, scratch_dir: Path
) -> None:
    # Both would be "golden-set": one results folder, so the second was skipped.
    (scratch_dir / "a").mkdir()
    (scratch_dir / "b").mkdir()
    one = str(write_golden(scratch_dir / "a" / "set.jsonl", GOLDEN_ROWS))
    two = str(write_golden(scratch_dir / "b" / "set.jsonl", GOLDEN_ROWS[:1]))
    with pytest.raises(ConfigError, match="share the name 'golden-set'"):
        await bench(echo_config, [one, two], scratch_dir / "run")
    assert not (scratch_dir / "run").exists()


async def test_agent_prompt_runs_resume_the_next_day_without_asking_again(
    echo_config: KumaruConfig, golden: str, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = scratch_dir / "agent"
    systems: list[str] = []
    original = EchoBackend.generate

    async def spy(self, messages, params):
        systems.append(messages[0].content)
        return await original(self, messages, params)

    monkeypatch.setattr(EchoBackend, "generate", spy)
    await bench(echo_config, [golden], out, use_agent_prompt=True)
    assert len(systems) == 4 and "Today's date is" in systems[0]

    tomorrow = datetime.now() + timedelta(days=1)
    monkeypatch.setattr(prompt, "_dt", SimpleNamespace(datetime=SimpleNamespace(now=lambda: tomorrow)))
    _, [summary] = await bench(echo_config, [golden], out, use_agent_prompt=True)
    assert len(systems) == 4  # every answer reused
    assert summary.complete

    # A different persona is a different run, though.
    echo_config.agent.system_prompt = "You are terse."
    with pytest.raises(ConfigError, match="different settings"):
        await bench(echo_config, [golden], out, use_agent_prompt=True)


def fingerprint(cfg: KumaruConfig, spec: str, **settings) -> str:
    """The fingerprint a run of `spec` would get, without running it."""
    benchmark = runner.resolve_benchmark(spec)
    return runner.BenchRunner(cfg, BenchSettings(**settings))._fingerprint(
        benchmark, "echo", GenerationParams()
    )


def test_documentation_indexes_join_the_fingerprint_by_content_when_used(
    echo_config: KumaruConfig, golden: str, scratch_dir: Path
) -> None:
    docs = scratch_dir / "docs"
    docs.mkdir()
    page = docs / "math.md"
    page.write_text("# Arithmetic\n\nTwo plus two is four.\n", encoding="utf-8")
    index, moved = scratch_dir / "math.sqlite", scratch_dir / "moved.sqlite"
    build_index(docs, index, topic="Math", version="1")
    agent = echo_config.agent

    bare, plain = fingerprint(echo_config, golden), fingerprint(echo_config, golden, use_agent_prompt=True)
    agent.max_retrieved_chars = 1234
    # Unused without an index: runs saved before indexes existed keep their fingerprints.
    assert fingerprint(echo_config, golden, use_agent_prompt=True) == plain

    agent.knowledge_indexes = [str(index)]
    assert fingerprint(echo_config, golden) == bare  # only the agent prompt retrieves
    with_index = fingerprint(echo_config, golden, use_agent_prompt=True)
    assert with_index != plain
    moved.write_bytes(index.read_bytes())
    agent.knowledge_indexes = [str(moved)]
    assert fingerprint(echo_config, golden, use_agent_prompt=True) == with_index  # same content
    agent.max_retrieved_chars = 4321
    assert fingerprint(echo_config, golden, use_agent_prompt=True) != with_index

    agent.max_retrieved_chars = 1234
    page.write_text("# Arithmetic\n\nTwo plus two is four, not five.\n", encoding="utf-8")
    moved.unlink()
    build_index(docs, moved, topic="Math", version="1")  # rebuilt where it was
    assert fingerprint(echo_config, golden, use_agent_prompt=True) != with_index


# -- comparing two runs ----------------------------------------------------------------


async def test_comparison_pairs_questions_and_shows_what_changed(
    echo_config: KumaruConfig, golden: str, scratch_dir: Path
) -> None:
    a, a_summaries = await bench(echo_config, [golden], scratch_dir / "a")
    b, _ = await bench(echo_config, [golden], scratch_dir / "b", thinking=True, thinking_budget=5)

    [c] = compare_runs(a, b, resamples=500)
    assert (c.benchmark, c.baseline, c.candidate) == ("golden-mini", "echo-default", "echo-think5")
    assert (c.n, c.unpaired, c.baseline_only, c.candidate_only) == (4, 0, 1, 0)
    assert c.delta == pytest.approx(-0.25)
    assert c.p_value == 1.0  # a single discordant pair proves nothing
    assert verdict(c) == "no clear difference"

    console = Console(file=io.StringIO(), width=200)
    render_summaries(console, a_summaries, details=True)
    render_comparisons(console, [c])
    render_flips(console, a, b, limit=5)
    shown = console.file.getvalue()
    assert "only A right (1)" in shown and "golden-mini/budget" in shown
    assert "only B right" not in shown


async def test_runs_with_no_benchmark_in_common_cannot_be_compared(
    echo_config: KumaruConfig, golden: str, scratch_dir: Path
) -> None:
    other = str(write_golden(scratch_dir / "other.jsonl", GOLDEN_ROWS[:1]))
    a, _ = await bench(echo_config, [golden], scratch_dir / "a")
    b, _ = await bench(echo_config, [other], scratch_dir / "b")
    with pytest.raises(ConfigError, match="share no benchmark"):
        compare_runs(a, b)


def comparison(**fields) -> Comparison:
    defaults = dict(
        benchmark="b", baseline="A", candidate="B", n=100,
        baseline_accuracy=0.5, baseline_ci=(0.4, 0.6), candidate_accuracy=0.5,
        candidate_ci=(0.4, 0.6), delta=0.0, delta_ci=(-0.1, 0.1), p_value=1.0,
        p_holm=1.0, baseline_only=0, candidate_only=0,
    )  # fmt: skip
    return Comparison(**{**defaults, **fields})


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"n": 0}, "no paired questions"),
        ({"delta": 0.1, "p_value": 0.001, "p_holm": 0.004}, "B better"),
        ({"delta": -0.1, "p_value": 0.001, "p_holm": 0.004}, "A better"),
        ({"delta": 0.1, "p_value": 0.03, "p_holm": 0.09}, "B better? (not after correction)"),
        ({"delta": 0.02, "p_value": 0.6, "p_holm": 1.0}, "no clear difference"),
    ],
)
def test_verdict_reads_the_corrected_p_value(fields: dict, expected: str) -> None:
    assert verdict(comparison(**fields)) == expected


# -- small helpers ---------------------------------------------------------------------


def test_default_label_names_the_model_and_the_thinking_mode() -> None:
    gguf = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS"
    assert default_label(gguf, None, GenerationParams(thinking=False)) == "Qwen3.6-35B-A3B-GGUF-nothink"
    assert default_label("echo", None, GenerationParams()) == "echo-default"
    profile = "configs/qwen36.yaml"
    assert default_label(gguf, profile, GenerationParams(thinking=True)) == "qwen36-think"
    assert default_label(gguf, profile, GenerationParams(thinking=True, thinking_budget=4096)) == "qwen36-think4096"


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (r"D:\kumaru\models\Qwen3-8B-Q4_K_M.gguf", "Qwen3-8B-Q4_K_M"),  # was "D"
        ("C:/models/llama/", "llama"),
        ("/srv/models/model.safetensors", "model"),
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),  # not a file extension
    ],
)
def test_default_label_takes_the_name_from_a_model_path(model_id: str, expected: str) -> None:
    assert default_label(model_id, None, GenerationParams()) == f"{expected}-default"


def test_summary_lines_say_how_each_answer_was_read() -> None:
    def shown(n: int, extraction: dict[str, int]) -> str:
        console = Console(file=io.StringIO(), width=200)
        summary = BenchSummary(benchmark="b", label="l", model="m", backend="echo", n=n, extraction=extraction)
        render_summaries(console, [summary])
        return console.file.getvalue()

    assert "answers found by: answer_tag 14, whole reply 13" in shown(27, {"answer_tag": 14})
    assert "tests: passed 80, failed 19, error 1" in shown(100, {"passed": 80, "failed": 19, "error": 1})
    nothing = shown(100, {})  # IFEval: every grader reads the whole reply
    assert "answers found by" not in nothing and "tests:" not in nothing


def test_safe_name_produces_portable_directory_names() -> None:
    assert safe_name("a b/c:d") == "a-b-c-d"
    assert safe_name("-qwen3.6-") == "qwen3.6"
    assert safe_name("...") == "run"


def test_item_seeds_are_pinned_because_old_runs_pair_on_them() -> None:
    # Changing this function silently unpairs every earlier run from every new one.
    assert item_seed(1234, "gsm8k/0") == 127331135
    assert item_seed(1234, "golden/1") == 2132921838
    assert item_seed(1235, "gsm8k/0") != item_seed(1234, "gsm8k/0")
    assert 0 <= item_seed(7, "x") < 2**31


def test_default_concurrency_follows_llama_server_slots_and_shared_context() -> None:
    def setup(*, unified: bool, slots: int | None = 4, context: int = 65_536):
        cfg = SimpleNamespace(backend=SimpleNamespace(llama_server=SimpleNamespace(parallel=2, kv_unified=unified)))
        info = SimpleNamespace(extra={"slots": slots} if slots else {}, context_length=context)
        return cfg, SimpleNamespace(name="llamacpp_server", info=lambda: info)

    assert default_concurrency(None, SimpleNamespace(name="echo"), 100) == 1
    assert default_concurrency(*setup(unified=False), 50_000) == 4
    assert default_concurrency(*setup(unified=False, slots=None), 50_000) == 2  # configured parallel
    assert default_concurrency(*setup(unified=True), 20_000) == 3  # 65536 // 20000
    assert default_concurrency(*setup(unified=True), 200_000) == 1  # never below one


def test_default_concurrency_fills_a_transformers_batch_only_for_greedy_runs() -> None:
    cfg = SimpleNamespace(backend=SimpleNamespace(max_batch_size=16))
    backend = SimpleNamespace(name="transformers")
    greedy, sampled = GenerationParams(temperature=0.0), GenerationParams(temperature=0.7)

    assert default_concurrency(cfg, backend, 100, params=greedy) == 16
    assert default_concurrency(cfg, backend, 100, params=sampled) == 1  # sampling never batches
    assert default_concurrency(cfg, backend, 100) == 1


def test_redact_hides_credentials_but_never_counts() -> None:
    config = {
        "backend": {"api_key": "sk-1", "max_tokens": 32, "tokenizer": "Qwen/Qwen3",
                    "extra": {"hf_token": "hf_x", "headers": {"x-api-key": "k", "Authorization": "b"}}},
        "server": {"api_key": None, "password": ""},
        "agent": {"max_history_tokens": 6144},
        "list": [{"client_secret": "s"}],
    }  # fmt: skip
    assert _redact(config) == {
        "backend": {"api_key": "<redacted>", "max_tokens": 32, "tokenizer": "Qwen/Qwen3",
                    "extra": {"hf_token": "<redacted>",
                              "headers": {"x-api-key": "<redacted>", "Authorization": "<redacted>"}}},
        "server": {"api_key": None, "password": ""},
        "agent": {"max_history_tokens": 6144},
        "list": [{"client_secret": "<redacted>"}],
    }  # fmt: skip


def test_read_records_keeps_the_last_answer_per_question_and_skips_torn_lines(scratch_dir: Path) -> None:
    assert read_records(scratch_dir / "missing.jsonl") == {}
    path = scratch_dir / "items.jsonl"
    path.write_text(
        ItemRecord(id="q1", error="TimeoutError: slow").model_dump_json() + "\n"
        + ItemRecord(id="q2", correct=True).model_dump_json() + "\n"
        + "\n"
        + '{"id": "q3", "corr\n'
        + ItemRecord(id="q1", correct=True).model_dump_json() + "\n",
        encoding="utf-8",
    )
    records = read_records(path)
    assert list(records) == ["q1", "q2"]
    assert records["q1"].correct and records["q1"].error is None
