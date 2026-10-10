"""Protect the llama-server backend: model resolution, command line, payload, failures.

Nothing here starts a process or reaches the internet; a few tests open
sockets on the loopback interface. Starting a real llama-server is covered by
`scripts\\verify.ps1` on a machine that has the binary and the weights.
"""

import socket
import sys
from pathlib import Path

import httpx
import pytest

import kumaru.backends.llamacpp_server as llama_server
from kumaru.backends.llamacpp_server import (
    LlamaServerBackend,
    build_server_args,
    dial_address,
    find_llama_server,
    free_port,
    match_gguf_files,
    parse_model_spec,
    port_in_use,
    resolve_model_file,
    shards_complete,
    url_host,
)
from kumaru.core.config import BackendConfig, LlamaServerOptions
from kumaru.core.errors import BackendError
from kumaru.core.types import GenerationParams, Message

SPEC = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS"


def server_config(**overrides) -> BackendConfig:
    return BackendConfig(name="llamacpp_server", model=SPEC, **overrides)


# -- model specs and file matching ----------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (SPEC, ("unsloth/Qwen3.6-35B-A3B-GGUF", "UD-IQ4_XS")),
        ("org/repo", ("org/repo", None)),  # no quant: fine if the repo has one GGUF
        ("org/repo:", ("org/repo", None)),
        ("models/qwen.gguf", None),  # a relative file path, not a repo
        ("model.GGUF", None),
        ("D:\\kumaru\\models\\qwen", None),
        ("/srv/models/qwen", None),
        ("org/sub/repo:Q4_0", None),
        ("repo-only:Q4_0", None),
        ("org/:Q4_0", None),
    ],
)
def test_parse_model_spec(spec: str, expected: tuple[str, str | None] | None) -> None:
    assert parse_model_spec(spec) == expected


LISTING = [
    "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
    "Qwen3.6-35B-A3B-IQ4_XS.gguf",
    "Qwen3.6-35B-A3B-Q4_K_M.gguf",
    "Q8_0/Qwen3.6-35B-A3B-Q8_0-00002-of-00002.gguf",
    "Q8_0/Qwen3.6-35B-A3B-Q8_0-00001-of-00002.gguf",
    "mmproj-F16.gguf",
    "README.md",
]


@pytest.mark.parametrize(
    ("quant", "expected"),
    [
        ("UD-IQ4_XS", ["Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"]),
        ("ud-iq4_xs", ["Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"]),  # case-insensitive
        ("IQ4_XS", ["Qwen3.6-35B-A3B-IQ4_XS.gguf"]),  # the shortest name wins over UD-
        ("4_K_M", []),  # must align to a separator, so "Q4_K_M" does not match
        ("F16", []),  # only the vision projector has it, and that is excluded
        (
            "Q8_0",  # every shard, first shard first
            [
                "Q8_0/Qwen3.6-35B-A3B-Q8_0-00001-of-00002.gguf",
                "Q8_0/Qwen3.6-35B-A3B-Q8_0-00002-of-00002.gguf",
            ],
        ),
    ],
)
def test_match_gguf_files(quant: str, expected: list[str]) -> None:
    assert match_gguf_files(LISTING, quant) == expected


def test_match_without_a_quant_requires_a_single_model() -> None:
    assert match_gguf_files(LISTING, None) == []  # ambiguous: the caller must name one
    assert match_gguf_files(["only.gguf", "mmproj-only.gguf"], None) == ["only.gguf"]


def test_match_breaks_ties_by_folder_and_accepts_dot_separators() -> None:
    assert match_gguf_files(["b/Model-Q4_0.gguf", "a/Model-Q4_0.gguf"], "Q4_0") == [
        "a/Model-Q4_0.gguf"
    ]  # the same answer whatever order the listing came in
    assert match_gguf_files(["model.Q4_0.gguf"], "Q4_0") == ["model.Q4_0.gguf"]


# -- command line ----------------------------------------------------------------------


def flag(args: list[str], name: str) -> str | None:
    """The value after `name`, or None when the flag is absent."""
    return args[args.index(name) + 1] if name in args else None


def test_default_command_line_is_exact() -> None:
    args = build_server_args(Path("llama-server"), Path("m.gguf"), server_config(), 8080)
    assert args == [
        "llama-server", "--model", "m.gguf", "--host", "127.0.0.1", "--port", "8080",
        "--ctx-size", "8192", "--parallel", "4", "--flash-attn", "on", "--jinja",
        "--alias", SPEC, "--no-webui", "--log-colors", "off", "--kv-unified",
    ]  # fmt: skip
    # n_gpu_layers=-1 leaves --n-gpu-layers out, so llama.cpp fits layers to free VRAM.


def test_every_knob_reaches_the_command_line() -> None:
    cfg = server_config(
        max_context_length=65536,
        n_gpu_layers=0,
        n_threads=8,
        llama_server=LlamaServerOptions(
            host="0.0.0.0", parallel=2, kv_unified=False, flash_attn="auto",
            cache_type_k="q8_0", cache_type_v="q4_0", n_cpu_moe=18,
            batch_size=2048, ubatch_size=512, extra_args=["--mlock", "--top-k", "20"],
        ),  # fmt: skip
    )
    args = build_server_args(Path("bin"), Path("m.gguf"), cfg, 1234)
    expected = {
        "--host": "0.0.0.0", "--port": "1234", "--ctx-size": "65536", "--parallel": "2",
        "--flash-attn": "auto", "--n-gpu-layers": "0", "--threads": "8",
        "--cache-type-k": "q8_0", "--cache-type-v": "q4_0", "--n-cpu-moe": "18",
        "--batch-size": "2048", "--ubatch-size": "512",
    }  # fmt: skip
    assert {name: flag(args, name) for name in expected} == expected
    assert "--kv-unified" not in args
    assert args[-3:] == ["--mlock", "--top-k", "20"]  # extra args go last, verbatim


def test_zero_moe_offload_is_passed_but_unset_is_omitted() -> None:
    zero = server_config(llama_server=LlamaServerOptions(n_cpu_moe=0))
    assert flag(build_server_args(Path("bin"), Path("m.gguf"), zero, 1), "--n-cpu-moe") == "0"
    assert "--n-cpu-moe" not in build_server_args(Path("bin"), Path("m.gguf"), server_config(), 1)


# -- request payload -------------------------------------------------------------------


class TestPayload:
    def test_llama_cpp_spellings_and_the_thinking_budget(self) -> None:
        cfg = server_config(llama_server=LlamaServerOptions(reasoning_budget_message="Answer now."))
        params = GenerationParams(thinking=True, thinking_budget=512, repetition_penalty=1.1)
        fields = LlamaServerBackend(cfg)._extended_fields(params)
        assert fields["repeat_penalty"] == 1.1  # llama.cpp's name for it
        assert "repetition_penalty" not in fields
        assert fields["reasoning_budget_tokens"] == 512
        assert fields["reasoning_budget_message"] == "Answer now."
        assert fields["chat_template_kwargs"] == {"enable_thinking": True}

    @pytest.mark.parametrize(("thinking", "budgeted"), [(True, True), (None, True), (False, False)])
    def test_budget_is_sent_unless_thinking_is_off(
        self, thinking: bool | None, budgeted: bool
    ) -> None:
        # None means "the model's default", which for Qwen3.6 is to think.
        params = GenerationParams(thinking=thinking, thinking_budget=64)
        fields = LlamaServerBackend(server_config())._extended_fields(params)
        assert ("reasoning_budget_tokens" in fields) is budgeted

    def test_neutral_values_add_nothing(self) -> None:
        params = GenerationParams(top_k=0, repetition_penalty=1.0)
        assert LlamaServerBackend(server_config())._extended_fields(params) == {}

    def test_payload_carries_the_extensions(self) -> None:
        params = GenerationParams(thinking=True, thinking_budget=64)
        payload = LlamaServerBackend(server_config())._payload(
            [Message.user("hi")], params, stream=True
        )
        assert payload["model"] == SPEC
        assert payload["reasoning_budget_tokens"] == 64


# -- introspection ---------------------------------------------------------------------


class TestInfo:
    def test_managed_server_before_start(self) -> None:
        info = LlamaServerBackend(server_config(max_context_length=65536)).info()
        assert info.quantization == "UD-IQ4_XS"
        assert info.context_length == 65536
        assert info.device == "gpu (auto-fit)"
        assert info.extra["managed"] is True
        assert "pid" not in info.extra  # None values are dropped until a process exists

    def test_attached_server_is_not_managed(self) -> None:
        cfg = BackendConfig(
            name="llamacpp_server", model="local.gguf", base_url="http://127.0.0.1:8080/v1"
        )
        info = LlamaServerBackend(cfg).info()
        assert info.extra["managed"] is False
        assert info.extra["base_url"] == "http://127.0.0.1:8080/v1"
        assert info.quantization == "gguf"  # not a repo spec, so no quant to report

    def test_context_length_prefers_what_the_server_reports(self) -> None:
        backend = LlamaServerBackend(server_config(n_gpu_layers=0))
        backend._props = {
            "default_generation_settings": {"n_ctx": 131072},
            "build_info": "b1234",
            "total_slots": 4,
        }
        info = backend.info()
        assert info.context_length == 131072
        assert (info.extra["build"], info.extra["slots"]) == ("b1234", 4)
        assert info.device == "cpu"

    @pytest.mark.parametrize(
        ("overrides", "device"),
        [
            ({}, "gpu (auto-fit)"),
            ({"n_gpu_layers": 0}, "cpu"),
            ({"n_gpu_layers": 20}, "gpu (20 layers)"),
            ({"base_url": "http://127.0.0.1:8080/v1"}, "external server"),
        ],
    )
    def test_device_describes_the_layer_setting(self, overrides: dict, device: str) -> None:
        assert LlamaServerBackend(server_config(**overrides)).info().device == device


# -- failures while serving ------------------------------------------------------------


class FakeProcess:
    """Stands in for the llama-server child process; `code` None means still running."""

    pid = 4242

    def __init__(self, code: int | None) -> None:
        self.code = code

    def poll(self) -> int | None:
        return self.code


async def test_a_server_that_dies_mid_request_is_reported_with_its_exit_code_and_log(
    scratch_dir: Path,
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    backend = LlamaServerBackend(server_config())
    backend._client = httpx.AsyncClient(base_url=backend._base_url, transport=httpx.MockTransport(refuse))
    backend._loaded = True
    backend._log_path = scratch_dir / "llama-server.log"
    backend._log_path.write_text("loading model\nCUDA error: out of memory\n", encoding="utf-8")
    ask = [Message.user("hi")], GenerationParams(max_tokens=4)
    try:
        backend._proc = FakeProcess(3)  # type: ignore[assignment]
        with pytest.raises(BackendError, match=r"exited \(code 3\) during the request") as error:
            await backend.generate(*ask)
        assert "CUDA error: out of memory" in error.value.message  # the log's last lines
        assert "Restart" in (error.value.hint or "")

        backend._proc = FakeProcess(None)  # type: ignore[assignment]  # alive: a network problem
        with pytest.raises(BackendError, match="Lost the connection"):
            await backend.generate(*ask)
    finally:
        backend._proc = None
        await backend._client.aclose()


async def test_a_fixed_port_that_is_taken_is_refused_before_anything_starts(
    scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llama_server, "find_llama_server", lambda binary: scratch_dir / "server.exe")
    monkeypatch.setattr(llama_server, "resolve_model_file", lambda config: scratch_dir / "m.gguf")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        options = LlamaServerOptions(port=port, log_file=str(scratch_dir / "server.log"))
        backend = LlamaServerBackend(server_config(llama_server=options))
        # Otherwise the health check would reach whatever already listens there.
        with pytest.raises(BackendError, match=f"Port {port} on 127.0.0.1 is already in use"):
            await backend.load()
    assert backend._proc is None and not (scratch_dir / "server.log").exists()


class TestAddresses:
    @pytest.mark.parametrize(
        ("host", "dialled", "in_url"),
        [
            ("127.0.0.1", "127.0.0.1", "127.0.0.1"),
            ("0.0.0.0", "127.0.0.1", "127.0.0.1"),
            ("", "127.0.0.1", "127.0.0.1"),
            ("::", "::1", "[::1]"),
            ("[::]", "::1", "[::1]"),
            ("::1", "::1", "[::1]"),
            ("localhost", "localhost", "localhost"),
        ],
    )
    def test_wildcard_binds_are_reached_on_loopback(self, host: str, dialled: str, in_url: str) -> None:
        assert dial_address(host) == dialled
        assert url_host(host) == in_url

    def test_port_in_use_sees_a_listener(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            assert port_in_use("127.0.0.1", port)
            assert port_in_use("0.0.0.0", port)  # a wildcard host is checked on loopback
        assert not port_in_use("127.0.0.1", port)

    def test_free_port_handles_ipv6_hosts(self) -> None:
        try:
            with socket.socket(socket.AF_INET6) as probe:
                probe.bind(("::1", 0))
        except OSError:
            pytest.skip("this machine has no IPv6 loopback")
        assert free_port("::1") > 0
        assert free_port("[::1]") > 0


# -- finding the binary and the weights -------------------------------------------------


def test_find_llama_server_search_order(scratch_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = scratch_dir / "custom-llama-server.exe"
    explicit.write_bytes(b"")
    assert find_llama_server(str(explicit)) == explicit
    with pytest.raises(BackendError, match="does not exist"):
        find_llama_server(str(scratch_dir / "missing.exe"))

    bundled = scratch_dir / "bundled-llama-server.exe"
    monkeypatch.setattr(llama_server, "BUNDLED_BINARY", bundled)
    monkeypatch.setattr(llama_server.shutil, "which", lambda name: None)
    with pytest.raises(BackendError, match="not found") as error:
        find_llama_server(None)
    assert "get-llama-server.ps1" in (error.value.hint or "")

    monkeypatch.setattr(llama_server.shutil, "which", lambda name: "C:/bin/llama-server.exe")
    assert find_llama_server(None) == Path("C:/bin/llama-server.exe")  # PATH as a last resort
    bundled.write_bytes(b"")
    assert find_llama_server(None) == bundled  # ...the bundled copy before it


class TestResolveModelFile:
    def test_explicit_model_path_wins_and_must_exist(self, scratch_dir: Path) -> None:
        weights = scratch_dir / "m.gguf"
        weights.write_bytes(b"GGUF")
        assert resolve_model_file(server_config(model_path=str(weights))) == weights
        with pytest.raises(BackendError, match="does not exist"):
            resolve_model_file(server_config(model_path=str(scratch_dir / "gone.gguf")))

    def test_a_local_file_as_the_model_is_used_directly(self, scratch_dir: Path) -> None:
        weights = scratch_dir / "m.gguf"
        weights.write_bytes(b"GGUF")
        cfg = BackendConfig(name="llamacpp_server", model=str(weights))
        assert resolve_model_file(cfg) == weights

    def test_neither_a_file_nor_a_repo_explains_the_format(self) -> None:
        cfg = BackendConfig(name="llamacpp_server", model="not-a-repo")
        with pytest.raises(BackendError, match="GGUF model not found"):
            resolve_model_file(cfg)

    def test_cached_weights_are_used_without_the_network(
        self, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # None in sys.modules makes `import huggingface_hub` fail, so any
        # attempt to reach the Hub would raise instead of downloading.
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        repo = scratch_dir / "unsloth" / "Qwen3.6-35B-A3B-GGUF"
        (repo / "Q8_0").mkdir(parents=True)
        for name in (
            "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
            "Qwen3.6-35B-A3B-IQ4_XS.gguf",
            "Q8_0/Qwen3.6-35B-A3B-Q8_0-00001-of-00002.gguf",
            "Q8_0/Qwen3.6-35B-A3B-Q8_0-00002-of-00002.gguf",
        ):
            (repo / name).write_bytes(b"GGUF")

        def resolve(quant: str) -> Path:
            spec = f"unsloth/Qwen3.6-35B-A3B-GGUF:{quant}"
            return resolve_model_file(
                BackendConfig(name="llamacpp_server", model=spec, models_dir=str(scratch_dir))
            )

        assert resolve("UD-IQ4_XS") == repo / "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
        assert resolve("Q8_0") == repo / "Q8_0" / "Qwen3.6-35B-A3B-Q8_0-00001-of-00002.gguf"
        with pytest.raises(BackendError, match="huggingface_hub"):
            resolve("Q4_K_M")  # not cached, so it would have to download

    def test_a_partly_downloaded_split_is_fetched_again(
        self, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # any download attempt raises
        folder = scratch_dir / "unsloth" / "Qwen3.6-35B-A3B-GGUF" / "Q8_0"
        folder.mkdir(parents=True)
        (folder / "Qwen3.6-35B-A3B-Q8_0-00001-of-00002.gguf").write_bytes(b"GGUF")
        cfg = BackendConfig(
            name="llamacpp_server",
            model="unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0",
            models_dir=str(scratch_dir),
        )
        with pytest.raises(BackendError, match="huggingface_hub"):
            resolve_model_file(cfg)  # part 2 is missing, so llama-server could not load it


@pytest.mark.parametrize(
    ("files", "complete"),
    [
        (["m-Q4_K_M.gguf"], True),
        (["m-Q8_0-00001-of-00002.gguf", "m-Q8_0-00002-of-00002.gguf"], True),
        (["Q8_0/M-Q8_0-00002-OF-00002.GGUF", "Q8_0/M-Q8_0-00001-OF-00002.GGUF"], True),
        (["m-Q8_0-00001-of-00002.gguf"], False),
        (["m-00001-of-00003.gguf", "m-00003-of-00003.gguf"], False),  # a gap
        (["m-00001-of-00002.gguf", "m-00002-of-00003.gguf"], False),  # disagreeing totals
        (["a.gguf", "b.gguf"], False),  # two models, not one
        ([], False),
    ],
)
def test_shards_complete(files: list[str], complete: bool) -> None:
    assert shards_complete(files) is complete
