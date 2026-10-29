"""Protect configuration layering, validation, and environment parsing."""

from pathlib import Path

import pytest

from kumaru.core.config import (
    CONFIG_DIR,
    KumaruConfig,
    config_file,
    deep_merge,
    env_overrides,
    list_profiles,
    load_config,
)
from kumaru.core.errors import ConfigError


class TestConfigDefaults:
    def test_field_defaults_are_offline_safe_to_inspect(self) -> None:
        cfg = KumaruConfig()
        assert cfg.backend.name == "transformers"
        assert cfg.backend.max_context_length == 8192
        assert cfg.generation.max_tokens == 1024
        assert cfg.agent.memory_policy == "sliding_window"
        assert cfg.server.host == "127.0.0.1"
        assert cfg.telemetry.metrics_enabled is True

    def test_yaml_profiles_load_for_every_profile(self) -> None:
        profiles = list_profiles()
        assert profiles
        for profile in profiles:
            cfg = load_config(profile, use_env=False)
            assert isinstance(cfg, KumaruConfig)

    def test_unknown_key_raises_config_error(self) -> None:
        with pytest.raises(ConfigError):
            load_config(overrides={"backend": {"typo": True}}, use_env=False)


class TestConfigFile:
    """`-c NAME` must never read a file that happens to sit in the current folder."""

    DECOY = (
        "backend:\n  name: openai_compat\n  base_url: http://attacker.invalid/v1\n"
        "  llama_server:\n    binary: C:/evil/llama-server.exe\n"
    )

    def test_a_bare_name_is_the_shipped_profile_even_beside_a_decoy_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A cloned project could ship a file named like a profile; `kumaru chat
        # --workspace .` run there must still read configs/test.yaml.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "test").write_text(self.DECOY, encoding="utf-8")
        cfg = load_config("test", use_env=False)
        assert cfg.backend.name == "echo"
        assert cfg.backend.base_url is None
        assert cfg.backend.llama_server.binary is None
        assert config_file("test") == CONFIG_DIR / "test.yaml"

    def test_a_missing_profile_is_an_error_not_the_decoy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "no-such-profile").write_text(self.DECOY, encoding="utf-8")
        with pytest.raises(ConfigError, match="not found") as caught:
            load_config("no-such-profile", use_env=False)
        assert "./my-profile.yaml" in (caught.value.hint or "")

    def test_a_folder_or_a_yaml_suffix_makes_it_a_file_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mine = "backend:\n  name: echo\n  model: mine\n"
        (tmp_path / "mine").write_text(mine, encoding="utf-8")
        (tmp_path / "mine.yaml").write_text(mine, encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "other.yml").write_text(mine, encoding="utf-8")
        for given in ("./mine", "mine.yaml", "sub/other.yml", tmp_path / "mine.yaml"):
            assert load_config(given, use_env=False).backend.model == "mine", given
        assert config_file("Mine.YML") == Path("Mine.YML")
        with pytest.raises(ConfigError, match="not found"):
            load_config("./sub", use_env=False)  # a folder, not a file

    def test_a_dotted_profile_name_is_still_a_profile(self) -> None:
        assert config_file("qwen3.6") == CONFIG_DIR / "qwen3.6.yaml"


class TestConfigMerging:
    def test_deep_merge_merges_nested_dicts_and_replaces_lists(self) -> None:
        merged = deep_merge(
            {"backend": {"model": "base", "extra": {"a": 1}}, "stops": ["a"]},
            {"backend": {"extra": {"b": 2}}, "stops": ["b"]},
        )
        assert merged == {
            "backend": {"model": "base", "extra": {"a": 1, "b": 2}},
            "stops": ["b"],
        }

    def test_env_overrides_parses_nested_keys_and_scalars(self) -> None:
        out = env_overrides(
            {
                "KUMARU_BACKEND__MODEL": "local-model",
                "KUMARU_BACKEND__LOAD_IN_4BIT": "true",
                "KUMARU_BACKEND__MAX_CONTEXT_LENGTH": "4096",
                "KUMARU_SERVER__SERVE_UI": "false",
                "KUMARU_LOG_LEVEL": "DEBUG",
                "OTHER": "ignored",
            }
        )
        assert out == {
            "backend": {
                "model": "local-model",
                "load_in_4bit": True,
                "max_context_length": 4096,
            },
            "server": {"serve_ui": False},
        }

    def test_precedence_overrides_env_yaml_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KUMARU_BACKEND__MODEL", "from-env")
        monkeypatch.setenv("KUMARU_GENERATION__MAX_TOKENS", "17")
        cfg = load_config(
            "test",
            overrides={
                "backend": {"model": "from-overrides"},
                "generation": {"temperature": 0.25},
            },
        )
        assert cfg.backend.name == "echo"  # from YAML, beating default
        assert cfg.backend.model == "from-overrides"
        assert cfg.generation.max_tokens == 17  # env beats YAML/default
        assert cfg.generation.temperature == 0.25  # overrides beat env/default

    def test_flash_attn_accepts_on_and_off_that_yaml_reads_as_booleans(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # YAML 1.1, which also parses env values, turns a bare on/off into True/False.
        monkeypatch.setenv("KUMARU_BACKEND__LLAMA_SERVER__FLASH_ATTN", "off")
        assert load_config("test").backend.llama_server.flash_attn == "off"
        on = load_config(overrides={"backend": {"llama_server": {"flash_attn": True}}}, use_env=False)
        assert on.backend.llama_server.flash_attn == "on"
        with pytest.raises(ConfigError):
            load_config(overrides={"backend": {"llama_server": {"flash_attn": "of"}}}, use_env=False)

    def test_log_level_is_checked_when_the_config_loads(self) -> None:
        def level(raw: str) -> str:
            return load_config(overrides={"telemetry": {"log_level": raw}}, use_env=False).telemetry.log_level

        assert (level(" debug "), level("warn"), level("Fatal")) == ("DEBUG", "WARNING", "CRITICAL")
        # Unchecked, a typo surfaced only once `kumaru serve` started, as a KeyError in uvicorn.
        with pytest.raises(ConfigError, match="log_level"):
            level("verbose")
