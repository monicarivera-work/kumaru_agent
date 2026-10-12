"""Protect configuration layering, validation, and environment parsing."""

import pytest

from kumaru.core.config import KumaruConfig, deep_merge, env_overrides, list_profiles, load_config
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
