"""Protect component registry lookup, aliases, and lazy import behavior."""

import pytest

from kumaru.core.errors import RegistryError
from kumaru.core.registry import Registry


class Sample:
    def __init__(self, value: int = 0) -> None:
        self.value = value


class TestRegistry:
    def test_register_get_and_create(self) -> None:
        registry: Registry[Sample] = Registry("sample")
        registry.register("primary")(Sample)
        assert registry.get("primary") is Sample
        assert isinstance(registry.create("primary", 7), Sample)
        assert registry.create("primary", 7).value == 7

    def test_aliases_resolve_but_are_hidden_from_names(self) -> None:
        registry: Registry[Sample] = Registry("sample")
        registry.register("primary", aliases=("alias",))(Sample)
        assert registry.get("alias") is Sample
        assert registry.names() == ["primary"]
        assert registry.all_names() == ["alias", "primary"]

    def test_duplicate_registration_raises_registry_error(self) -> None:
        registry: Registry[Sample] = Registry("sample")
        registry.register("primary")(Sample)
        with pytest.raises(RegistryError):
            registry.register("primary")(Sample)

    def test_unknown_name_raises_registry_error_with_hint(self) -> None:
        registry: Registry[Sample] = Registry("sample")
        registry.register("primary")(Sample)
        with pytest.raises(RegistryError) as excinfo:
            registry.get("missing")
        assert "Unknown sample" in str(excinfo.value)
        assert "primary" in (excinfo.value.hint or "")

    def test_register_lazy_resolves_real_dotted_target(self) -> None:
        registry = Registry("backend")
        registry.register_lazy("echo", "kumaru.backends.echo:EchoBackend", aliases=("e",))
        cls = registry.get("e")
        assert cls.__name__ == "EchoBackend"
        assert registry.names() == ["echo"]
        assert registry.all_names() == ["e", "echo"]

    def test_contains_and_len_count_primary_names(self) -> None:
        registry: Registry[Sample] = Registry("sample")
        registry.register("primary", aliases=("alias",))(Sample)
        assert "primary" in registry
        assert "alias" in registry
        assert "missing" not in registry
        assert len(registry) == 1
