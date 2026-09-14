"""A tiny name -> factory registry, used to make backends pluggable.

Why this exists
---------------
Without it, ``build_backend()`` would be a growing ``if/elif`` chain that
imports every backend eagerly - which would force torch to be installed even
for someone who only wants the HTTP backend. With a registry, each backend
module registers itself under a string name and is only imported when that
name is actually selected (see ``kumaru.backends.load_backend``).

The registry is generic (``Registry[T]``) so the same mechanism can later hold
tools, retrievers, or evaluators without duplication.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from kumaru.core.errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a lowercase string name to a factory for ``T``."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    @property
    def kind(self) -> str:
        return self._kind

    def register(
        self, name: str, factory: Callable[..., T] | None = None
    ) -> Callable[..., T] | Callable[[Callable[..., T]], Callable[..., T]]:
        """Register ``factory`` under ``name``.

        Usable directly (``registry.register("echo", EchoBackend)``) or as a
        decorator (``@registry.register("echo")``).
        """
        key = name.strip().lower()

        def _add(fn: Callable[..., T]) -> Callable[..., T]:
            if key in self._items:
                raise RegistryError(f"{self._kind} '{key}' is already registered")
            self._items[key] = fn
            return fn

        if factory is not None:
            return _add(factory)
        return _add

    def get(self, name: str) -> Callable[..., T]:
        """Return the factory registered under ``name``."""
        key = name.strip().lower()
        try:
            return self._items[key]
        except KeyError:
            raise RegistryError(
                f"unknown {self._kind} '{name}'",
                hint=f"available: {', '.join(self.names()) or '(none registered)'}",
            ) from None

    def create(self, name: str, *args: object, **kwargs: object) -> T:
        """Look up ``name`` and call its factory."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        """Registered names, sorted, for help text and error hints."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.strip().lower() in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)
