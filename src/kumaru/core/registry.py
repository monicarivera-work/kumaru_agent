"""A typed component registry — Kumaru's extensibility mechanism.

The problem it solves
---------------------
A system that must "absorb new techniques" cannot hard-code its parts. If
`server.py` does `if backend == "transformers": ... elif backend == "vllm": ...`
then every new idea requires editing core files, and every edit risks breaking
what already works.

The registry inverts that. Components *announce themselves*::

    @BACKENDS.register("my_new_backend")
    class MyNewBackend(Backend): ...

and the rest of the system only ever asks for them *by name*, resolved from
config at runtime. Adding a capability becomes an additive change — a new file
plus one decorator — which is the property that lets a codebase absorb research
for years without rotting.

Third-party extension
---------------------
`load_entry_points()` discovers components installed by *other* pip packages
that declare the `kumaru.backends` / `kumaru.tools` / `kumaru.benchmarks`
entry-point group. That means
someone can `pip install kumaru-mamba` and the backend shows up in your UI's
dropdown with no change to this repository.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

from kumaru.core.errors import RegistryError
from kumaru.core.logging import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


class Registry(Generic[T]):
    """A name -> class mapping with decorator registration and lazy imports.

    Parameters
    ----------
    kind:
        Human-readable noun used in error messages ("backend", "tool", ...).
    entry_point_group:
        Optional setuptools entry-point group scanned by `load_entry_points()`
        to discover third-party plugins.
    """

    def __init__(self, kind: str, *, entry_point_group: str | None = None) -> None:
        self._kind = kind
        self._entry_point_group = entry_point_group
        self._items: dict[str, type[T]] = {}
        # Maps name -> "module.path:ClassName" for components whose imports are
        # expensive (torch takes seconds). Resolved only when actually used, so
        # `kumaru serve --backend echo` never pays for importing torch.
        self._lazy: dict[str, str] = {}
        # Aliases are resolvable but hidden from user-facing listings, so the UI
        # shows four backends rather than nine names for the same four things.
        self._aliases: set[str] = set()
        self._entry_points_loaded = False

    # -- registration -------------------------------------------------------

    def register(self, name: str, *, aliases: tuple[str, ...] = ()) -> Callable[[type[T]], type[T]]:
        """Decorator that registers a class under `name` (and any aliases)."""

        def decorator(cls: type[T]) -> type[T]:
            for key in (name, *aliases):
                if key in self._items:
                    raise RegistryError(f"{self._kind} '{key}' is already registered")
                self._items[key] = cls
            self._aliases.update(aliases)
            return cls

        return decorator

    def register_lazy(self, name: str, target: str, *, aliases: tuple[str, ...] = ()) -> None:
        """Register `"package.module:ClassName"` without importing it yet."""
        for key in (name, *aliases):
            self._lazy[key] = target
        self._aliases.update(aliases)

    # -- lookup -------------------------------------------------------------

    def get(self, name: str) -> type[T]:
        """Resolve a registered class by name, importing lazily if needed."""
        if name in self._items:
            return self._items[name]

        if name in self._lazy:
            module_path, _, attr = self._lazy[name].partition(":")
            module = importlib.import_module(module_path)
            cls = getattr(module, attr)
            self._items[name] = cls
            return cls

        if not self._entry_points_loaded:
            self.load_entry_points()
            if name in self._items or name in self._lazy:
                return self.get(name)

        raise RegistryError(
            f"Unknown {self._kind}: '{name}'.",
            hint=f"Available {self._kind}s: {', '.join(self.names()) or '(none)'}",
        )

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Resolve by name and instantiate in one step."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        """Primary names only — what a human should see in a list or dropdown."""
        return sorted((set(self._items) | set(self._lazy)) - self._aliases)

    def all_names(self) -> list[str]:
        """Every resolvable name, aliases included."""
        return sorted(set(self._items) | set(self._lazy))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and (name in self._items or name in self._lazy)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())
    # -- plugins ------------------------------------------------------------

    def load_entry_points(self) -> None:
        """Discover components contributed by other installed packages.

        Failures are logged, never raised: one broken third-party plugin must
        not prevent the whole application from starting.
        """
        self._entry_points_loaded = True
        if not self._entry_point_group:
            return
        try:
            entries = importlib_metadata.entry_points(group=self._entry_point_group)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.debug("entry-point scan failed for %s: %s", self._entry_point_group, exc)
            return

        for ep in entries:
            try:
                self._items.setdefault(ep.name, ep.load())
                logger.debug("loaded plugin %s '%s'", self._kind, ep.name)
            except Exception as exc:
                logger.warning("failed to load plugin %s '%s': %s", self._kind, ep.name, exc)


# ---------------------------------------------------------------------------
# The global registries.
#
# Lazy registration keeps CLI startup fast: importing `kumaru.backends.
# transformers_backend` pulls in torch (multiple seconds, hundreds of MB), and
# we only want to pay that when the transformers backend is actually selected.
# ---------------------------------------------------------------------------

BACKENDS: Registry[Any] = Registry("backend", entry_point_group="kumaru.backends")
TOOLS: Registry[Any] = Registry("tool", entry_point_group="kumaru.tools")

BACKENDS.register_lazy("echo", "kumaru.backends.echo:EchoBackend")
BACKENDS.register_lazy(
    "transformers", "kumaru.backends.transformers_backend:TransformersBackend", aliases=("hf",)
)
BACKENDS.register_lazy(
    "llamacpp", "kumaru.backends.llamacpp_backend:LlamaCppBackend", aliases=("gguf",)
)
BACKENDS.register_lazy(
    "openai_compat",
    "kumaru.backends.openai_compat:OpenAICompatBackend",
    aliases=("ollama", "vllm", "remote"),
)
BACKENDS.register_lazy(
    "llamacpp_server",
    "kumaru.backends.llamacpp_server:LlamaServerBackend",
    aliases=("llama-server", "llama_server"),
)

# Benchmarks for `kumaru bench`. Lazy for the same reason: listing them must not
# import `datasets` (seconds) or download anything.
BENCHMARKS: Registry[Any] = Registry("benchmark", entry_point_group="kumaru.benchmarks")

BENCHMARKS.register_lazy("gsm8k", "kumaru.eval.benchmarks.gsm8k:GSM8K")
BENCHMARKS.register_lazy(
    "gsm8k_train", "kumaru.eval.benchmarks.gsm8k:GSM8KTrain", aliases=("gsm8k-train",)
)
BENCHMARKS.register_lazy("mmlu_pro", "kumaru.eval.benchmarks.mmlu_pro:MMLUPro", aliases=("mmlu-pro",))
BENCHMARKS.register_lazy("ifeval", "kumaru.eval.benchmarks.ifeval:IFEval")
BENCHMARKS.register_lazy("humaneval", "kumaru.eval.benchmarks.humaneval:HumanEval")
# HumanEval in eight languages (MultiPL-E). One benchmark per language, so each
# gets its own score, confidence interval and paired comparison.
MULTIPL_E_LANGUAGES = {
    "py": "MultiPLEPython", "js": "MultiPLEJavaScript", "ts": "MultiPLETypeScript",
    "cs": "MultiPLECSharp", "java": "MultiPLEJava", "cpp": "MultiPLECpp", "go": "MultiPLEGo",
    "rs": "MultiPLERust",
}  # fmt: skip
for _lang, _cls in MULTIPL_E_LANGUAGES.items():
    BENCHMARKS.register_lazy(
        f"multipl_e_{_lang}", f"kumaru.eval.benchmarks.multipl_e:{_cls}", aliases=(f"multipl-e-{_lang}",)
    )
BENCHMARKS.register_lazy("golden", "kumaru.eval.benchmarks.golden:GoldenSet")

# Names that stand for several benchmarks. A suite skips members whose compiler
# or runtime is missing, and says so; naming a member directly fails instead.
BENCHMARK_SUITES: dict[str, tuple[str, ...]] = {
    "multipl_e": tuple(f"multipl_e_{lang}" for lang in MULTIPL_E_LANGUAGES),
}
SUITE_ALIASES: dict[str, str] = {"multipl-e": "multipl_e"}


def suite_members(name: str) -> tuple[str, ...]:
    """The benchmarks a suite name stands for, or () if `name` is not a suite."""
    return BENCHMARK_SUITES.get(SUITE_ALIASES.get(name, name), ())
