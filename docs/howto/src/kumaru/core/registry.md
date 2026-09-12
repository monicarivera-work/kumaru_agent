# `src\kumaru\core\registry.py`

> Typed name-to-class registry used to add backends and tools without hard-coding them into callers.

**Read this when:** you add a backend, register a tool, debug `Unknown backend`, or package a third-party Kumaru extension.

---

## What it does
This file defines the generic `Registry` class and the global `BACKENDS` and `TOOLS` registries. Components register themselves by name, optionally with aliases, and callers resolve them by name from config. Expensive built-in backends are registered lazily so selecting one backend does not import every backend dependency.

## Why it exists
Kumaru needs to absorb new inference techniques and tools as additive changes. Without a registry, callers would contain `if backend == ...` chains, every new component would require editing central files, and heavy imports such as torch would slow down unrelated commands.

## Mental model
Components announce themselves; the rest of the system asks for a named component. Primary names appear in user-facing lists, aliases remain resolvable but hidden. Entry points let installed third-party packages contribute components at runtime.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `Registry` | `Registry[T](kind: str, *, entry_point_group: str | None = None)` | Generic component registry. |
| `Registry.register` | `register(self, name: str, *, aliases: tuple[str, ...] = ()) -> Callable[[type[T]], type[T]]` | Decorator for registering an already imported class. |
| `Registry.register_lazy` | `register_lazy(self, name: str, target: str, *, aliases: tuple[str, ...] = ()) -> None` | Register `"module.path:ClassName"` without importing it. |
| `Registry.get` | `get(self, name: str) -> type[T]` | Resolve a class by name, loading lazy or entry-point registrations if needed. |
| `Registry.create` | `create(self, name: str, *args: Any, **kwargs: Any) -> T` | Resolve and instantiate a component. |
| `Registry.names` | `names(self) -> list[str]` | Return primary names only, sorted for UI/CLI display. |
| `Registry.all_names` | `all_names(self) -> list[str]` | Return every resolvable name, including aliases. |
| `Registry.load_entry_points` | `load_entry_points(self) -> None` | Discover installed plugin components from the configured entry-point group. |
| `BACKENDS` | `Registry[Any]` | Global backend registry using entry-point group `kumaru.backends`. |
| `TOOLS` | `Registry[Any]` | Global tool registry using entry-point group `kumaru.tools`. |

Built-in lazy backend names are `echo`, `transformers` (`hf`), `llamacpp` (`gguf`), and `openai_compat` (`ollama`, `vllm`, `remote`).

## How to use it

Create a backend from config:

```python
from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS

cfg = load_config(use_env=False, overrides={"backend": {"name": "echo"}})
backend = BACKENDS.create(cfg.backend.name, cfg.backend)
```

List user-facing backend names:

```python
from kumaru.core.registry import BACKENDS

print(BACKENDS.names())
```

Resolve aliases when accepting config:

```python
from kumaru.core.registry import BACKENDS

backend_cls = BACKENDS.get("hf")
```

Register a lightweight in-process component:

```python
from kumaru.core.registry import Registry

class DemoTool:
    pass

tools = Registry[DemoTool]("tool")
tools.register("demo")(DemoTool)

assert tools.create("demo").__class__ is DemoTool
```

## How to extend it

For built-in components with heavy imports, use lazy registration:

```python
from kumaru.core.registry import BACKENDS

BACKENDS.register_lazy(
    "my_backend",
    "kumaru.backends.my_backend:MyBackend",
    aliases=("mine",),
)
```

Then implement the class in the target module. For a backend, subclass `kumaru.backends.base.Backend` and accept `BackendConfig` in the constructor through the base class.

Third-party packages should expose entry points instead of editing this repository. A package can declare:

```toml
[project.entry-points."kumaru.backends"]
my_backend = "kumaru_my_backend:MyBackend"

[project.entry-points."kumaru.tools"]
my_tool = "kumaru_my_tool:MyTool"
```

`load_entry_points()` logs plugin load failures and continues so one broken plugin does not prevent startup.

## Failure modes
| Symptom | Likely cause | Fix |
|---|---|---|
| `Unknown backend: 'x'` | Name is not registered, plugin is not installed, or config uses the wrong name. | Check `BACKENDS.names()`, install the plugin, or add registration. |
| Duplicate registration error | A primary name or alias already exists in `_items`. | Choose a unique name or remove the duplicate registration. |
| `names()` does not show an alias | Aliases are intentionally hidden from user-facing lists. | Use `all_names()` when you need aliases too. |
| `kumaru --help` imports torch | A heavy backend was registered by importing the class directly. | Replace direct import registration with `register_lazy()`. |
| Plugin backend does not appear | Entry-point group or package metadata is wrong, or loading failed. | Use group `kumaru.backends` and inspect logs for plugin warnings. |

## Related files
- `src\kumaru\core\errors.py` — provides `RegistryError` for duplicate and unknown names.
- `src\kumaru\core\logging.py` — logs entry-point scan and plugin load failures.
- `src\kumaru\cli.py` — uses `BACKENDS.names()` in `info()` and `BACKENDS.create()` in chat mode.
- `src\kumaru\backends\base.py` — defines the backend contract registered classes implement.
- `src\kumaru\agent\chat.py` — uses tools indirectly through `ToolBelt`, which depends on registered tool names.
