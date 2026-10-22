# `src\kumaru\core\registry.py`

> A typed name-to-class registry. Backends, tools and benchmarks are added by registering them, never by editing their callers.

**Read this when:**

- you add a backend, tool, or benchmark;
- you debug `Unknown backend: '...'`;
- you package a third-party Kumaru extension;
- `kumaru --help` suddenly got slow.

---

## What it does

This file defines the generic `Registry` class and three global registries.

| Registry | Kind (in error text) | Entry-point group | Built-in entries (aliases in parentheses) |
|---|---|---|---|
| `BACKENDS` | `backend` | `kumaru.backends` | `echo`, `transformers` (`hf`), `llamacpp` (`gguf`), `openai_compat` (`ollama`, `vllm`, `remote`), `llamacpp_server` (`llama-server`, `llama_server`) — all lazy |
| `TOOLS` | `tool` | `kumaru.tools` | `current_time`, registered by decorator when `kumaru.agent.tools` is imported |
| `BENCHMARKS` | `benchmark` | `kumaru.benchmarks` | `gsm8k`, `gsm8k_train` (`gsm8k-train`), `mmlu_pro` (`mmlu-pro`), `ifeval`, `humaneval`, `multipl_e_<lang>` (`multipl-e-<lang>`) for each of the eight `MULTIPL_E_LANGUAGES`, `golden` — all lazy |

v0.2 adds the `llamacpp_server` backend and the whole `BENCHMARKS` registry, which `kumaru bench` uses.

It also defines **benchmark suites**: names that stand for several benchmarks. `BENCHMARK_SUITES["multipl_e"]` is the eight MultiPL-E languages, and `SUITE_ALIASES` lets `multipl-e` mean the same. A suite is not a registry entry (`"multipl_e" in BENCHMARKS` is False): the bench runner expands it with `suite_members()` before resolving each member, and skips a member whose compiler or runtime is missing ([`runner.md`](../eval/benchmarks/runner.md)).

Callers resolve a component by the name in config or on the command line. `BACKENDS.create(cfg.backend.name, cfg.backend)` appears in `src\kumaru\server\app.py`, `src\kumaru\cli.py`, `src\kumaru\eval\harness.py`, and `src\kumaru\eval\benchmarks\runner.py`.

## Why it exists

Kumaru has to absorb new inference engines, tools and evaluations as **additive** changes. Without a registry:

- callers grow `if backend == ...` chains;
- every new component edits a central file;
- every edit risks breaking what already works.

Lazy registration solves a second problem. Importing the transformers backend pulls in torch, which costs seconds and hundreds of MB, and listing benchmarks would import `datasets`. With `"module:Class"` strings, these happen only when you actually select the component:

- `kumaru --help` and `kumaru bench list` stay fast;
- `kumaru serve -b echo` never imports torch.

## Mental model

A registry holds three structures:

- `_items` — name to class, filled by `register()` decorators and by lazy targets once they are resolved;
- `_lazy` — name to `"module.path:ClassName"`, which is not imported yet;
- `_aliases` — names hidden from listings.

`get(name)` looks in three places, in order:

1. **`_items`** — return the class.
2. **`_lazy`** — import the module, cache the class in `_items`, and return it. The import runs the module's own `@BACKENDS.register(...)` decorator, which registers the same class under the primary name and its aliases.
3. **Entry points** — run `load_entry_points()` once per registry, then retry. If the name is still missing, raise `RegistryError` with the available primary names in the hint.

Three invariants follow:

- **Plugins are discovered lazily.** Entry points are scanned only when a lookup misses steps 1 and 2. Until then, `names()`, `kumaru info`, `/v1/info.available_backends` and the UI's backend list **do not include installed plugins**. A plugin is still selectable by name, because the miss triggers the scan.
- **`_items` wins.** Whatever lands in `_items` first under a name is what `get` returns. That includes a decorator applied to a name that is still only lazy, and an entry point named like an unresolved built-in (`load_entry_points` uses `_items.setdefault`, which does not check `_lazy`). See the shadowing row in Failure modes.
- **Aliases are hidden.** They resolve, but only primary names appear in `names()`, `iter()` and `len()`. The UI therefore shows five backends rather than eleven names.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `Registry` | `Registry[T](kind: str, *, entry_point_group: str \| None = None)` | Generic registry; `kind` is the noun used in error messages. |
| `Registry.register` | `register(self, name: str, *, aliases: tuple[str, ...] = ()) -> Callable[[type[T]], type[T]]` | Class decorator. Raises `RegistryError("{kind} '{key}' is already registered")` if the name or an alias is already in `_items`. |
| `Registry.register_lazy` | `register_lazy(self, name: str, target: str, *, aliases: tuple[str, ...] = ()) -> None` | Register `"package.module:ClassName"` without importing it. No duplicate check: it overwrites earlier lazy targets. |
| `Registry.get` | `get(self, name: str) -> type[T]` | Resolve a class: `_items`, then `_lazy`, then entry points; otherwise `RegistryError("Unknown {kind}: '{name}'.", hint="Available {kind}s: ...")`. |
| `Registry.create` | `create(self, name: str, *args: Any, **kwargs: Any) -> T` | `get(name)(*args, **kwargs)`. |
| `Registry.names` | `names(self) -> list[str]` | Sorted primary names from `_items` and `_lazy`, without aliases. Does not scan entry points. |
| `Registry.all_names` | `all_names(self) -> list[str]` | Sorted names including aliases. |
| `Registry.__contains__` | `name in registry -> bool` | True for any registered or lazy name, alias included. No import and no entry-point scan. |
| `Registry.__iter__` / `__len__` | `iter(registry)`, `len(registry)` | Iterate over and count `names()`. |
| `Registry.load_entry_points` | `load_entry_points(self) -> None` | Scan the entry-point group once. Plugin load failures are logged as warnings and never raised. |
| `BACKENDS` | `Registry[Any]` | `Registry("backend", entry_point_group="kumaru.backends")`. |
| `TOOLS` | `Registry[Any]` | `Registry("tool", entry_point_group="kumaru.tools")`. |
| `BENCHMARKS` | `Registry[Any]` | `Registry("benchmark", entry_point_group="kumaru.benchmarks")`. |
| `MULTIPL_E_LANGUAGES` | `dict[str, str]` | MultiPL-E language code to class name in `kumaru.eval.benchmarks.multipl_e`: `py`, `js`, `ts`, `cs`, `java`, `cpp`, `go`, `rs`. Each registers `multipl_e_<code>`, with the alias `multipl-e-<code>`. |
| `BENCHMARK_SUITES` | `dict[str, tuple[str, ...]]` | Suite name to member benchmarks, in run order: `{"multipl_e": ("multipl_e_py", ..., "multipl_e_rs")}`. `kumaru bench list` shows a row for each. |
| `SUITE_ALIASES` | `dict[str, str]` | Alternative suite names: `{"multipl-e": "multipl_e"}`. |
| `suite_members` | `suite_members(name: str) -> tuple[str, ...]` | The members of a suite (aliases accepted), or `()` when `name` is not a suite. |

## How to use it

### Create a backend from config

```python
from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS

cfg = load_config("test", use_env=False)
backend = BACKENDS.create(cfg.backend.name, cfg.backend)    # echo
await backend.load()
```

### Aliases resolve to the same class

```python
from kumaru.core.registry import BACKENDS

assert BACKENDS.get("llama-server") is BACKENDS.get("llamacpp_server")
print(BACKENDS.names())
# ['echo', 'llamacpp', 'llamacpp_server', 'openai_compat', 'transformers']
```

### List benchmarks without importing them

```python
from kumaru.core.registry import BENCHMARKS, suite_members

print(BENCHMARKS.names())
# ['golden', 'gsm8k', 'gsm8k_train', 'humaneval', 'ifeval', 'mmlu_pro', 'multipl_e_cpp', 'multipl_e_cs',
#  'multipl_e_go', 'multipl_e_java', 'multipl_e_js', 'multipl_e_py', 'multipl_e_rs', 'multipl_e_ts']
print("mmlu-pro" in BENCHMARKS)        # True: alias, no import
print(suite_members("multipl-e"))
# ('multipl_e_py', 'multipl_e_js', 'multipl_e_ts', 'multipl_e_cs', 'multipl_e_java', 'multipl_e_cpp', 'multipl_e_go', 'multipl_e_rs')
```

`kumaru bench list` goes further: it calls `get()` on each entry to read the class attributes `description`, `source` and `needs_code_execution`. That imports the benchmark modules, but it still downloads no data.

### A private registry, for tests or a sub-system

```python
from kumaru.core.registry import Registry

class DemoTool: ...

tools: Registry[DemoTool] = Registry("tool")
tools.register("demo", aliases=("d",))(DemoTool)
assert isinstance(tools.create("d"), DemoTool)
assert tools.names() == ["demo"]
```

## How to extend it

### A built-in component

There are two parts, and you need both.

1. **The class registers itself with a decorator.** This matches every built-in backend:

   ```python
   # src\kumaru\backends\my_backend.py
   from kumaru.backends.base import Backend
   from kumaru.core.registry import BACKENDS

   @BACKENDS.register("my_backend", aliases=("mine",))
   class MyBackend(Backend):
       ...
   ```

2. **Add a lazy pointer** at the bottom of `src\kumaru\core\registry.py`, so the name is listed and resolvable without importing the module:

   ```python
   BACKENDS.register_lazy("my_backend", "kumaru.backends.my_backend:MyBackend", aliases=("mine",))
   ```

The two must agree on the name and aliases:

- A lazy name without the decorator still resolves. The class is cached only under the name you asked for.
- A decorator without the lazy line registers only if something imports the module first. Nothing does for backends, so the name is unknown.

Keep the module's heavy imports such as torch or llama_cpp inside `load()`, and raise `DependencyMissingError` when they are missing. Otherwise `get()` fails with a raw `ModuleNotFoundError` instead of an install hint. The backends do this: `llamacpp_backend.py` raises it with `extra="gguf"` and `transformers_backend.py` with `extra="torch"`.

Benchmarks follow the same pattern with `BENCHMARKS.register_lazy(...)`. Subclass `Benchmark` from `src\kumaru\eval\benchmarks\base.py`; `create(name)` instantiates it with no arguments.

A new MultiPL-E language is one entry in `MULTIPL_E_LANGUAGES` plus its class; the loop registers it and the `multipl_e` suite picks it up (the full steps are in [`multipl_e.md`](../eval/benchmarks/multipl_e.md#how-to-extend-it)). A new suite is one `BENCHMARK_SUITES` entry; its members must be registered benchmarks.

### A third-party plugin (no change to this repository)

```toml
# the plugin package's pyproject.toml
[project.entry-points."kumaru.backends"]
mamba = "kumaru_mamba:MambaBackend"

[project.entry-points."kumaru.tools"]
web_search = "kumaru_search:WebSearchTool"

[project.entry-points."kumaru.benchmarks"]
my_eval = "kumaru_my_eval:MyEval"
```

After `pip install`, `kumaru chat -b mamba` works: the lookup misses, scans the entry points, and finds the plugin. To make plugins appear in `names()` before first use, call `BACKENDS.load_entry_points()` at startup. v0.2 does not call it anywhere except inside `get()`.

Do not give a plugin entry point the name of a built-in.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `error: Unknown backend: 'x'.` with `hint: Available backends: echo, llamacpp, llamacpp_server, openai_compat, transformers` | Typo, plugin not installed, or the component is missing its `register_lazy` line. | Use a listed name or alias, install the plugin, or add the lazy registration. |
| HTTP 503 `Failed to load new backend: Unknown backend: ...` | The same `RegistryError`, raised while swapping the backend through `POST /v1/backend`. | As above. |
| An installed plugin is missing from `kumaru info`, `/v1/info` or the UI dropdown | `names()` does not scan entry points; the scan happens only on a lookup miss. | Select it by name (`-b mamba`), or call `load_entry_points()` at startup. |
| `{kind} 'x' is already registered` at import time | Two decorators use the same name or alias. This also happens if an entry point already claimed a built-in's name before its module was imported. | Rename one of them. Never reuse built-in names in plugins. |
| A built-in name returns someone else's class | That class reached `_items` first, either through an entry point with the same name (once a lookup miss triggered the scan) or through a decorator applied before the lazy import. | Rename the plugin or the decorator; `_items` always wins. |
| `ModuleNotFoundError` from `get()` | A `register_lazy` target names a module that does not exist (typo), or the module imports a missing dependency at top level. | Fix the target string, and move heavy imports into `load()` behind `DependencyMissingError`. |
| `names()` does not show an alias | By design. | `all_names()`. |
| `Unknown benchmark: 'multipl_e'.` from your own `BENCHMARKS.get` or `create` | A suite is not a registry entry; only the bench runner expands it. | Loop over `suite_members(name)`, or pass the suite to `kumaru bench run`. |
| `kumaru --help` or `kumaru bench list` is slow | A heavy module is imported at registration time (decorator-only registration that something imports eagerly), or a module imports torch or `datasets` at top level. | Register lazily, and defer heavy imports into the method that needs them. |
| `kumaru info` shows `tools: (none)` in custom code | `TOOLS` is filled only when `kumaru.agent.tools` is imported. | Import `kumaru.agent.tools`, as `kumaru info` does. |
| A plugin silently fails to appear | `ep.load()` raised; it is logged as a warning (`failed to load plugin backend 'x': ...`) and skipped. | Run with `$env:KUMARU_LOG_LEVEL = "DEBUG"` and read the warning. |

## Known limitations

- **Name collisions are settled by registration order, not by priority.** No error is raised except in the first case:
  - A second `register()` for a name already in `_items` raises.
  - `load_entry_points()` skips an entry point whose name is taken (`setdefault`) after already importing it, and still logs it at DEBUG as `loaded plugin backend 'x'`. Of two plugins with the same name, the first that `importlib.metadata` lists wins, which depends on the installation. A plugin named like an unresolved built-in shadows it once a lookup miss has triggered the scan.
  - `register_lazy()` overwrites, so the last target registered for a name wins.

  Workaround: give plugins unique names, and check that `BACKENDS.get(name)` returns the class you expect.

## Related files

- `src\kumaru\core\errors.py` — `RegistryError`, raised for duplicate and unknown names.
- `src\kumaru\backends\base.py` — the `Backend` contract that registered backends implement.
- `src\kumaru\backends\llamacpp_server.py` — the v0.2 backend behind `llamacpp_server`.
- `src\kumaru\agent\tools.py` — `@TOOLS.register("current_time")`; `ToolBelt` creates enabled tools by name.
- `src\kumaru\eval\benchmarks\runner.py` — `resolve_benchmark()` (registry name or a `.jsonl` golden-set path), `list_benchmarks()` (one row per benchmark and per suite) and the suite expansion in `BenchRunner._resolve`.
- `src\kumaru\eval\benchmarks\multipl_e.py` — the classes `MULTIPL_E_LANGUAGES` names ([`multipl_e.md`](../eval/benchmarks/multipl_e.md)).
- `src\kumaru\cli.py` — `kumaru info` lists all three registries and the benchmark suites; `chat`, `eval` and `bench` create from them.
- `src\kumaru\server\routes_chat.py` — `/v1/info` reports `BACKENDS.names()` and `TOOLS.names()`.
