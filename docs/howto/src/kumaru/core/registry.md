# `src/kumaru/core/registry.py`

> Generic lowercase name-to-factory registry used for pluggable components.

**Read this when:** registering runtime backends or designing another plugin point.

---

## What it does
It maps normalized string names to callables and can instantiate them later. Registration works directly or as a decorator, and errors include available names.

## Why it exists
Kumaru avoids eager imports of optional dependencies. A registry lets code select `echo` without importing torch-only backends and can support future tools or retrievers.

## Mental model
Names are stripped and lowercased. Factories are callables; `create()` simply calls the registered factory with your args.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `Registry` | `Registry(kind: str)` | Create a registry for a component kind. |
| `Registry.kind` | `kind(self) -> str` | Human-readable registry kind. |
| `Registry.register` | `register(self, name: str, factory: Callable[..., T] | None = None) -> Callable[..., T] | Callable[[Callable[..., T]], Callable[..., T]]` | Register a factory directly or as decorator. |
| `Registry.get` | `get(self, name: str) -> Callable[..., T]` | Return a factory. |
| `Registry.create` | `create(self, name: str, *args: object, **kwargs: object) -> T` | Instantiate via factory. |
| `Registry.names` | `names(self) -> list[str]` | Sorted registered names. |

## How to use it
```python
from kumaru.core.registry import Registry
r = Registry[str]("thing")
r.register("hello", lambda: "world")
print(r.create("hello"))
```

```python
from kumaru.core.registry import Registry
r = Registry[int]("number")
@r.register("one")
def make_one():
    return 1
print(r.get("one")())
```

```python
from kumaru.core.registry import Registry
r = Registry[object]("item")
print("missing" in r, r.names())
```

## How to extend it
Create a dedicated `Registry[YourType]` and publish a small registration API. Example: `tool_registry = Registry[Tool]("tool")`; then modules can call `@tool_registry.register("search")`.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Duplicate registration raises `RegistryError` | Same normalized name was registered twice. | Choose a unique name or guard registration. |
| Lookup says unknown | Module that registers the item was never imported. | Import the plugin module or add a lazy loader like `backends.__init__`. |
| Name case mismatch | Names are normalized, but display strings are not. | Use lowercase canonical names in docs/config. |

## Related files
- [Backend registry](../backends/__init__.md)
- [Errors](./errors.md)
