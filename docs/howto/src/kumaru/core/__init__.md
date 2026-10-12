# `src\kumaru\core\__init__.py`

> Boundary marker for Kumaru's dependency-inward core layer.

**Read this when:** you are adding code under `kumaru.core`, reviewing imports for cycles, or deciding whether a shared contract belongs in core.

---

## What it does
This module contains the architectural rule for the core package. `kumaru.core` defines contracts, configuration, errors, registries, logging, and data types that other layers depend on. It deliberately does not import from the backends, agent, server, train or eval packages.

## Why it exists
Kumaru needs a stable centre that feature layers can share without forming import cycles. Without this boundary, adding a backend or tool could require editing core files or importing heavy runtime modules during simple operations such as config inspection.

## Mental model
Dependencies point inward. Core may use standard library and low-level dependencies such as Pydantic or YAML, but product layers depend on core, not the other way round. If a core module needs knowledge of a concrete backend, the dependency has probably been inverted incorrectly.

The one deliberate exception is `registry.py`. It names the built-in backends, and since v0.2 the `kumaru bench` benchmarks (`kumaru.eval.benchmarks.*`), but only as lazy `"module:Class"` strings. Nothing is imported until the first lookup by that name, so the import graph still points inward. The module docstring in `core\__init__.py` lists only backends, agent, server and train; the same rule applies to `kumaru.eval`.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| _(none)_ | N/A | This package initializer exports no runtime symbols. Import concrete APIs from the modules inside `kumaru.core`. |

## How to use it

Import core contracts directly from their owning module:

```python
from kumaru.core.config import load_config
from kumaru.core.types import Message

cfg = load_config(use_env=False)
message = Message.user("Hello")
```

Keep feature code depending on core:

```python
from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS

config = load_config("test")
backend = BACKENDS.create(config.backend.name, config.backend)  # imports kumaru.backends.echo only now
```

Avoid reverse imports from core into feature layers:

```python
# Do not do this inside kumaru.core:
# from kumaru.backends.echo import EchoBackend
```

Use registries or abstract contracts when core-adjacent code must connect to implementations:

```python
from kumaru.core.registry import BACKENDS

BACKENDS.register_lazy("custom", "my_package.backend:CustomBackend")
```

## How to extend it

Add a new core module only when it defines a contract or utility needed by multiple layers and does not depend on a feature layer.

For example, adding a shared request identifier type belongs in core if the server, agent, and evaluation harness all need it:

```python
# src\kumaru\core\request.py
from pydantic import BaseModel

class RequestContext(BaseModel):
    request_id: str
```

Consumers can then import it:

```python
from kumaru.core.request import RequestContext
```

If the new code imports `kumaru.agent`, `kumaru.backends`, `kumaru.server`, `kumaru.train` or `kumaru.eval`, keep it out of `kumaru.core` and place it in the owning feature package instead.

## Failure modes
| Symptom | Likely cause | Fix |
|---|---|---|
| Import cycle during startup | A core module imports a feature module that already imports core. | Move the feature import out of core; use a registry or abstract type. |
| `kumaru config` imports torch or backend libraries | Core or CLI imported a concrete backend too early. | Replace concrete imports with lazy registry entries. |
| A new backend requires editing core | The extension point was hard-coded instead of registered. | Register the backend in `BACKENDS` or through an entry point. |
| Shared models differ between server and agent | A type was duplicated outside core. | Move the shared data contract into `kumaru.core.types` or another focused core module. |

## Related files
- `src\kumaru\core\types.py` — canonical data models shared across layers.
- `src\kumaru\core\config.py` — typed configuration object consumed by subsystems.
- `src\kumaru\core\errors.py` — exception hierarchy used by CLI and server.
- `src\kumaru\core\registry.py` — extension mechanism that avoids hard-coded implementation imports.
- `src\kumaru\core\logging.py` — shared logging setup.
- `src\kumaru\backends\base.py` — consumes core config and type contracts without reversing the dependency.
- `src\kumaru\agent\chat.py` — consumes core contracts to orchestrate a turn.
