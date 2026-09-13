from pathlib import Path
root = Path('/home/runner/work/kumaru_agent/kumaru_agent')
base = root / 'docs/howto'

def link_items(items):
    return '\n'.join(f'- {x}' for x in items)

def table(rows):
    return '| Name | Signature | What it\'s for |\n|---|---|---|\n' + '\n'.join(f'| `{a}` | `{b}` | {c} |' for a,b,c in rows)

def gotchas(rows):
    return '| Symptom | Likely cause | Fix |\n|---|---|---|\n' + '\n'.join(f'| {a} | {b} | {c} |' for a,b,c in rows)

def doc(title, summary, read, what, why, model, api, snippets, extend, got, related):
    return f"""# `{title}`

> {summary}

**Read this when:** {read}

---

## What it does
{what}

## Why it exists
{why}

## Mental model
{model}

## Public API
{api}

## How to use it
{snippets}

## How to extend it
{extend}

## Gotchas
{gotchas(got)}

## Related files
{link_items(related)}
"""

def write(rel, content):
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip() + '\n', encoding='utf-8')

def py_snip(code):
    return f"```python\n{code.strip()}\n```"

def bash_snip(code):
    return f"```bash\n{code.strip()}\n```"

def yaml_snip(code):
    return f"```yaml\n{code.strip()}\n```"

def js_snip(code):
    return f"```javascript\n{code.strip()}\n```"

# Python package and core docs
write('src/kumaru/__init__.md', doc(
'src/kumaru/__init__.py','Package entry point that re-exports Kumaru\'s stable public types and helpers.','importing Kumaru from another Python program or checking the installed version.',
'This file exposes configuration loading, core chat value types, deliberate exceptions, and `__version__`. It avoids forcing callers to know the internal layer layout for common integrations.',
'It gives integrations a stable facade. Without it, every caller would import deep paths and small internal refactors would become breaking changes.',
'Use `kumaru` for stable library imports; use deeper modules only for layer-specific operations. `__all__` is the top-level compatibility boundary.',
table([('__version__','str','Current package version.'),('KumaruConfig','class KumaruConfig','Root config dataclass.'),('load_config','load_config(path: str | Path | None = None, *, env: dict[str, str] | None = None) -> KumaruConfig','Load defaults, YAML, and env overrides.'),('ChatRequest','class ChatRequest','Backend request payload.'),('Message','class Message','Immutable chat message.'),('Role','class Role(str, Enum)','Message role values.'),('Usage','class Usage','Token accounting.'),('KumaruError','class KumaruError(Exception)','Base deliberate failure.'),('ConfigError','class ConfigError(KumaruError)','Configuration failure.'),('BackendError','class BackendError(KumaruError)','Generation/backend failure.'),('ModelNotAvailableError','class ModelNotAvailableError(BackendError)','Missing model or dependency.')]),
'\n\n'.join([py_snip('from kumaru import __version__\nprint(__version__)'), py_snip('from kumaru import load_config\nconfig = load_config("configs/default.yaml")\nprint(config.backend.name)'), py_snip('from kumaru import ChatRequest, Message\nreq = ChatRequest(messages=[Message.user("hello")], max_tokens=32)\nprint(req.as_dicts())')]),
'Only re-export names that are meant for external callers. For example, to make streaming chunks top-level, import `StreamChunk` from `kumaru.core.types` and add it to `__all__`.',
[('`ImportError` for a backend-specific class','The facade intentionally omits backend implementations.','Import backend classes from `kumaru.backends.*` or use `load_backend`.'),('Version differs from package metadata','`__version__` was not updated with release metadata.','Update release process to keep both in sync.')],
['[CLI](./cli.md)','[Core package](./core/__init__.md)','[Configuration](./core/config.md)']))

write('src/kumaru/cli.md', doc(
'src/kumaru/cli.py','Command-line interface for serving, chatting, asking one question, printing config, and listing backends.','operating Kumaru from a shell or adding a new CLI command.',
'It builds the `kumaru` argument parser, loads configuration, applies CLI overrides, sets up logging, and dispatches subcommands. It is the only layer that exits with numeric status codes or configures process logging.',
'Library code should raise exceptions so notebooks, tests, and servers can handle failures. The CLI converts those exceptions into human-readable stderr and exit codes.',
'Common flags can appear before or after the subcommand. CLI flags override env vars and YAML because they are the most explicit operator input.',
table([('EXIT_OK','int','Successful command status.'),('EXIT_ERROR','int','Failed command status.'),('build_parser','build_parser() -> argparse.ArgumentParser','Construct the command parser.'),('main','main(argv: Sequence[str] | None = None) -> int','Parse args and run a command.')]),
'\n\n'.join([bash_snip('kumaru serve -c configs/default.yaml --host 127.0.0.1 --port 8000'), bash_snip('kumaru ask --backend echo "What backend is active?"'), py_snip('from kumaru.cli import main\nexit_code = main(["backends"])\nprint(exit_code)'), py_snip('from kumaru.cli import build_parser\nargs = build_parser().parse_args(["ask", "hello"])\nprint(args.command)')]),
'Add subcommands in `build_parser()`, then dispatch them in `main()`. Keep `setup_logging` and exception-to-exit-code handling in this file; deeper code should raise `KumaruError` subclasses instead.',
[('`kumaru ask` exits 1','Config validation or backend loading raised `KumaruError`.','Read stderr hint; run `kumaru config -c <file>` to inspect effective settings.'),('Flag ignored','The flag is not copied in `_config_from_args`.','Add an override assignment before `config.validate()`.'),('Backend list omits a custom backend','Runtime registration has not happened in this process.','Register it before calling `available_backends()` or add it to `BACKEND_PATHS`.')],
['[Package facade](./__init__.md)','[Agent](./agent/chat.md)','[Config](./core/config.md)','[Backends](./backends/__init__.md)']))

write('src/kumaru/core/__init__.md', doc(
'src/kumaru/core/__init__.py','Core-layer facade for configuration, errors, logging, registry, and value types.','working inside Kumaru internals and needing dependency-safe imports.',
'It re-exports the vocabulary that all upper layers can depend on. The module keeps `core` independent from backends, agent orchestration, and HTTP serving.',
'The one-way dependency rule prevents cycles and keeps upper layers independently testable. Without it, importing a type could accidentally load web or model dependencies.',
'`kumaru.core` is the bottom layer. Anything here may be imported upward, but it must not import from `kumaru.backends`, `kumaru.agent`, or `kumaru.server`.',
table([('AgentConfig','class AgentConfig','Agent prompt/sampling/memory settings.'),('BackendConfig','class BackendConfig','Backend engine settings.'),('ServerConfig','class ServerConfig','HTTP server settings.'),('KumaruConfig','class KumaruConfig','Full config tree.'),('load_config','load_config(path: str | Path | None = None, *, env: dict[str, str] | None = None) -> KumaruConfig','Load validated config.'),('KumaruError','class KumaruError(Exception)','Base deliberate error.'),('Registry','class Registry(Generic[T])','Name-to-factory registry.'),('Message','class Message','Conversation value type.'),('Role','class Role(str, Enum)','Message role enum.'),('StreamChunk','class StreamChunk','Streaming reply piece.'),('Usage','class Usage','Token usage.'),('get_logger','get_logger(name: str) -> logging.Logger','Namespaced logger.'),('setup_logging','setup_logging(level: str = "INFO", *, json_format: bool = False) -> None','Configure root logging.')]),
'\n\n'.join([py_snip('from kumaru.core import Message, Role\nmsg = Message(Role.USER, "hello")'), py_snip('from kumaru.core import load_config\nconfig = load_config()'), py_snip('from kumaru.core import Registry\nregistry = Registry[int]("number")\nregistry.register("one", lambda: 1)\nprint(registry.create("one"))')]),
'Add new foundational types here only if every upper layer may depend on them. Keep imports acyclic: a new `core` module must not import backend, agent, or server symbols.',
[('Import cycle','A core module imported an upper layer.','Move shared code down into core or inject it from the upper layer.'),('Optional dependency imports during startup','A core re-export imports a backend module.','Re-export only core-owned symbols here.')],
['[Types](./types.md)','[Config](./config.md)','[Errors](./errors.md)','[Registry](./registry.md)','[Logging](./logging.md)']))

write('src/kumaru/core/types.md', doc(
'src/kumaru/core/types.py','Immutable value types shared across CLI, agent, backends, and server translations.','constructing messages, backend requests, streamed chunks, or usage reports.',
'It defines roles, messages, usage accounting, stream chunks, and chat requests. These types are SDK-agnostic, so every layer can exchange the same objects.',
'Backends should not care whether input came from the browser, CLI, or tests. Plain dataclasses and a string enum keep the wire format simple while protecting internal code from mutation surprises.',
'`Message` objects are immutable and normalize string roles to `Role`. `ChatRequest` carries per-request sampling so one backend instance can serve different generation settings.',
table([('Role','class Role(str, Enum)','`system`, `user`, and `assistant` roles.'),('Message','Message(role: Role, content: str)','One immutable conversation turn.'),('Message.to_dict','to_dict(self) -> dict[str, str]','OpenAI-style dict.'),('Message.from_dict','from_dict(cls, data: dict[str, Any]) -> Message','Build from OpenAI-style dict.'),('Message.system','system(cls, content: str) -> Message','Create system message.'),('Message.user','user(cls, content: str) -> Message','Create user message.'),('Message.assistant','assistant(cls, content: str) -> Message','Create assistant message.'),('Usage','Usage(prompt_tokens: int = 0, completion_tokens: int = 0)','Token accounting.'),('Usage.total_tokens','total_tokens(self) -> int','Prompt plus completion tokens.'),('Usage.to_dict','to_dict(self) -> dict[str, int]','JSON-safe usage dict.'),('StreamChunk','StreamChunk(delta: str = "", done: bool = False, usage: Usage | None = None)','One streamed reply event.'),('ChatRequest','ChatRequest(messages: list[Message] = ..., max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95, stop: list[str] = ...)','Backend generation request.'),('ChatRequest.as_dicts','as_dicts(self) -> list[dict[str, str]]','Messages in OpenAI wire format.')]),
'\n\n'.join([py_snip('from kumaru.core.types import Message\nmsg = Message.user("Explain SSE")\nprint(msg.to_dict())'), py_snip('from kumaru.core.types import Message\nmsg = Message.from_dict({"role": "assistant", "content": "ok"})\nprint(msg.role.value)'), py_snip('from kumaru.core.types import ChatRequest, Message\nreq = ChatRequest(messages=[Message.system("Be brief"), Message.user("Hi")], stop=["END"])\nprint(req.as_dicts())'), py_snip('from kumaru.core.types import StreamChunk, Usage\nchunks = [StreamChunk(delta="hi"), StreamChunk(done=True, usage=Usage(2, 1))]')]),
'Add fields only when all backends can safely ignore or consume them. For example, a future `tools` field belongs on `ChatRequest` if it is per-request and backend-facing.',
[('`ValueError: ... is not a valid Role`','Role string is not `system`, `user`, or `assistant`.','Validate/translate external input before constructing `Message`.'),('Cannot assign `msg.content`','`Message` is frozen.','Create a new `Message` with changed content.'),('Usage shows zero tokens','Backend did not report usage.','Treat zero as unknown unless the backend documents exact counts.')],
['[Prompt assembly](../agent/prompt.md)','[Backend base](../backends/base.md)','[Server schemas](../server/schemas.md)']))

write('src/kumaru/core/errors.md', doc(
'src/kumaru/core/errors.py','Deliberate exception hierarchy with HTTP status metadata.','raising operator-facing failures or mapping errors to API responses.',
'It defines a base `KumaruError` plus specific config, backend, model-availability, and registry errors. Each error can serialize itself to JSON and carries an HTTP status.',
'Routes need to distinguish expected operational failures from bugs without parsing strings. Without this hierarchy, the API would either leak tracebacks or flatten useful hints into generic 500s.',
'Raise `KumaruError` subclasses for failures an operator can understand. Let ordinary Python exceptions escape for programming bugs.',
table([('KumaruError','KumaruError(message: str, *, hint: str | None = None)','Base deliberate Kumaru failure.'),('KumaruError.http_status','int','Default HTTP status for the error class.'),('KumaruError.to_dict','to_dict(self) -> dict[str, str]','JSON-safe error payload.'),('ConfigError','class ConfigError(KumaruError)','Invalid startup configuration.'),('BackendError','class BackendError(KumaruError)','Generation/network/OOM backend failure.'),('ModelNotAvailableError','class ModelNotAvailableError(BackendError)','Missing model or optional dependency.'),('RegistryError','class RegistryError(KumaruError)','Unknown or duplicate registry name.')]),
'\n\n'.join([py_snip('from kumaru.core.errors import ConfigError\nraise ConfigError("agent.max_tokens must be > 0")'), py_snip('from kumaru.core.errors import ModelNotAvailableError\nerr = ModelNotAvailableError("model missing", hint="download weights")\nprint(err.http_status, err.to_dict())'), py_snip('from kumaru.core.errors import KumaruError\ntry:\n    raise KumaruError("planned failure")\nexcept KumaruError as exc:\n    print(exc.message)')]),
'Create a subclass when a distinct handler or HTTP status is useful. Example: `class RateLimitError(BackendError): http_status = 429` keeps routing code generic.',
[('API returns 500 for expected failure','Code raised a plain exception.','Raise a `KumaruError` subclass with a helpful hint.'),('UI lacks remediation text','`hint` was omitted.','Pass `hint=` when the operator can take a concrete action.'),('Config error appears mid-request','Validation was deferred.','Move checks into config loading/startup.')],
['[Config](./config.md)','[Registry](./registry.md)','[Chat routes](../server/routes_chat.md)']))

write('src/kumaru/core/config.md', doc(
'src/kumaru/core/config.py','Typed configuration loaded from defaults, YAML, and `KUMARU_*` environment variables.','editing config files, adding settings, or debugging effective runtime config.',
'It defines dataclasses for backend, agent, and server settings, validates unsafe values, and applies environment overrides. Secrets are read through environment variable names rather than committed YAML values.',
'Raw dicts make typos silent and push failures into runtime. Typed sections fail fast with actionable `ConfigError` messages before Kumaru accepts traffic.',
'Precedence is defaults, then YAML, then environment overrides, then CLI overrides in `cli.py`. `load_config()` always returns a validated `KumaruConfig`.',
table([('DEFAULT_SYSTEM_PROMPT','str','Built-in assistant system prompt.'),('BackendConfig','BackendConfig(name: str = "echo", model: str = "", base_url: str = "http://localhost:11434/v1", api_key_env: str = "KUMARU_API_KEY", timeout_s: float = 120.0, device: str = "auto", dtype: str = "auto", context_length: int = 4096, n_threads: int = 0, gpu_layers: int = 0, extra: dict[str, Any] = ...)','Backend settings.'),('BackendConfig.api_key','api_key(self) -> str','Read secret from `api_key_env`.'),('AgentConfig','AgentConfig(system_prompt: str = DEFAULT_SYSTEM_PROMPT, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95, max_history_turns: int = 12, max_history_chars: int = 12000, stop: list[str] = ...)','Conversation settings.'),('ServerConfig','ServerConfig(host: str = "127.0.0.1", port: int = 8000, cors_origins: list[str] = ..., serve_ui: bool = True)','HTTP settings.'),('KumaruConfig','KumaruConfig(backend: BackendConfig = ..., agent: AgentConfig = ..., server: ServerConfig = ..., log_level: str = "INFO", log_json: bool = False)','Whole config tree.'),('KumaruConfig.to_dict','to_dict(self) -> dict[str, Any]','Plain safe dict.'),('KumaruConfig.validate','validate(self) -> None','Fail fast on invalid values.'),('ENV_OVERRIDES','dict[str, str]','Environment variable to dotted config path map.'),('from_dict','from_dict(data: dict[str, Any]) -> KumaruConfig','Build config from parsed YAML dict.'),('apply_env_overrides','apply_env_overrides(config: KumaruConfig, env: dict[str, str] | None = None) -> KumaruConfig','Mutate config with `KUMARU_*` values.'),('load_config','load_config(path: str | Path | None = None, *, env: dict[str, str] | None = None) -> KumaruConfig','Load YAML, apply env, validate.')]),
'\n\n'.join([py_snip('from kumaru.core.config import load_config\nconfig = load_config("configs/openai-compat.yaml")\nprint(config.backend.base_url)'), py_snip('from kumaru.core.config import load_config\nconfig = load_config(env={"KUMARU_BACKEND": "echo", "KUMARU_PORT": "9000"})\nprint(config.server.port)'), py_snip('from kumaru.core.config import from_dict\nconfig = from_dict({"agent": {"max_tokens": 128}})\nconfig.validate()'), bash_snip('KUMARU_MODEL=llama3.1:8b kumaru config -c configs/openai-compat.yaml')]),
'Add a dataclass field, optionally add an `ENV_OVERRIDES` entry, then include validation if bad values would fail later. Example: add `seed: int = 0` to `AgentConfig`; `from_dict()` will accept YAML automatically.',
[('`unknown key(s)`','YAML key does not match a dataclass field.','Fix spelling or add the setting to the correct dataclass.'),('Secret missing despite YAML edit','`api_key` is intentionally not a YAML field.','Set the env var named by `backend.api_key_env`.'),('Env var type error','`_coerce` could not convert the string.','Use integer/float/list syntax matching the existing value type.')],
['[Default config](../../../configs/default.yaml.md)','[CLI](../cli.md)','[OpenAI backend](../backends/openai_compat.md)']))

write('src/kumaru/core/registry.md', doc(
'src/kumaru/core/registry.py','Generic lowercase name-to-factory registry used for pluggable components.','registering runtime backends or designing another plugin point.',
'It maps normalized string names to callables and can instantiate them later. Registration works directly or as a decorator, and errors include available names.',
'Kumaru avoids eager imports of optional dependencies. A registry lets code select `echo` without importing torch-only backends and can support future tools or retrievers.',
'Names are stripped and lowercased. Factories are callables; `create()` simply calls the registered factory with your args.',
table([('Registry','Registry(kind: str)','Create a registry for a component kind.'),('Registry.kind','kind(self) -> str','Human-readable registry kind.'),('Registry.register','register(self, name: str, factory: Callable[..., T] | None = None) -> Callable[..., T] | Callable[[Callable[..., T]], Callable[..., T]]','Register a factory directly or as decorator.'),('Registry.get','get(self, name: str) -> Callable[..., T]','Return a factory.'),('Registry.create','create(self, name: str, *args: object, **kwargs: object) -> T','Instantiate via factory.'),('Registry.names','names(self) -> list[str]','Sorted registered names.')]),
'\n\n'.join([py_snip('from kumaru.core.registry import Registry\nr = Registry[str]("thing")\nr.register("hello", lambda: "world")\nprint(r.create("hello"))'), py_snip('from kumaru.core.registry import Registry\nr = Registry[int]("number")\n@r.register("one")\ndef make_one():\n    return 1\nprint(r.get("one")())'), py_snip('from kumaru.core.registry import Registry\nr = Registry[object]("item")\nprint("missing" in r, r.names())')]),
'Create a dedicated `Registry[YourType]` and publish a small registration API. Example: `tool_registry = Registry[Tool]("tool")`; then modules can call `@tool_registry.register("search")`.',
[('Duplicate registration raises `RegistryError`','Same normalized name was registered twice.','Choose a unique name or guard registration.'),('Lookup says unknown','Module that registers the item was never imported.','Import the plugin module or add a lazy loader like `backends.__init__`.'),('Name case mismatch','Names are normalized, but display strings are not.','Use lowercase canonical names in docs/config.')],
['[Backend registry](../backends/__init__.md)','[Errors](./errors.md)']))

write('src/kumaru/core/logging.md', doc(
'src/kumaru/core/logging.py','Central logging setup and Kumaru-namespaced logger helper.','configuring process logs or adding logging to a module.',
'It configures the root logger once, optionally with JSON formatting, and returns namespaced loggers. Extra logging fields are preserved in JSON output.',
'Library modules must not call `logging.basicConfig`. Central setup prevents duplicate handlers and makes structured logs available without touching every call site later.',
'Application entry points call `setup_logging()`. All other code calls `get_logger("area")` and inherits operator settings.',
table([('JsonFormatter','class JsonFormatter(logging.Formatter)','Render log records as JSON objects.'),('JsonFormatter.format','format(self, record: logging.LogRecord) -> str','Format one record.'),('setup_logging','setup_logging(level: str = "INFO", *, json_format: bool = False) -> None','Configure root logging once; later calls adjust level.'),('get_logger','get_logger(name: str) -> logging.Logger','Return `kumaru.<name>` logger unless already namespaced.')]),
'\n\n'.join([py_snip('from kumaru.core.logging import setup_logging\nsetup_logging("DEBUG", json_format=True)'), py_snip('from kumaru.core.logging import get_logger\nlog = get_logger("agent.chat")\nlog.info("chat turn", extra={"session": "default"})'), py_snip('from kumaru.core.logging import get_logger\nassert get_logger("kumaru.server").name == "kumaru.server"')]),
'If adding a new process entry point, call `setup_logging(config.log_level, json_format=config.log_json)` once near startup. Inside modules, add `log = get_logger("module.name")` and use `extra={...}` for fields.',
[('Duplicate log lines','Another entry point configured handlers separately.','Use `setup_logging` and avoid `basicConfig` elsewhere.'),('JSON logs missing custom field','Field name collides with reserved `LogRecord` attributes.','Choose a non-reserved key in `extra`.'),('Later `json_format=True` has no effect','Formatter is chosen only on first setup.','Configure JSON on the first call at process start.')],
['[CLI](../cli.md)','[Server app](../server/app.md)']))


# Backends
write('src/kumaru/backends/__init__.md', doc(
'src/kumaru/backends/__init__.py','Backend registry and lazy loader for selectable generation engines.','selecting, listing, or adding a backend.',
'It lists built-in backend import paths, exposes a runtime registry, and instantiates the backend named by `BackendConfig.name`. Built-ins are imported only when selected.',
'Optional dependencies like torch should not be required for echo or HTTP use. Lazy loading prevents one backend\'s dependency stack from blocking all others.',
'Built-ins live in `BACKEND_PATHS`; runtime plugins live in `backend_registry`. `load_backend()` checks runtime registrations first, then lazy built-ins.',
table([('BACKEND_PATHS','dict[str, str]','Built-in backend name to `module:ClassName` map.'),('backend_registry','Registry[Backend]','Runtime backend registrations.'),('available_backends','available_backends() -> list[str]','Sorted selectable names.'),('load_backend','load_backend(config: BackendConfig) -> Backend','Instantiate selected backend.'),('Backend','class Backend','Base backend contract.'),('GenerationResult','class GenerationResult','Non-streamed backend result.'),('estimate_tokens','estimate_tokens(text: str) -> int','Rough token count helper.')]),
'\n\n'.join([py_snip('from kumaru.backends import available_backends\nprint(available_backends())'), py_snip('from kumaru.backends import load_backend\nfrom kumaru.core.config import BackendConfig\nbackend = load_backend(BackendConfig(name="echo"))\nprint(backend.health())'), py_snip('from kumaru.backends import backend_registry\nfrom kumaru.backends.echo import EchoBackend\nbackend_registry.register("my_echo", EchoBackend)')]),
'To publish a built-in backend, add a `"name": "module:ClassName"` entry to `BACKEND_PATHS`. For application-local plugins, register a factory with `backend_registry.register("name", Factory)` before calling `load_backend()`.',
[('`unknown backend`','Name is absent from runtime registry and `BACKEND_PATHS`.','Register it or fix `backend.name`.'),('Import error mentions optional extra','Selected backend dependency is not installed.','Install the backend extra named in the hint.'),('Runtime backend not used','Registration happened after `load_backend()`.','Register before constructing `ChatAgent`.')],
['[Backend base](./base.md)','[Echo backend](./echo.md)','[OpenAI-compatible backend](./openai_compat.md)','[Config](../core/config.md)']))

write('src/kumaru/backends/base.md', doc(
'src/kumaru/backends/base.py','Abstract backend contract for engines that turn chat requests into text.','implementing a new generation backend or calling a backend directly.',
'It defines `GenerationResult`, the abstract `Backend.generate()` method, default non-incremental streaming, health reporting, context-manager cleanup, and rough token estimation.',
'The agent should not know whether text comes from a GPU model, GGUF file, or HTTP server. A small backend contract isolates orchestration from engine details.',
'Implement `generate()` first; inherit `stream()` until the engine supports real streaming. `close()` must be safe to call more than once.',
table([('GenerationResult','GenerationResult(text: str, usage: Usage = Usage())','Complete non-streamed reply.'),('Backend','Backend(config: BackendConfig)','Base class for generation engines.'),('Backend.name','str','Registry/logging/health backend name.'),('Backend.generate','generate(self, request: ChatRequest) -> GenerationResult','Required full generation method.'),('Backend.stream','stream(self, request: ChatRequest) -> Iterator[StreamChunk]','Optional incremental generation.'),('Backend.health','health(self) -> dict[str, object]','Liveness and identity details.'),('Backend.close','close(self) -> None','Release resources idempotently.'),('estimate_tokens','estimate_tokens(text: str) -> int','Approximate text tokens as chars/4.')]),
'\n\n'.join([py_snip('from kumaru.backends.base import estimate_tokens\nprint(estimate_tokens("hello world"))'), py_snip('from kumaru.backends.echo import EchoBackend\nfrom kumaru.core.config import BackendConfig\nfrom kumaru.core.types import ChatRequest, Message\nbackend = EchoBackend(BackendConfig())\nresult = backend.generate(ChatRequest(messages=[Message.user("hi")]))\nprint(result.text)'), py_snip('from kumaru.backends.echo import EchoBackend\nfrom kumaru.core.config import BackendConfig\nfrom kumaru.core.types import ChatRequest, Message\nwith EchoBackend(BackendConfig(), delay_s=0) as backend:\n    for chunk in backend.stream(ChatRequest(messages=[Message.user("hi")])):\n        print(chunk.delta, chunk.done)')]),
'Create a subclass with `name` and `generate()`. Example: `class ReverseBackend(Backend): name = "reverse"; def generate(self, request): return GenerationResult(request.messages[-1].content[::-1])`.',
[('Agent cannot instantiate subclass','`generate()` was not implemented.','Implement the abstract method.'),('Streaming appears as one chunk','Subclass inherits default `stream()`.','Override `stream()` for true incremental output.'),('Token counts look approximate','Backend used `estimate_tokens`.','Use engine-reported token counts when available.')],
['[Backend registry](./__init__.md)','[Echo backend](./echo.md)','[Core types](../core/types.md)']))

write('src/kumaru/backends/echo.md', doc(
'src/kumaru/backends/echo.py','Dependency-free deterministic backend that echoes the last user message.','smoke-testing Kumaru without model weights or isolating backend problems.',
'It returns a banner explaining that no real model is active and includes the last user message when present. It streams in small chunks with an optional delay to exercise UI streaming.',
'Fresh clones and tests need a backend that always works without network, GPU, or downloaded weights. Echo proves the CLI, agent, API, and UI paths independently of real model behavior.',
'`EchoBackend` is not intelligent; it is an operational diagnostic. It still respects `max_tokens` approximately so truncation paths can be tested.',
table([('EchoBackend','EchoBackend(config: BackendConfig, *, delay_s: float = 0.01)','Echo backend instance.'),('EchoBackend.name','str','Always `echo`.'),('EchoBackend.generate','generate(self, request: ChatRequest) -> GenerationResult','Return full echo/banner result.'),('EchoBackend.stream','stream(self, request: ChatRequest) -> Iterator[StreamChunk]','Yield 8-character chunks and final usage.'),('EchoBackend.health','health(self) -> dict[str, object]','Report ready echo backend.')]),
'\n\n'.join([py_snip('from kumaru.backends.echo import EchoBackend\nfrom kumaru.core.config import BackendConfig\nbackend = EchoBackend(BackendConfig(), delay_s=0)'), py_snip('from kumaru.core.types import ChatRequest, Message\nresult = backend.generate(ChatRequest(messages=[Message.user("ping")], max_tokens=20))\nprint(result.text)'), py_snip('for chunk in backend.stream(ChatRequest(messages=[Message.user("ping")])):\n    if chunk.delta:\n        print(chunk.delta, end="")'), bash_snip('kumaru ask --backend echo "ping"')]),
'Extend it for tests by subclassing or injecting a different `delay_s`. For example, use `EchoBackend(BackendConfig(), delay_s=0)` in test fixtures to avoid sleeps.',
[('Answer says Kumaru cannot answer','The `echo` backend is selected.','Switch `backend.name` to `openai_compat`, `transformers`, or `llamacpp`.'),('UI streaming seems slow','Default `delay_s` sleeps between chunks.','Use `delay_s=0` in tests.'),('Reply is cut off','`request.max_tokens * 4` truncation applied.','Increase `agent.max_tokens` for diagnostics.')],
['[Backend base](./base.md)','[Backend registry](./__init__.md)','[Test config](../../../configs/test.yaml.md)']))

write('src/kumaru/backends/openai_compat.md', doc(
'src/kumaru/backends/openai_compat.py','HTTP backend for any OpenAI-compatible `/chat/completions` server.','connecting Kumaru to Ollama, llama.cpp server, LM Studio, vLLM, or OpenAI-compatible APIs.',
'It builds chat-completion payloads, posts to `/chat/completions`, parses non-streamed and SSE streamed responses, reads optional bearer tokens from the environment, and checks `/models` for health.',
'One wire format covers many local and remote inference servers. Operators can scale by moving the model server and changing `base_url`, without changing the agent or UI.',
'`backend.model` is required. `BackendConfig.extra` is merged into every payload for server-specific parameters.',
table([('OpenAICompatBackend','OpenAICompatBackend(config: BackendConfig, *, client: httpx.Client | None = None)','OpenAI-compatible HTTP backend.'),('OpenAICompatBackend.name','str','Always `openai_compat`.'),('OpenAICompatBackend.generate','generate(self, request: ChatRequest) -> GenerationResult','POST non-streamed chat completion.'),('OpenAICompatBackend.stream','stream(self, request: ChatRequest) -> Iterator[StreamChunk]','POST streamed chat completion and parse SSE.'),('OpenAICompatBackend.health','health(self) -> dict[str, object]','GET `/models` and report readiness.'),('OpenAICompatBackend.close','close(self) -> None','Close owned HTTP client.')]),
'\n\n'.join([py_snip('from kumaru.core.config import BackendConfig\nfrom kumaru.backends.openai_compat import OpenAICompatBackend\nconfig = BackendConfig(name="openai_compat", model="llama3.1:8b", base_url="http://localhost:11434/v1")\nbackend = OpenAICompatBackend(config)'), py_snip('from kumaru.core.types import ChatRequest, Message\nreply = backend.generate(ChatRequest(messages=[Message.user("hello")]))\nprint(reply.text)'), yaml_snip('backend:\n  name: openai_compat\n  model: llama3.1:8b\n  base_url: http://localhost:11434/v1\n  api_key_env: KUMARU_API_KEY'), bash_snip('KUMARU_API_KEY=sk-... kumaru serve -c configs/openai-compat.yaml')]),
'Add provider-specific options under `backend.extra`; `_payload()` merges them after standard fields. Example: `extra: {frequency_penalty: 0.2}` sends that key with every request.',
[('`backend.model must be set`','Model name is empty.','Set `backend.model` to the server\'s model id.'),('HTTP 404 model not found','Server rejected the model name.','Check loaded model names and `base_url`.'),('Connection failure','Model server is not reachable.','Start the server or fix `backend.base_url`.'),('No token usage in UI','Streaming server does not emit usage events.','Expect elapsed time only, or use a server that sends usage.')],
['[OpenAI config](../../../configs/openai-compat.yaml.md)','[Backend base](./base.md)','[Chat request types](../core/types.md)']))

write('src/kumaru/backends/transformers_backend.md', doc(
'src/kumaru/backends/transformers_backend.py','In-process Hugging Face Transformers backend for local or fine-tuned models.','running weights directly inside Kumaru without a separate model server.',
'It lazily imports torch and transformers, loads tokenizer/model, formats chat prompts, serializes generation with a lock, and supports full and streamed generation.',
'Operators with enough VRAM may want one process and direct access to local fine-tunes or LoRA-compatible directories. Lazy imports keep bare installs usable without torch.',
'`backend.model` is a Hugging Face id or local path. `device=auto` prefers CUDA; `dtype=auto` chooses bf16/fp16 on CUDA and float32 on CPU.',
table([('TransformersBackend','TransformersBackend(config: BackendConfig)','Load tokenizer and model.'),('TransformersBackend.name','str','Always `transformers`.'),('TransformersBackend.generate','generate(self, request: ChatRequest) -> GenerationResult','Run blocking model generation.'),('TransformersBackend.stream','stream(self, request: ChatRequest) -> Iterator[StreamChunk]','Stream via `TextIteratorStreamer` when available.'),('TransformersBackend.health','health(self) -> dict[str, object]','Report backend, model, device, ready.'),('TransformersBackend.close','close(self) -> None','Delete model and clear CUDA cache when available.')]),
'\n\n'.join([yaml_snip('backend:\n  name: transformers\n  model: Qwen/Qwen2.5-7B-Instruct\n  device: auto\n  dtype: auto'), py_snip('from kumaru.core.config import BackendConfig\nfrom kumaru.backends.transformers_backend import TransformersBackend\nbackend = TransformersBackend(BackendConfig(name="transformers", model="Qwen/Qwen2.5-1.5B-Instruct"))'), py_snip('from kumaru.core.types import ChatRequest, Message\nresult = backend.generate(ChatRequest(messages=[Message.user("Say hi")], max_tokens=16))\nprint(result.text)'), bash_snip('kumaru serve -c configs/local-gpu.yaml')]),
'Pass Hugging Face loader options through `backend.extra`. Example: set `extra: {trust_remote_code: true}` only for models you intentionally trust.',
[('Requires torch + transformers','Optional dependencies are missing.','Install the `transformers` extra and an appropriate torch build.'),('OOM while loading','Model/context/dtype too large for hardware.','Use a smaller model, quantized backend, lower context, or CPU.'),('Unknown dtype','`backend.dtype` is not a torch dtype attribute.','Use values such as `float16`, `bfloat16`, `float32`, or `auto`.'),('Concurrent requests queue','A lock serializes one model.','Run more processes or a dedicated model server for concurrency.')],
['[Local GPU config](../../../configs/local-gpu.yaml.md)','[Backend base](./base.md)','[Llama.cpp backend](./llamacpp_backend.md)']))

write('src/kumaru/backends/llamacpp_backend.md', doc(
'src/kumaru/backends/llamacpp_backend.py','In-process llama.cpp backend for quantized GGUF models.','running local CPU/GPU-layer-offloaded GGUF weights.',
'It validates the `.gguf` path, lazily imports `llama_cpp`, creates a `Llama` context, serializes access with a lock, and exposes full/streamed chat completions.',
'GGUF models make real local inference practical on laptops without CUDA. `gpu_layers` lets operators offload part of the model when limited VRAM is available.',
'`backend.model` must be a local GGUF file path. `context_length`, `n_threads`, `gpu_layers`, and `extra` are passed into the llama.cpp runtime.',
table([('LlamaCppBackend','LlamaCppBackend(config: BackendConfig)','Load a GGUF model.'),('LlamaCppBackend.name','str','Always `llamacpp`.'),('LlamaCppBackend.generate','generate(self, request: ChatRequest) -> GenerationResult','Run non-streamed chat completion.'),('LlamaCppBackend.stream','stream(self, request: ChatRequest) -> Iterator[StreamChunk]','Yield llama.cpp streamed deltas.'),('LlamaCppBackend.health','health(self) -> dict[str, object]','Report model file, context length, GPU layers, ready.'),('LlamaCppBackend.close','close(self) -> None','Close llama context if supported.')]),
'\n\n'.join([yaml_snip('backend:\n  name: llamacpp\n  model: models/qwen2.5-7b-instruct-q4_k_m.gguf\n  context_length: 4096\n  n_threads: 0\n  gpu_layers: 0'), py_snip('from kumaru.core.config import BackendConfig\nfrom kumaru.backends.llamacpp_backend import LlamaCppBackend\nbackend = LlamaCppBackend(BackendConfig(name="llamacpp", model="models/model.gguf"))'), py_snip('from kumaru.core.types import ChatRequest, Message\nfor chunk in backend.stream(ChatRequest(messages=[Message.user("hello")])):\n    print(chunk.delta, end="")'), bash_snip('kumaru serve -c configs/cpu-only.yaml')]),
'Pass advanced llama.cpp constructor options via `backend.extra`. Example: `extra: {chat_format: chatml}` if a GGUF requires a specific chat format.',
[('GGUF file not found','`backend.model` path is wrong or relative to another cwd.','Use a valid path, preferably under `models/` from the repo root.'),('Import error for llama_cpp','Optional dependency is missing.','Install the `llamacpp` extra.'),('OOM or swapping','Context or GPU layers too high.','Lower `context_length` or `gpu_layers`; use a smaller quantization.'),('Requests serialize','llama.cpp context is protected by a lock.','Run multiple processes for parallelism.')],
['[CPU config](../../../configs/cpu-only.yaml.md)','[Low VRAM config](../../../configs/low-vram.yaml.md)','[Backend base](./base.md)']))
