# `src/kumaru/academic/cli.py`

Standalone `kumaru academic` Typer subcommands. The callback loads configuration
and the chosen database, not a model. Commands import their operations lazily.
JSON output escapes untrusted terminal controls; JSON input errors map to
Kumaru's normal CLI errors. Export writes a new file with traceable provenance
and an untrusted-reference notice, ready for `kumaru learn`.

Use `kumaru academic --help`, with group `--database`/`--config` flags before the
subcommand. No approval is inferred from an `add` record; use `review` with
evidence. [Complete commands/examples](../../../../ACADEMIC.md).
