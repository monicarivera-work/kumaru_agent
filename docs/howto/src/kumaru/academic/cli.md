# `src/kumaru/academic/cli.py`

Standalone `kumaru academic` Typer subcommands. The callback loads configuration
and the chosen database, not a model. Commands import their operations lazily.
JSON output escapes untrusted terminal controls; JSON input errors map to
Kumaru's normal CLI errors. Export writes a new file with traceable provenance
and an untrusted-reference notice, ready for `kumaru learn`.

Use `kumaru academic --help`, with group `--database`/`--config` flags before the
subcommand. No approval is inferred from an `add` record; use `review` with
evidence. [Complete commands/examples](../../../../ACADEMIC.md).

Claim commands are `claim-add`, `claims`, `claim-show`, `claim-review`.
Editorial commands are `notice-add`, `notices`, `notice-review`, `check-updates`,
`monitor`. Claims/notices accept paginated listing flags; monitor batches cap at
20 and explicitly return per-source errors.
