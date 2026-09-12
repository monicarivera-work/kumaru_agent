"""Core layer: the contracts every other layer depends on.

Nothing in `kumaru.core` may import from `kumaru.backends`, `kumaru.agent`,
`kumaru.server`, or `kumaru.train`. Dependencies point *inward* only. That rule
is what keeps the system extensible: you can add a backend or a tool without
touching the core, and you can never create an import cycle.
"""
