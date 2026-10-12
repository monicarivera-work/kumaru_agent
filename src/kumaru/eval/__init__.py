"""Evaluation harness.

If you cannot measure it, you cannot scale it. Every claim in
docs/SCALABILITY.md is meant to be reproducible with `kumaru eval`.
"""

from kumaru.eval.harness import Case, EvalResult, run_eval

__all__ = ["Case", "EvalResult", "run_eval"]
