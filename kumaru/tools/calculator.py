"""
kumaru/tools/calculator.py
---------------------------
A simple calculator tool.

Why include this?
  It illustrates the full tool lifecycle without any external API calls,
  making it ideal for learning and unit testing.

  It also shows a real-world problem: LLMs are bad at arithmetic.  By
  offloading maths to a deterministic tool we get correct answers every time.

Safety note
-----------
We use ``ast.literal_eval`` instead of ``eval`` to parse expressions.
``eval`` can execute arbitrary Python code – never use it on user input.
``ast.literal_eval`` only evaluates literals (numbers, strings, …) and
raises ``ValueError`` on anything else.

For a production calculator you would use a proper expression parser library
such as `simpleeval` or `pyparsing`.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from kumaru.tools.base import BaseTool, ToolError

# Mapping of safe arithmetic operators.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a parsed AST node using only safe operators."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ToolError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.BinOp):
        op_fn = _OPS.get(type(node.op))
        if op_fn is None:
            raise ToolError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _OPS.get(type(node.op))
        if op_fn is None:
            raise ToolError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))
    raise ToolError(f"Unsupported expression node: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """Evaluate a mathematical expression and return the result.

    The model should call this whenever it needs to compute a numeric answer.

    Example expressions:
      "2 + 2"        → "4.0"
      "10 / 3"       → "3.3333333333333335"
      "(2 ** 10)"    → "1024.0"
    """

    name = "calculator"
    description = (
        "Evaluates a mathematical expression and returns the numeric result. "
        "Use this for any arithmetic, including addition, subtraction, "
        "multiplication, division, powers, and modulo. "
        "Do NOT use this for symbolic algebra or calculus."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "A mathematical expression to evaluate, e.g. '2 + 2' or "
                    "'(100 * 1.08) / 12'."
                ),
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    def run(self, expression: str, **_: Any) -> str:  # type: ignore[override]
        """Evaluate *expression* and return the result as a string."""
        expression = expression.strip()
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"Invalid expression: {exc}") from exc

        try:
            result = _safe_eval(tree.body)
        except ZeroDivisionError:
            raise ToolError("Division by zero")

        # Return a clean string representation.
        if result == int(result):
            return str(int(result))
        return str(result)
