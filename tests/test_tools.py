"""Tests for kumaru/tools/"""

import pytest
from kumaru.tools.base import BaseTool, ToolError
from kumaru.tools.calculator import CalculatorTool
from kumaru.tools.web_search import WebSearchTool


# ──────────────────────────────────────────────────────────────────────────────
# CalculatorTool
# ──────────────────────────────────────────────────────────────────────────────

class TestCalculatorTool:
    def setup_method(self):
        self.calc = CalculatorTool()

    def test_addition(self):
        assert self.calc.run(expression="2 + 2") == "4"

    def test_subtraction(self):
        assert self.calc.run(expression="10 - 3") == "7"

    def test_multiplication(self):
        assert self.calc.run(expression="6 * 7") == "42"

    def test_division(self):
        result = self.calc.run(expression="10 / 4")
        assert result == "2.5"

    def test_integer_division(self):
        assert self.calc.run(expression="10 // 3") == "3"

    def test_modulo(self):
        assert self.calc.run(expression="10 % 3") == "1"

    def test_exponentiation(self):
        assert self.calc.run(expression="2 ** 10") == "1024"

    def test_complex_expression(self):
        result = self.calc.run(expression="(2 + 3) * 4")
        assert result == "20"

    def test_unary_negation(self):
        assert self.calc.run(expression="-5 + 10") == "5"

    def test_division_by_zero_raises_tool_error(self):
        with pytest.raises(ToolError, match="Division by zero"):
            self.calc.run(expression="1 / 0")

    def test_invalid_expression_raises_tool_error(self):
        with pytest.raises(ToolError):
            self.calc.run(expression="not an expression!!!")

    def test_string_constant_raises_tool_error(self):
        with pytest.raises(ToolError):
            self.calc.run(expression="'hello'")

    def test_to_openai_schema(self):
        schema = self.calc.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculator"
        assert "expression" in schema["function"]["parameters"]["properties"]


# ──────────────────────────────────────────────────────────────────────────────
# WebSearchTool
# ──────────────────────────────────────────────────────────────────────────────

class TestWebSearchTool:
    def setup_method(self):
        self.search = WebSearchTool()  # no api_key → uses stub

    def test_stub_returns_string(self):
        result = self.search.run(query="Python programming")
        assert isinstance(result, str)
        assert "STUB" in result

    def test_stub_respects_num_results(self):
        result = self.search.run(query="AI agents", num_results=2)
        # Stub generates one line per result
        assert result.count("Result for") == 2

    def test_empty_query_raises_tool_error(self):
        with pytest.raises(ToolError, match="cannot be empty"):
            self.search.run(query="   ")

    def test_num_results_clamped_to_max_5(self):
        result = self.search.run(query="test", num_results=10)
        assert result.count("Result for") == 5

    def test_num_results_clamped_to_min_1(self):
        result = self.search.run(query="test", num_results=0)
        assert result.count("Result for") == 1

    def test_to_openai_schema(self):
        schema = self.search.to_openai_schema()
        assert schema["function"]["name"] == "web_search"


# ──────────────────────────────────────────────────────────────────────────────
# BaseTool contract
# ──────────────────────────────────────────────────────────────────────────────

class TestBaseToolContract:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore[abstract]

    def test_concrete_tool_repr(self):
        calc = CalculatorTool()
        assert "calculator" in repr(calc)
