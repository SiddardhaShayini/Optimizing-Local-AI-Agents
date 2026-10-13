"""
Unit tests for FastAgent core functionality.
"""
import pytest
from fast_agent import (
    FastAgent,
    AdaptiveComputeController,
    find_free_port,
    is_port_in_use,
    safe_parse_args,
    REACT_GRAMMAR,
)


def test_safe_parse_args_basic():
    # Numbers
    assert safe_parse_args("42") == 42
    assert safe_parse_args("3.14") == 3.14

    # Tuples and lists
    assert safe_parse_args("(10, 20)") == (10, 20)
    assert safe_parse_args("10, 20") == (10, 20)
    assert safe_parse_args("[1, 2, 3]") == [1, 2, 3]

    # Dictionaries and JSON
    assert safe_parse_args('{"a": 1, "b": "test"}') == {"a": 1, "b": "test"}

    # Strings
    assert safe_parse_args("'report.pdf'") == "report.pdf"
    assert safe_parse_args('"report.pdf"') == "report.pdf"
    assert safe_parse_args("report.pdf") == "report.pdf"


def test_safe_parse_args_blocks_rce():
    # Dangerous expressions must NEVER execute and must be parsed as safe literal string or fail
    malicious = "__import__('os').system('echo pwned')"
    parsed = safe_parse_args(malicious)
    # It must either remain a raw unexecuted string or fail, but not execute
    assert parsed == malicious


def test_adaptive_compute_controller():
    # Reflex: short query without tool indicators
    tier1 = AdaptiveComputeController.inspect_budget("Hello there!")
    assert tier1["compute_tier"] == "REFLEX"
    assert tier1["allow_tools"] is False

    # Tool-assisted: math or calculation
    tier2 = AdaptiveComputeController.inspect_budget(
        "Multiply 36 by 12", has_tools=True, tool_names=["multiply"]
    )
    assert tier2["compute_tier"] == "TOOL_ASSISTED"
    assert tier2["allow_tools"] is True

    # Deliberative: complex reasoning
    tier3 = AdaptiveComputeController.inspect_budget(
        "Explain step by step how to implement a Raft consensus cluster in Python",
        has_tools=False,
    )
    assert tier3["compute_tier"] == "DELIBERATIVE"


def test_react_grammar_defined():
    assert "Thought: " in REACT_GRAMMAR
    assert "Action: " in REACT_GRAMMAR
    assert "Final Answer: " in REACT_GRAMMAR


def test_port_utilities():
    free_port = find_free_port(9999)
    assert isinstance(free_port, int)
    assert free_port > 0


def test_tool_execution_safety():
    # Test _exec_tool logic using a mock FastAgent instance without booting C++ server
    agent = FastAgent.__new__(FastAgent)
    agent.tools = {}

    @agent.tool("Multiplies two numbers: (a, b)")
    def multiply(a: int, b: int) -> int:
        return a * b

    @agent.tool("Greets a user: name")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    # Multi-arg tuple
    res = agent._exec_tool("multiply", "(6, 7)")
    assert res == "42"

    # Multi-arg comma separated
    res = agent._exec_tool("multiply", "6, 7")
    assert res == "42"

    # Single arg string
    res = agent._exec_tool("greet", "'Alice'")
    assert res == "Hello, Alice!"

    # Non-existent tool
    res = agent._exec_tool("unknown_tool", "123")
    assert "not found" in res.lower()
