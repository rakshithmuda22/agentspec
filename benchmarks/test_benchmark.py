"""
Proper benchmarks using pytest-benchmark (industry standard).

Run: pytest benchmarks/test_benchmark.py --benchmark-only
"""
import pytest
from agentspec import ContractSet, AgentSession, ToolCall


def _build_session(n_calls: int) -> AgentSession:
    """Build a session with n tool calls."""
    tools = ["search", "fetch", "parse", "summarize", "write", "validate"]
    return AgentSession(
        tool_calls=[
            ToolCall(name=tools[i % len(tools)], args={"step": i}, result=f"result_{i}")
            for i in range(n_calls)
        ]
    )


def _build_spec() -> ContractSet:
    """Build a realistic contract set with 7 contracts."""
    spec = ContractSet("benchmark_agent")
    spec.must_call("search")
    spec.must_call("summarize")
    spec.must_call_before("search", "summarize")
    spec.must_call_before("fetch", "parse")
    spec.must_not_call("delete_file")
    spec.must_call_at_most("search", n=100)
    spec.must_call_at_least("search", n=1)
    return spec


# --- Benchmarks at different session sizes ---

def test_check_5_calls(benchmark):
    """7 contracts on a 5-call session."""
    session = _build_session(5)
    spec = _build_spec()
    result = benchmark(spec.check, session)
    assert result.overall.value == "PASS"


def test_check_10_calls(benchmark):
    """7 contracts on a 10-call session."""
    session = _build_session(10)
    spec = _build_spec()
    result = benchmark(spec.check, session)
    assert result.overall.value == "PASS"


def test_check_25_calls(benchmark):
    """7 contracts on a 25-call session."""
    session = _build_session(25)
    spec = _build_spec()
    result = benchmark(spec.check, session)
    assert result.overall.value == "PASS"


def test_check_50_calls(benchmark):
    """7 contracts on a 50-call session."""
    session = _build_session(50)
    spec = _build_spec()
    result = benchmark(spec.check, session)
    assert result.overall.value == "PASS"


def test_check_100_calls(benchmark):
    """7 contracts on a 100-call session."""
    session = _build_session(100)
    spec = _build_spec()
    result = benchmark(spec.check, session)
    assert result.overall.value == "PASS"


def test_check_with_assert(benchmark):
    """7 contracts + assert_all_pass() on a 25-call session."""
    session = _build_session(25)
    spec = _build_spec()

    def check_and_assert():
        report = spec.check(session)
        report.assert_all_pass()
        return report

    result = benchmark(check_and_assert)
    assert result.overall.value == "PASS"
