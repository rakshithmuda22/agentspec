"""
pytest_plugin.py — AgentSpec as a pytest plugin.

Register this in conftest.py:
    pytest_plugins = ["pytest_plugin"]

Or install via pyproject.toml entry_points for automatic discovery.

Provides the `contract_spec` fixture factory and the `agent_session` fixture.
"""
from __future__ import annotations
import pytest
from core.contracts import ContractSet, AgentSession, ToolCall


@pytest.fixture
def contract_spec():
    """
    Fixture factory for ContractSet.

    Usage in tests:
        def test_my_agent(contract_spec):
            spec = contract_spec("my_agent")
            spec.must_call("search")
            ...
            report = spec.check(session)
            report.assert_all_pass()
    """
    def _make_spec(name: str) -> ContractSet:
        return ContractSet(name)
    return _make_spec


@pytest.fixture
def agent_session():
    """
    Fixture factory for AgentSession from a list of tool names.

    Usage in tests:
        def test_my_agent(agent_session):
            session = agent_session("search", "summarize", "write_report")
            ...
    """
    def _make_session(*tool_names: str, **metadata) -> AgentSession:
        return AgentSession(
            tool_calls=[ToolCall(name=name, step=i) for i, name in enumerate(tool_names)],
            metadata=metadata,
        )
    return _make_session


@pytest.fixture
def assert_contracts():
    """
    Fixture that provides a one-call contract checker.
    Builds a spec from a dict, checks a session, raises on failure.

    Usage:
        def test_agent(assert_contracts, agent_session):
            session = agent_session("search", "summarize")
            assert_contracts(session, {
                "must_call": ["search"],
                "must_not_call": ["delete_file"],
                "must_call_before": [("search", "summarize")],
            })
    """
    def _assert(session: AgentSession, rules: dict, name: str = "inline_spec") -> None:
        spec = ContractSet(name)
        for tool in rules.get("must_call", []):
            spec.must_call(tool)
        for tool in rules.get("must_not_call", []):
            spec.must_not_call(tool)
        for first, second in rules.get("must_call_before", []):
            spec.must_call_before(first, second)
        for tool, n in rules.get("must_call_at_most", []):
            spec.must_call_at_most(tool, n)
        for tool, n in rules.get("must_call_at_least", []):
            spec.must_call_at_least(tool, n)
        for tool, n in rules.get("must_call_exactly", []):
            spec.must_call_exactly(tool, n)
        spec.check(session).assert_all_pass()

    return _assert
