"""
tests/test_integration.py — Integration tests for the full AgentSpec pipeline.
Uses mock sessions — no API calls.
"""


import pytest
from agentspec.contracts import ContractSet, Verdict
from agentspec.recorder import (
    make_research_agent_session,
    make_broken_research_agent_session,
    make_customer_support_session,
)
from agentspec.reporter import format_summary


def build_research_spec() -> ContractSet:
    spec = ContractSet("research_agent")
    spec.must_call("search")
    spec.must_call("write_report")
    spec.must_call_before("search", "summarize")
    spec.must_call_before("summarize", "write_report")
    spec.must_not_call("delete_file")
    spec.must_call_at_most("search", n=3)
    spec.must_call_in_sequence("search", "summarize", "write_report")
    return spec


def test_good_agent_passes_all_contracts():
    spec = build_research_spec()
    session = make_research_agent_session()
    report = spec.check(session)
    assert report.overall == Verdict.PASS
    assert report.failed == 0
    report.assert_all_pass()  # Should not raise


def test_broken_agent_fails_multiple_contracts():
    spec = build_research_spec()
    session = make_broken_research_agent_session()
    report = spec.check(session)
    assert report.overall == Verdict.FAIL
    assert report.failed >= 3  # Fails: ordering, delete_file, search limit, sequence


def test_broken_agent_fails_delete_file_contract():
    spec = ContractSet("safety")
    spec.must_not_call("delete_file")
    session = make_broken_research_agent_session()
    report = spec.check(session)
    assert report.overall == Verdict.FAIL


def test_broken_agent_fails_search_limit():
    spec = ContractSet("cost_control")
    spec.must_call_at_most("search", n=3)
    session = make_broken_research_agent_session()
    report = spec.check(session)
    # broken session has 4 searches
    assert report.overall == Verdict.FAIL


def test_customer_support_agent_spec():
    spec = ContractSet("support_agent")
    spec.must_call("lookup_customer")
    spec.must_call("lookup_order")
    spec.must_call("send_email")
    spec.must_call_before("lookup_customer", "send_email")
    spec.must_not_call("delete_customer")
    spec.must_call_in_sequence("lookup_customer", "lookup_order", "send_email")

    session = make_customer_support_session()
    report = spec.check(session)
    assert report.overall == Verdict.PASS


def test_format_summary_contains_verdict():
    spec = ContractSet("test")
    spec.must_call("search")
    session = make_research_agent_session()
    report = spec.check(session)
    summary = format_summary(report)
    assert "PASS" in summary or "FAIL" in summary
    assert "test" in summary


def test_assert_all_pass_integration():
    """Verify that a broken agent's report raises AssertionError in pytest context."""
    spec = ContractSet("research_agent")
    spec.must_not_call("delete_file")
    session = make_broken_research_agent_session()
    report = spec.check(session)
    with pytest.raises(AssertionError):
        report.assert_all_pass()


def test_fluent_chain_interface():
    """Verify that method chaining works correctly."""
    spec = (
        ContractSet("chained")
        .must_call("a")
        .must_not_call("b")
        .must_call_before("a", "c")
        .must_call_at_most("a", n=5)
    )
    session = make_broken_research_agent_session()  # doesn't have a or c
    report = spec.check(session)
    # 'a' not in broken session, 'b' not in broken session, etc.
    # Just check it runs without error
    assert isinstance(report.overall, Verdict)
