"""
tests/test_scenarios.py — Parameterized scenario testing with AgentSpec.

Shows the pattern for testing the SAME contract spec against MULTIPLE
agent scenarios. This is the primary use case for CI regression testing:
  - Add a new scenario whenever you find a new agent failure mode
  - The same ContractSet gates all of them
"""
import pytest


from agentspec.contracts import ContractSet, AgentSession, ToolCall, Verdict


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def session(*tool_names: str, **kw) -> AgentSession:
    return AgentSession(
        tool_calls=[ToolCall(name=n, step=i) for i, n in enumerate(tool_names)],
        metadata=kw,
    )


# ---------------------------------------------------------------------------
# Define one ContractSet that ALL good research agent runs must satisfy
# ---------------------------------------------------------------------------

RESEARCH_SPEC = ContractSet("research_agent_contract")
RESEARCH_SPEC.must_call("search")
RESEARCH_SPEC.must_call("write_report")
RESEARCH_SPEC.must_call_before("search", "summarize")
RESEARCH_SPEC.must_call_before("summarize", "write_report")
RESEARCH_SPEC.must_not_call("delete_file")
RESEARCH_SPEC.must_call_at_most("search", n=3)


# ---------------------------------------------------------------------------
# Passing scenarios — all should PASS against RESEARCH_SPEC
# ---------------------------------------------------------------------------

PASSING_SCENARIOS = [
    pytest.param(
        session("search", "summarize", "write_report"),
        id="minimal_correct"
    ),
    pytest.param(
        session("search", "search", "summarize", "write_report"),
        id="two_searches_before_summarize"
    ),
    pytest.param(
        session("search", "validate", "summarize", "review", "write_report"),
        id="extra_tools_between_required_steps"
    ),
    pytest.param(
        session("search", "search", "search", "summarize", "write_report"),
        id="three_searches_at_limit"
    ),
    pytest.param(
        session("search", "summarize", "summarize", "write_report"),
        id="double_summarize_allowed"
    ),
]


@pytest.mark.parametrize("good_session", PASSING_SCENARIOS)
def test_research_spec_passes_good_sessions(good_session):
    """Every session in PASSING_SCENARIOS must pass all contracts."""
    report = RESEARCH_SPEC.check(good_session)
    assert report.overall == Verdict.PASS, (
        f"Expected PASS but got FAIL:\n{report}"
    )


# ---------------------------------------------------------------------------
# Failing scenarios — all should FAIL against RESEARCH_SPEC
# ---------------------------------------------------------------------------

FAILING_SCENARIOS = [
    pytest.param(
        session("summarize", "search", "write_report"),
        "must_call_before(search, summarize)",
        id="summarize_before_search"
    ),
    pytest.param(
        session("search", "summarize"),
        "must_call(write_report)",
        id="missing_write_report"
    ),
    pytest.param(
        session("summarize", "write_report"),
        "must_call(search)",
        id="missing_search"
    ),
    pytest.param(
        session("search", "summarize", "delete_file", "write_report"),
        "must_not_call(delete_file)",
        id="calls_forbidden_delete_file"
    ),
    pytest.param(
        session("search", "search", "search", "search", "summarize", "write_report"),
        "must_call_at_most(search, n=3)",
        id="exceeds_search_limit"
    ),
]


@pytest.mark.parametrize("bad_session,expected_failure", FAILING_SCENARIOS)
def test_research_spec_catches_bad_sessions(bad_session, expected_failure):
    """Every session in FAILING_SCENARIOS must trigger at least one contract failure."""
    report = RESEARCH_SPEC.check(bad_session)
    assert report.overall == Verdict.FAIL, (
        f"Expected FAIL for scenario but got PASS. Expected failure: {expected_failure}"
    )
    # Verify the specific expected contract failed
    failed_names = [r.contract_name for r in report.results if r.verdict == Verdict.FAIL]
    assert any(expected_failure in name for name in failed_names), (
        f"Expected '{expected_failure}' to fail but it passed. Failures: {failed_names}"
    )


# ---------------------------------------------------------------------------
# Test: must_call_exactly
# ---------------------------------------------------------------------------

class TestMustCallExactly:

    def test_passes_exact_count(self):
        spec = ContractSet("exact")
        spec.must_call_exactly("confirm", n=1)
        s = session("search", "confirm", "write_report")
        assert spec.check(s).overall == Verdict.PASS

    def test_fails_too_few(self):
        spec = ContractSet("exact")
        spec.must_call_exactly("confirm", n=2)
        s = session("search", "confirm", "write_report")
        report = spec.check(s)
        assert report.overall == Verdict.FAIL
        assert "too few" in report.results[0].message

    def test_fails_too_many(self):
        spec = ContractSet("exact")
        spec.must_call_exactly("search", n=1)
        s = session("search", "search", "summarize")
        report = spec.check(s)
        assert report.overall == Verdict.FAIL
        assert "too many" in report.results[0].message

    def test_passes_zero_exactly(self):
        spec = ContractSet("exact")
        spec.must_call_exactly("delete_file", n=0)
        s = session("search", "summarize")
        assert spec.check(s).overall == Verdict.PASS

    def test_fails_zero_expected_but_called(self):
        spec = ContractSet("exact")
        spec.must_call_exactly("delete_file", n=0)
        s = session("search", "delete_file", "summarize")
        assert spec.check(s).overall == Verdict.FAIL


# ---------------------------------------------------------------------------
# Test: must_call_with_arg
# ---------------------------------------------------------------------------

class TestMustCallWithArg:

    def test_passes_when_arg_matches(self):
        spec = ContractSet("test")
        spec.must_call_with_arg("search", "source", "verified")
        s = AgentSession(tool_calls=[
            ToolCall(name="search", args={"query": "AI", "source": "verified"}, step=0),
        ])
        assert spec.check(s).overall == Verdict.PASS

    def test_fails_when_arg_wrong_value(self):
        spec = ContractSet("test")
        spec.must_call_with_arg("search", "source", "verified")
        s = AgentSession(tool_calls=[
            ToolCall(name="search", args={"query": "AI", "source": "unverified"}, step=0),
        ])
        report = spec.check(s)
        assert report.overall == Verdict.FAIL
        assert "verified" in report.results[0].message

    def test_fails_when_tool_not_called(self):
        spec = ContractSet("test")
        spec.must_call_with_arg("search", "source", "verified")
        s = session("summarize")
        assert spec.check(s).overall == Verdict.FAIL

    def test_passes_when_one_of_many_calls_matches(self):
        spec = ContractSet("test")
        spec.must_call_with_arg("search", "mode", "deep")
        s = AgentSession(tool_calls=[
            ToolCall(name="search", args={"mode": "quick"}, step=0),
            ToolCall(name="search", args={"mode": "deep"}, step=1),
        ])
        assert spec.check(s).overall == Verdict.PASS


# ---------------------------------------------------------------------------
# Test: check_all (parameterized batch checking)
# ---------------------------------------------------------------------------

class TestCheckAll:

    def test_check_all_returns_one_report_per_session(self):
        spec = ContractSet("batch")
        spec.must_call("search")
        sessions = [
            session("search", "summarize"),
            session("summarize"),  # will fail
            session("search"),
        ]
        reports = spec.check_all(sessions)
        assert len(reports) == 3
        assert reports[0].overall == Verdict.PASS
        assert reports[1].overall == Verdict.FAIL
        assert reports[2].overall == Verdict.PASS

    def test_check_all_empty_list(self):
        spec = ContractSet("batch")
        spec.must_call("search")
        assert spec.check_all([]) == []


# ---------------------------------------------------------------------------
# Test: pytest fixtures from pytest_plugin.py
# ---------------------------------------------------------------------------

def test_contract_spec_fixture(contract_spec, agent_session):
    """Test that the contract_spec fixture creates a working ContractSet."""
    spec = contract_spec("test_via_fixture")
    spec.must_call("search")
    session_obj = agent_session("search", "summarize")
    report = spec.check(session_obj)
    report.assert_all_pass()


def test_agent_session_fixture(agent_session):
    """Test that the agent_session fixture creates a proper AgentSession."""
    s = agent_session("a", "b", "c")
    assert len(s.tool_calls) == 3
    assert s.call_names() == ["a", "b", "c"]
    assert s.tool_calls[0].step == 0


def test_assert_contracts_fixture(assert_contracts, agent_session):
    """Test the assert_contracts convenience fixture."""
    s = agent_session("search", "summarize", "write_report")
    assert_contracts(s, {
        "must_call": ["search", "write_report"],
        "must_not_call": ["delete_file"],
        "must_call_before": [("search", "summarize")],
        "must_call_at_most": [("search", 3)],
    })


def test_assert_contracts_fixture_fails_correctly(assert_contracts, agent_session):
    """assert_contracts should raise AssertionError on failure."""
    s = agent_session("summarize", "write_report")  # missing search
    with pytest.raises(AssertionError) as exc:
        assert_contracts(s, {"must_call": ["search"]})
    assert "search" in str(exc.value)
