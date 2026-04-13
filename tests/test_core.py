"""
tests/test_core.py — Unit tests for AgentSpec core contract engine.
No API calls. All deterministic.
"""
import pytest


from agentspec.contracts import (
    ContractSet, AgentSession, ToolCall,
    Verdict
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def make_session(*tool_names: str) -> AgentSession:
    """Quick helper: build a session from a list of tool names."""
    return AgentSession(
        tool_calls=[ToolCall(name=name, step=i) for i, name in enumerate(tool_names)]
    )


# ---------------------------------------------------------------------------
# Test: must_call
# ---------------------------------------------------------------------------

class TestMustCall:

    def test_passes_when_tool_called(self):
        spec = ContractSet("test")
        spec.must_call("search")
        session = make_session("search", "summarize")
        report = spec.check(session)
        assert report.overall == Verdict.PASS

    def test_fails_when_tool_not_called(self):
        spec = ContractSet("test")
        spec.must_call("write_report")
        session = make_session("search", "summarize")
        report = spec.check(session)
        assert report.overall == Verdict.FAIL

    def test_passes_when_tool_called_multiple_times(self):
        spec = ContractSet("test")
        spec.must_call("search")
        session = make_session("search", "search", "search")
        report = spec.check(session)
        assert report.overall == Verdict.PASS

    def test_empty_session_fails(self):
        spec = ContractSet("test")
        spec.must_call("search")
        session = make_session()
        report = spec.check(session)
        assert report.overall == Verdict.FAIL


# ---------------------------------------------------------------------------
# Test: must_not_call
# ---------------------------------------------------------------------------

class TestMustNotCall:

    def test_passes_when_tool_absent(self):
        spec = ContractSet("test")
        spec.must_not_call("delete_file")
        session = make_session("search", "summarize")
        report = spec.check(session)
        assert report.overall == Verdict.PASS

    def test_fails_when_forbidden_tool_called(self):
        spec = ContractSet("test")
        spec.must_not_call("delete_file")
        session = make_session("search", "delete_file", "summarize")
        report = spec.check(session)
        assert report.overall == Verdict.FAIL

    def test_empty_session_passes(self):
        spec = ContractSet("test")
        spec.must_not_call("delete_file")
        session = make_session()
        report = spec.check(session)
        assert report.overall == Verdict.PASS


# ---------------------------------------------------------------------------
# Test: must_call_before
# ---------------------------------------------------------------------------

class TestMustCallBefore:

    def test_passes_correct_order(self):
        spec = ContractSet("test")
        spec.must_call_before("search", "summarize")
        session = make_session("search", "summarize")
        report = spec.check(session)
        assert report.overall == Verdict.PASS

    def test_fails_reversed_order(self):
        spec = ContractSet("test")
        spec.must_call_before("search", "summarize")
        session = make_session("summarize", "search")
        report = spec.check(session)
        assert report.overall == Verdict.FAIL

    def test_fails_when_first_missing(self):
        spec = ContractSet("test")
        spec.must_call_before("search", "summarize")
        session = make_session("summarize")
        report = spec.check(session)
        assert report.overall == Verdict.FAIL

    def test_fails_when_second_missing(self):
        spec = ContractSet("test")
        spec.must_call_before("search", "summarize")
        session = make_session("search")
        report = spec.check(session)
        assert report.overall == Verdict.FAIL

    def test_passes_with_interleaved_other_tools(self):
        spec = ContractSet("test")
        spec.must_call_before("search", "summarize")
        session = make_session("search", "validate", "enrich", "summarize")
        report = spec.check(session)
        assert report.overall == Verdict.PASS

    def test_uses_first_occurrence_of_each(self):
        """If a tool is called multiple times, ordering checks first occurrence."""
        spec = ContractSet("test")
        spec.must_call_before("search", "summarize")
        # search appears at 0, then summarize at 1, then search again at 2
        session = make_session("search", "summarize", "search")
        report = spec.check(session)
        assert report.overall == Verdict.PASS


# ---------------------------------------------------------------------------
# Test: must_call_at_most
# ---------------------------------------------------------------------------

class TestMustCallAtMost:

    def test_passes_under_limit(self):
        spec = ContractSet("test")
        spec.must_call_at_most("search", n=3)
        session = make_session("search", "search")
        report = spec.check(session)
        assert report.overall == Verdict.PASS

    def test_passes_at_exact_limit(self):
        spec = ContractSet("test")
        spec.must_call_at_most("search", n=2)
        session = make_session("search", "search")
        report = spec.check(session)
        assert report.overall == Verdict.PASS

    def test_fails_over_limit(self):
        spec = ContractSet("test")
        spec.must_call_at_most("search", n=2)
        session = make_session("search", "search", "search")
        report = spec.check(session)
        assert report.overall == Verdict.FAIL

    def test_passes_zero_calls_within_limit(self):
        spec = ContractSet("test")
        spec.must_call_at_most("search", n=3)
        session = make_session("summarize")
        report = spec.check(session)
        assert report.overall == Verdict.PASS


# ---------------------------------------------------------------------------
# Test: must_call_at_least
# ---------------------------------------------------------------------------

class TestMustCallAtLeast:

    def test_passes_over_minimum(self):
        spec = ContractSet("test")
        spec.must_call_at_least("search", n=2)
        session = make_session("search", "search", "search")
        report = spec.check(session)
        assert report.overall == Verdict.PASS

    def test_passes_at_exact_minimum(self):
        spec = ContractSet("test")
        spec.must_call_at_least("search", n=2)
        session = make_session("search", "search")
        report = spec.check(session)
        assert report.overall == Verdict.PASS

    def test_fails_under_minimum(self):
        spec = ContractSet("test")
        spec.must_call_at_least("search", n=3)
        session = make_session("search", "search")
        report = spec.check(session)
        assert report.overall == Verdict.FAIL


# ---------------------------------------------------------------------------
# Test: must_not_call_after
# ---------------------------------------------------------------------------

class TestMustNotCallAfter:

    def test_passes_when_trigger_not_called(self):
        spec = ContractSet("test")
        spec.must_not_call_after("search", trigger="finalize")
        session = make_session("search", "search")
        report = spec.check(session)
        assert report.overall == Verdict.PASS

    def test_passes_when_forbidden_not_called_after_trigger(self):
        spec = ContractSet("test")
        spec.must_not_call_after("search", trigger="finalize")
        session = make_session("search", "finalize")
        report = spec.check(session)
        assert report.overall == Verdict.PASS

    def test_fails_when_forbidden_called_after_trigger(self):
        spec = ContractSet("test")
        spec.must_not_call_after("search", trigger="finalize")
        session = make_session("search", "finalize", "search")
        report = spec.check(session)
        assert report.overall == Verdict.FAIL


# ---------------------------------------------------------------------------
# Test: must_call_in_sequence
# ---------------------------------------------------------------------------

class TestMustCallInSequence:

    def test_passes_correct_sequence(self):
        spec = ContractSet("test")
        spec.must_call_in_sequence("search", "summarize", "write_report")
        session = make_session("search", "summarize", "write_report")
        report = spec.check(session)
        assert report.overall == Verdict.PASS

    def test_passes_with_extras_between(self):
        spec = ContractSet("test")
        spec.must_call_in_sequence("search", "write_report")
        session = make_session("search", "validate", "enrich", "write_report")
        report = spec.check(session)
        assert report.overall == Verdict.PASS

    def test_fails_wrong_order(self):
        spec = ContractSet("test")
        spec.must_call_in_sequence("search", "summarize", "write_report")
        session = make_session("summarize", "search", "write_report")
        report = spec.check(session)
        assert report.overall == Verdict.FAIL

    def test_fails_missing_step(self):
        spec = ContractSet("test")
        spec.must_call_in_sequence("search", "summarize", "write_report")
        session = make_session("search", "write_report")
        report = spec.check(session)
        assert report.overall == Verdict.FAIL


# ---------------------------------------------------------------------------
# Test: Multiple contracts in one spec
# ---------------------------------------------------------------------------

class TestMultipleContracts:

    def test_all_pass(self):
        spec = ContractSet("multi")
        spec.must_call("search")
        spec.must_not_call("delete_file")
        spec.must_call_before("search", "summarize")
        session = make_session("search", "summarize")
        report = spec.check(session)
        assert report.passed == 3
        assert report.failed == 0
        assert report.overall == Verdict.PASS

    def test_partial_fail(self):
        spec = ContractSet("multi")
        spec.must_call("search")           # PASS
        spec.must_not_call("delete_file")  # PASS
        spec.must_call("write_report")     # FAIL
        session = make_session("search", "summarize")
        report = spec.check(session)
        assert report.passed == 2
        assert report.failed == 1
        assert report.overall == Verdict.FAIL

    def test_all_fail(self):
        spec = ContractSet("multi")
        spec.must_call("write_report")    # FAIL
        spec.must_not_call("summarize")   # FAIL
        session = make_session("summarize")
        report = spec.check(session)
        assert report.failed == 2

    def test_empty_spec_on_empty_session(self):
        """Zero contracts → always passes."""
        spec = ContractSet("empty")
        session = make_session()
        report = spec.check(session)
        assert report.overall == Verdict.PASS
        assert report.passed == 0


# ---------------------------------------------------------------------------
# Test: ContractReport
# ---------------------------------------------------------------------------

class TestContractReport:

    def test_assert_all_pass_ok(self):
        spec = ContractSet("test")
        spec.must_call("search")
        session = make_session("search")
        report = spec.check(session)
        report.assert_all_pass()  # Should not raise

    def test_assert_all_pass_raises_on_failure(self):
        spec = ContractSet("test")
        spec.must_call("write_report")
        session = make_session("search")
        report = spec.check(session)
        with pytest.raises(AssertionError) as exc_info:
            report.assert_all_pass()
        assert "write_report" in str(exc_info.value)

    def test_report_str_contains_verdict(self):
        spec = ContractSet("test")
        spec.must_call("search")
        session = make_session("search")
        report = spec.check(session)
        report_str = str(report)
        assert "PASS" in report_str

    def test_report_counts(self):
        spec = ContractSet("test")
        spec.must_call("search")    # PASS
        spec.must_call("missing")   # FAIL
        session = make_session("search")
        report = spec.check(session)
        assert report.passed == 1
        assert report.failed == 1


# ---------------------------------------------------------------------------
# Test: AgentSession helpers
# ---------------------------------------------------------------------------

class TestAgentSession:

    def test_calls_filters_correctly(self):
        session = make_session("search", "summarize", "search")
        assert len(session.calls("search")) == 2
        assert len(session.calls("summarize")) == 1
        assert len(session.calls("missing")) == 0

    def test_call_names_preserves_order(self):
        session = make_session("a", "b", "c", "a")
        assert session.call_names() == ["a", "b", "c", "a"]

    def test_index_of_first_found(self):
        session = make_session("a", "b", "c")
        assert session.index_of_first("b") == 1

    def test_index_of_first_not_found(self):
        session = make_session("a", "b", "c")
        assert session.index_of_first("z") is None

    def test_index_of_first_returns_first_occurrence(self):
        session = make_session("a", "b", "a", "a")
        assert session.index_of_first("a") == 0
