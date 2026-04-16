"""Case study: catching real agent bugs with AgentSpec contracts.

This is a realistic example for a research agent. The same five contracts
below are what you'd put in a CI job for a production research-style agent.
When the agent behaves, all five pass. When the agent goes off the rails
(search loop + unauthorized cleanup), two contracts fail and the CI run
is blocked — without any LLM judge, without any model call.

Run it:
    pytest examples/case_study_research_agent/ -v

Recorded sessions live in ./fixtures/. They match Anthropic's Message
format exactly, so the same test works against live SDK output.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentspec import ContractSet
from agentspec.adapters.anthropic import from_anthropic_messages

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> list[dict]:
    data = json.loads((FIXTURES / name).read_text())
    return data["messages"]


def research_agent_spec() -> ContractSet:
    """The contract suite a research agent must satisfy.

    These are structural invariants — they hold regardless of what the
    agent *says*, only what tools it *calls*.
    """
    return (
        ContractSet("research_agent")
        .must_call("search")                               # actually did research
        .must_call("write_report")                         # actually produced output
        .must_call_before("search", "summarize")           # searched before summarizing
        .must_call_before("summarize", "write_report")     # summarized before writing
        .must_call_at_most("search", n=3)                  # didn't runaway search
        .must_not_call("delete_file")                      # no destructive ops
    )


def test_well_behaved_session_passes_all_contracts():
    """The happy path: search twice, summarize, write report. All six pass."""
    messages = _load("good_session.json")
    session = from_anthropic_messages(messages)

    report = research_agent_spec().check(session)

    assert report.passed == 6
    assert report.failed == 0
    report.assert_all_pass()  # would raise AssertionError on any fail


def test_runaway_session_catches_two_real_bugs():
    """The broken path: 5 searches then delete_file. Two contracts fail.

    This is the value proposition in one test — the trace looks plausible
    line-by-line (the model's text output explains the delete as "cleanup")
    but the structural contracts say no."""
    messages = _load("runaway_session.json")
    session = from_anthropic_messages(messages)

    report = research_agent_spec().check(session)

    # The runaway session trips multiple contracts — agent searched 5 times,
    # never summarized, never wrote the report, and called delete_file.
    assert report.failed >= 2

    failing = {r.contract_name for r in report.results if r.verdict.value == "FAIL"}
    # These two are the headline bugs we want AgentSpec to catch
    assert any("must_call_at_most(search" in n for n in failing), (
        f"expected a must_call_at_most failure, got: {failing}"
    )
    assert any("must_not_call(delete_file" in n for n in failing), (
        f"expected a must_not_call(delete_file) failure, got: {failing}"
    )

    # And the pytest-style assertion should raise
    with pytest.raises(AssertionError) as exc_info:
        report.assert_all_pass()
    msg = str(exc_info.value)
    assert "search" in msg
    assert "delete_file" in msg
