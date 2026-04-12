"""
demo/demo.py — Shows AgentSpec's value in under 60 seconds.

Demonstrates:
  1. A well-behaved agent session that passes all contracts
  2. A broken agent session that fails multiple contracts
  3. How to use AgentSpec in pytest (shown as code)

Run:
    python demo/demo.py

Works WITHOUT an API key — uses mock agent sessions.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.contracts import ContractSet, AgentSession, ToolCall
from core.recorder import make_research_agent_session, make_broken_research_agent_session
from core.reporter import print_report, format_summary


# ---------------------------------------------------------------------------
# Define contracts for a "research agent"
# ---------------------------------------------------------------------------
def build_research_agent_spec() -> ContractSet:
    spec = ContractSet("research_agent")
    spec.must_call("search")                          # Must search at least once
    spec.must_call("write_report")                    # Must produce a report
    spec.must_call_before("search", "summarize")      # Must search before summarizing
    spec.must_call_before("summarize", "write_report") # Must summarize before writing
    spec.must_not_call("delete_file")                 # Must never delete files
    spec.must_call_at_most("search", n=3)             # Should not search more than 3 times
    spec.must_call_in_sequence("search", "summarize", "write_report")  # Must follow this flow
    return spec


def main():
    BORDER = "=" * 60

    print(f"\n{BORDER}")
    print("  AGENTSPEC — Structural Contract Testing for LLM Agents")
    print("  Write behavioral assertions. Get deterministic pass/fail.")
    print(BORDER)

    mode_label = "MOCK (demo)" if config.is_mock_mode() else "LIVE (Claude API)"
    print(f"\n  Mode: {mode_label}")
    if config.is_mock_mode():
        print("  Using simulated agent traces — no API key required.")

    spec = build_research_agent_spec()

    print(f"\n  Contracts defined for '{spec.name}':")
    for name, _ in spec._contracts:
        print(f"    • {name}")

    # -----------------------------------------------------------------------
    # Test 1: Good agent
    # -----------------------------------------------------------------------
    print(f"\n{'-' * 60}")
    print("  TEST 1: Well-behaved agent")
    print(f"  (searches twice, then summarizes, then writes report)")
    print(f"{'-' * 60}")

    good_session = make_research_agent_session()
    good_report = spec.check(good_session)
    print_report(good_report, show_session_trace=True, call_names=good_session.call_names())

    # -----------------------------------------------------------------------
    # Test 2: Broken agent
    # -----------------------------------------------------------------------
    print(f"{'-' * 60}")
    print("  TEST 2: Broken agent")
    print(f"  (summarizes first, calls delete_file, searches 4 times)")
    print(f"{'-' * 60}")

    broken_session = make_broken_research_agent_session()
    broken_report = spec.check(broken_session)
    print_report(broken_report, show_session_trace=True, call_names=broken_session.call_names())

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"{BORDER}")
    print("  RESULTS SUMMARY:")
    print(f"  {format_summary(good_report)}")
    print(f"  {format_summary(broken_report)}")

    # -----------------------------------------------------------------------
    # Show pytest usage
    # -----------------------------------------------------------------------
    print(f"\n{'-' * 60}")
    print("  HOW TO USE IN PYTEST (copy-paste ready):")
    print(f"{'-' * 60}")
    print("""
  from core.contracts import ContractSet, AgentSession, ToolCall

  def test_research_agent_behavior():
      spec = ContractSet("research_agent")
      spec.must_call("search")
      spec.must_call_before("search", "summarize")
      spec.must_not_call("delete_file")
      spec.must_call_at_most("search", n=3)

      # Replace this with your real agent runner
      session = run_my_agent("summarize recent AI news")

      report = spec.check(session)
      report.assert_all_pass()   # raises AssertionError if any contract fails
""")
    print(BORDER + "\n")


if __name__ == "__main__":
    main()
