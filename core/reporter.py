"""
core/reporter.py — Human-readable report generation for contract check results.
No API calls. Pure formatting.
"""
from __future__ import annotations
from core.contracts import ContractReport, Verdict


def print_report(report: ContractReport, show_session_trace: bool = False, call_names: list[str] | None = None) -> None:
    """Print a formatted contract report to stdout."""
    BORDER = "=" * 60
    THIN = "-" * 60

    verdict_color = report.overall.value
    print(f"\n{BORDER}")
    print(f"  AgentSpec — Contract Report")
    print(f"  Spec: '{report.spec_name}'")
    print(THIN)

    if show_session_trace and call_names is not None:
        print("  Tool call trace:")
        for i, name in enumerate(call_names):
            print(f"    [{i}] {name}")
        print(THIN)

    for result in report.results:
        icon = "✓" if result.verdict == Verdict.PASS else "✗"
        print(f"  [{icon}] {result.contract_name}")
        print(f"      {result.message}")

    print(THIN)
    print(f"  {report.passed} passed / {report.failed} failed")
    print(f"  Overall verdict: {report.overall.value}")
    print(f"{BORDER}\n")


def format_summary(report: ContractReport) -> str:
    """One-line summary. Good for CI output."""
    return (
        f"AgentSpec '{report.spec_name}': "
        f"{report.passed} passed, {report.failed} failed — {report.overall.value}"
    )
