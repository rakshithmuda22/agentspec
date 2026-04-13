"""
agentspec/reporter.py — Human-readable report generation for contract check results.
No API calls. Pure formatting.
"""
from __future__ import annotations

import logging
from agentspec.contracts import ContractReport, Verdict

logger = logging.getLogger("agentspec")


def print_report(
    report: ContractReport,
    show_session_trace: bool = False,
    call_names: list[str] | None = None,
) -> str:
    """
    Format and log a contract report. Returns the formatted string.

    Uses the 'agentspec' logger at INFO level so users can control output
    via standard Python logging configuration.
    """
    lines = _format_report(report, show_session_trace, call_names)
    output = "\n".join(lines)
    logger.info(output)
    return output


def _format_report(
    report: ContractReport,
    show_session_trace: bool = False,
    call_names: list[str] | None = None,
) -> list[str]:
    """Build the formatted report lines without side effects."""
    BORDER = "=" * 60
    THIN = "-" * 60

    lines = [
        "",
        BORDER,
        "  AgentSpec — Contract Report",
        f"  Spec: '{report.spec_name}'",
        THIN,
    ]

    if show_session_trace and call_names is not None:
        lines.append("  Tool call trace:")
        for i, name in enumerate(call_names):
            lines.append(f"    [{i}] {name}")
        lines.append(THIN)

    for result in report.results:
        icon = "\u2713" if result.verdict == Verdict.PASS else "\u2717"
        lines.append(f"  [{icon}] {result.contract_name}")
        lines.append(f"      {result.message}")

    lines.extend([
        THIN,
        f"  {report.passed} passed / {report.failed} failed",
        f"  Overall verdict: {report.overall.value}",
        BORDER,
        "",
    ])
    return lines


def format_summary(report: ContractReport) -> str:
    """One-line summary suitable for CI output or logging."""
    return (
        f"AgentSpec '{report.spec_name}': "
        f"{report.passed} passed, {report.failed} failed — {report.overall.value}"
    )
