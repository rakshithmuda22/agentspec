"""
AgentSpec — pytest-style behavioral contracts for LLM agents.

Usage:
    from agentspec import ContractSet, AgentSession, ToolCall

    spec = ContractSet("research_agent")
    spec.must_call("search")
    spec.must_call_before("search", "summarize")
    spec.must_not_call("delete_file")

    report = spec.check(session)
    report.assert_all_pass()
"""

from agentspec.contracts import (
    AgentSession,
    ContractSet,
    ContractReport,
    ContractResult,
    ToolCall,
    Verdict,
)

__version__ = "0.2.0"
__all__ = [
    "AgentSession",
    "ContractSet",
    "ContractReport",
    "ContractResult",
    "ToolCall",
    "Verdict",
]
