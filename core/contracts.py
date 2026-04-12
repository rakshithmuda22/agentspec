"""
core/contracts.py — Contract definitions and assertion engine.

AgentSpec lets you write STRUCTURAL contracts about LLM agent behavior.
These are not quality metrics (no LLM-graded hallucination scores).
They are deterministic pass/fail assertions about tool-call sequences.

Example usage:
    from core.contracts import AgentSession, ContractSet

    spec = ContractSet("research_agent")
    spec.must_call("search")
    spec.must_call_before("search", "summarize")
    spec.must_not_call("delete_file")
    spec.must_call_at_most("search", n=3)

    session = AgentSession(tool_calls=[
        ToolCall(name="search", args={"query": "India AI"}, result="..."),
        ToolCall(name="summarize", args={}, result="..."),
    ])

    report = spec.check(session)
    print(report)  # pass/fail per contract, overall verdict
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class ToolCall:
    """A single tool invocation recorded from an agent run."""
    name: str
    args: dict = field(default_factory=dict)
    result: str = ""
    # Optional: step index in the agent's execution trace
    step: Optional[int] = None


@dataclass
class AgentSession:
    """
    A recorded agent execution: the sequence of tool calls made during one run.
    This is what you replay against contracts.

    In production, you'd build this from OpenTelemetry traces, LangSmith traces,
    or any agent framework that logs tool calls. In tests, you build it directly.
    """
    tool_calls: list[ToolCall]
    metadata: dict = field(default_factory=dict)

    def calls(self, tool_name: str) -> list[ToolCall]:
        """Return all calls to a specific tool, in order."""
        return [tc for tc in self.tool_calls if tc.name == tool_name]

    def call_names(self) -> list[str]:
        """Return the ordered sequence of tool names called."""
        return [tc.name for tc in self.tool_calls]

    def index_of_first(self, tool_name: str) -> Optional[int]:
        """Return the position (0-indexed) of the first call to this tool, or None."""
        for i, tc in enumerate(self.tool_calls):
            if tc.name == tool_name:
                return i
        return None


@dataclass
class ContractResult:
    """Result of checking one contract against a session."""
    contract_name: str
    verdict: Verdict
    message: str

    def __str__(self) -> str:
        icon = "✓" if self.verdict == Verdict.PASS else "✗"
        return f"  [{icon}] {self.contract_name}: {self.message}"


@dataclass
class ContractReport:
    """Full report from checking all contracts in a ContractSet."""
    spec_name: str
    results: list[ContractResult]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.FAIL)

    @property
    def overall(self) -> Verdict:
        return Verdict.PASS if self.failed == 0 else Verdict.FAIL

    def __str__(self) -> str:
        lines = [
            "",
            f"  AgentSpec Report — '{self.spec_name}'",
            "  " + "-" * 50,
        ]
        for result in self.results:
            lines.append(str(result))
        lines += [
            "  " + "-" * 50,
            f"  {self.passed} passed, {self.failed} failed",
            f"  Overall: {self.overall.value}",
            "",
        ]
        return "\n".join(lines)

    def assert_all_pass(self):
        """Raise AssertionError if any contract failed. Use in pytest."""
        if self.overall == Verdict.FAIL:
            failed_msgs = [r.message for r in self.results if r.verdict == Verdict.FAIL]
            raise AssertionError(
                f"AgentSpec '{self.spec_name}' failed {self.failed} contract(s):\n"
                + "\n".join(f"  - {m}" for m in failed_msgs)
            )


class ContractSet:
    """
    A collection of structural contracts for one agent or agent configuration.

    Contracts define WHAT the agent must do structurally — not how well it does it.
    Think of this as the behavioral interface of your agent, expressed as assertions.
    """

    def __init__(self, name: str):
        self.name = name
        # Each contract is stored as a (contract_name, checker_fn) tuple.
        # checker_fn: (AgentSession) -> ContractResult
        self._contracts: list[tuple[str, object]] = []

    # ------------------------------------------------------------------
    # Contract definition methods
    # ------------------------------------------------------------------

    def must_call(self, tool: str) -> "ContractSet":
        """Agent MUST call this tool at least once."""
        contract_name = f"must_call({tool})"

        def check(session: AgentSession) -> ContractResult:
            if session.calls(tool):
                return ContractResult(contract_name, Verdict.PASS, f"'{tool}' was called")
            return ContractResult(
                contract_name, Verdict.FAIL,
                f"'{tool}' was never called (calls made: {session.call_names()})"
            )

        self._contracts.append((contract_name, check))
        return self

    def must_not_call(self, tool: str) -> "ContractSet":
        """Agent MUST NOT call this tool."""
        contract_name = f"must_not_call({tool})"

        def check(session: AgentSession) -> ContractResult:
            calls = session.calls(tool)
            if not calls:
                return ContractResult(contract_name, Verdict.PASS, f"'{tool}' was correctly not called")
            return ContractResult(
                contract_name, Verdict.FAIL,
                f"'{tool}' was called {len(calls)} time(s) but must never be called"
            )

        self._contracts.append((contract_name, check))
        return self

    def must_call_before(self, first: str, second: str) -> "ContractSet":
        """
        Agent MUST call 'first' before 'second'.
        Both tools must be called; 'first' must appear earlier in the trace.
        """
        contract_name = f"must_call_before({first}, {second})"

        def check(session: AgentSession) -> ContractResult:
            idx_first = session.index_of_first(first)
            idx_second = session.index_of_first(second)

            if idx_first is None:
                return ContractResult(
                    contract_name, Verdict.FAIL,
                    f"'{first}' was never called (required before '{second}')"
                )
            if idx_second is None:
                return ContractResult(
                    contract_name, Verdict.FAIL,
                    f"'{second}' was never called (required after '{first}')"
                )
            if idx_first < idx_second:
                return ContractResult(
                    contract_name, Verdict.PASS,
                    f"'{first}' (step {idx_first}) correctly precedes '{second}' (step {idx_second})"
                )
            return ContractResult(
                contract_name, Verdict.FAIL,
                f"'{second}' (step {idx_second}) was called before '{first}' (step {idx_first})"
            )

        self._contracts.append((contract_name, check))
        return self

    def must_call_at_most(self, tool: str, n: int) -> "ContractSet":
        """Agent MUST NOT call this tool more than n times."""
        contract_name = f"must_call_at_most({tool}, n={n})"

        def check(session: AgentSession) -> ContractResult:
            count = len(session.calls(tool))
            if count <= n:
                return ContractResult(
                    contract_name, Verdict.PASS,
                    f"'{tool}' called {count} time(s) (limit: {n})"
                )
            return ContractResult(
                contract_name, Verdict.FAIL,
                f"'{tool}' called {count} time(s), exceeds limit of {n}"
            )

        self._contracts.append((contract_name, check))
        return self

    def must_call_at_least(self, tool: str, n: int) -> "ContractSet":
        """Agent MUST call this tool at least n times."""
        contract_name = f"must_call_at_least({tool}, n={n})"

        def check(session: AgentSession) -> ContractResult:
            count = len(session.calls(tool))
            if count >= n:
                return ContractResult(
                    contract_name, Verdict.PASS,
                    f"'{tool}' called {count} time(s) (minimum: {n})"
                )
            return ContractResult(
                contract_name, Verdict.FAIL,
                f"'{tool}' called only {count} time(s), need at least {n}"
            )

        self._contracts.append((contract_name, check))
        return self

    def must_not_call_after(self, forbidden: str, trigger: str) -> "ContractSet":
        """
        Once 'trigger' is called, 'forbidden' must not be called again.
        Useful for: "after generate_report is called, don't call search again."
        """
        contract_name = f"must_not_call_after({forbidden}, trigger={trigger})"

        def check(session: AgentSession) -> ContractResult:
            trigger_idx = session.index_of_first(trigger)
            if trigger_idx is None:
                return ContractResult(
                    contract_name, Verdict.PASS,
                    f"trigger '{trigger}' never called — constraint vacuously satisfied"
                )
            # Find any call to 'forbidden' after the trigger
            violations = [
                tc for i, tc in enumerate(session.tool_calls)
                if tc.name == forbidden and i > trigger_idx
            ]
            if not violations:
                return ContractResult(
                    contract_name, Verdict.PASS,
                    f"'{forbidden}' correctly not called after '{trigger}'"
                )
            return ContractResult(
                contract_name, Verdict.FAIL,
                f"'{forbidden}' called {len(violations)} time(s) after '{trigger}'"
            )

        self._contracts.append((contract_name, check))
        return self

    def must_call_in_sequence(self, *tools: str) -> "ContractSet":
        """
        All listed tools must be called and must appear in this exact relative order.
        Other tools may appear between them.
        """
        contract_name = f"must_call_in_sequence({', '.join(tools)})"

        def check(session: AgentSession) -> ContractResult:
            indices = []
            for tool in tools:
                idx = session.index_of_first(tool)
                if idx is None:
                    return ContractResult(
                        contract_name, Verdict.FAIL,
                        f"'{tool}' was never called (required in sequence: {list(tools)})"
                    )
                indices.append((tool, idx))

            # Check they appear in strictly increasing order
            for i in range(1, len(indices)):
                prev_tool, prev_idx = indices[i - 1]
                curr_tool, curr_idx = indices[i]
                if prev_idx >= curr_idx:
                    return ContractResult(
                        contract_name, Verdict.FAIL,
                        f"'{curr_tool}' (step {curr_idx}) did not follow '{prev_tool}' (step {prev_idx})"
                    )

            order_str = " → ".join(f"{t}[{i}]" for t, i in indices)
            return ContractResult(
                contract_name, Verdict.PASS,
                f"Sequence correct: {order_str}"
            )

        self._contracts.append((contract_name, check))
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def check(self, session: AgentSession) -> ContractReport:
        """Run all contracts against this session. Returns a full report."""
        results = [checker(session) for _, checker in self._contracts]
        return ContractReport(spec_name=self.name, results=results)
