![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![pytest](https://img.shields.io/badge/tested%20with-pytest-yellow)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![tests passing](https://img.shields.io/badge/tests-48%20passed-brightgreen)

# AgentSpec — pytest-style structural contracts for LLM agents

## Why I Built This

Anthropic's "Demystifying evals for AI agents" blog makes the case that agent evaluation
needs to go beyond output quality — you need to verify that agents followed the right process,
not just that they produced a plausible answer. LangChain's 2026 State of AI Agents report
found that 32% of teams cite output quality as their top barrier to production. But when I
looked at the tooling landscape, every framework was solving the *quality* half of the problem
(is the output good?) while ignoring the *behavioral* half (did the agent do the right thing?).
AgentSpec exists to close that gap: deterministic, structural assertions about tool-call
behavior that run in CI and give you a reliable pass/fail signal.

## The Problem

AI engineers in 2026 can tell you if their agent's output is *good* (DeepEval, RAGAS, LangSmith
all do this). What they cannot easily tell you is whether their agent *behaved correctly* — in a
deterministic, pass/fail way.

The same questions come up every week in engineering teams:
- "Did the agent call the search tool before summarizing, or did it summarize thin air again?"
- "We hard-limit search to 3 calls for cost control — did this deployment exceed that?"
- "Someone added a tool that can delete data. Does our agent ever call it in production?"
- "After the report was written, did the agent keep searching? (It shouldn't.)"

These are not quality questions. They are structural questions. The answers are deterministic —
either the tool was called, or it wasn't. But no existing tool gives engineers a simple way
to express these assertions and run them in CI.

DeepEval grades LLM output quality (hallucination, relevancy, coherence). LangSmith traces
what happened. Maxim AI is a full observability platform. None of them are designed for writing
5-line behavioral contracts that run offline in 50 milliseconds.

## What This Solves

AgentSpec is a pip-installable Python library for writing structural contracts about LLM agent
tool-call behavior. Think of it as pytest, but for agent behavior.

```python
from core.contracts import ContractSet

spec = ContractSet("research_agent")
spec.must_call("search")                          # Must search at least once
spec.must_call_before("search", "summarize")      # Must search BEFORE summarizing
spec.must_not_call("delete_file")                 # Must NEVER call delete_file
spec.must_call_at_most("search", n=3)             # Cost control: max 3 searches
spec.must_call_in_sequence("search", "summarize", "write_report")

report = spec.check(my_agent_session)
report.assert_all_pass()   # Raises AssertionError in pytest if anything fails
```

Output when a broken agent runs:
```
  [✓] must_call(search): 'search' was called
  [✗] must_call_before(search, summarize): 'summarize' (step 0) was called before 'search' (step 1)
  [✗] must_not_call(delete_file): 'delete_file' was called 1 time(s) but must never be called
  [✗] must_call_at_most(search, n=3): 'search' called 4 time(s), exceeds limit of 3

  1 passed / 3 failed — FAIL
```

## How It Works

```
Agent runs → tool calls are recorded as AgentSession
                    │
                    ▼
             ContractSet.check(session)
             (pure Python, no LLM calls, no network)
                    │
                    ▼
             ContractReport → report.assert_all_pass()
```

**3 key decisions:**
- **Pure Python, no platform** — each contract is a function `(AgentSession) -> ContractResult`. Runs in CI in milliseconds. No API key. No rate limits. No external service.
- **Framework-agnostic `AgentSession`** — build it from Anthropic tool_use blocks, LangSmith traces, OpenTelemetry spans, or manually in tests. AgentSpec doesn't care where the trace came from.
- **`report.assert_all_pass()` bridges to pytest** — one method, zero friction. Drop AgentSpec into any existing pytest suite with no changes to your test infrastructure.

## Install

```bash
pip install agentspec
```

Or from source:
```bash
git clone https://github.com/rakshithmuda22/agentspec.git
cd agentspec
pip install -e ".[dev]"
python demo/demo.py
```

Demo runs without an API key — shows a well-behaved agent (7/7 contracts pass) and a
broken agent (6/7 contracts fail) side by side.

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from core.contracts import ContractSet, AgentSession, ToolCall

# Define behavioral contracts for your agent
spec = ContractSet("my_agent")
spec.must_call("search")
spec.must_call_before("search", "summarize")
spec.must_not_call("delete_file")

# Build a session from your agent's tool calls
session = AgentSession(tool_calls=[
    ToolCall(name="search", args={"q": "latest news"}),
    ToolCall(name="summarize", args={"text": "..."}),
])

# Check contracts — deterministic, no LLM needed
report = spec.check(session)
report.assert_all_pass()  # Use in pytest — raises AssertionError on failure
```

## Contracts Available

| Method | What it checks |
|--------|----------------|
| `must_call(tool)` | Tool called at least once |
| `must_not_call(tool)` | Tool never called |
| `must_call_before(a, b)` | Tool `a` appears before `b` in trace |
| `must_call_at_most(tool, n)` | Tool called ≤ n times |
| `must_call_at_least(tool, n)` | Tool called ≥ n times |
| `must_not_call_after(tool, trigger)` | After `trigger`, `tool` never called again |
| `must_call_in_sequence(*tools)` | All tools appear in this order |

## Comparison

| Capability | AgentSpec | DeepEval | LangSmith | Maxim AI |
|---|---|---|---|---|
| Output quality scoring (hallucination, relevancy) | -- | Yes | Yes | Yes |
| Structural tool-call assertions (must_call, ordering) | **Yes** | -- | -- | -- |
| Deterministic pass/fail (no LLM grader) | **Yes** | -- | Partial | -- |
| Runs offline, no API key needed | **Yes** | -- | -- | -- |
| Sub-100ms execution | **Yes** | -- | -- | -- |
| pytest integration (assert_all_pass) | **Yes** | Yes | -- | -- |
| Trace visualization & observability | -- | -- | Yes | Yes |
| Production monitoring & dashboards | -- | -- | Yes | Yes |
| CI/CD gate ready | **Yes** | Yes | Partial | Partial |
| Framework-agnostic (works with any agent) | **Yes** | Partial | LangChain-first | Partial |

**AgentSpec does not replace these tools.** It fills a gap none of them cover: fast, deterministic,
structural assertions about *what the agent did* (not how good the output was). Use DeepEval for
quality scoring, LangSmith for observability, and AgentSpec for behavioral contracts in CI.

## Technical Decisions

**1. Deterministic, not probabilistic**
This is not an eval framework. There's no LLM grading anything. The pass/fail verdict is 100% deterministic — if the tool was called, it was called. This makes AgentSpec useful in CI/CD gates where you need reliable green/red signals.

**2. Contracts as Python method calls, not YAML or DSL**
The alternative was a YAML-based contract spec. YAML is fine for simple cases but breaks down when you need conditional logic or dynamic values (like "search must be called at most `cost_limit / 0.01` times"). Python handles this naturally. Engineers already know it.

**3. First-occurrence semantics for ordering contracts**
`must_call_before("search", "summarize")` checks the FIRST occurrence of each tool.
This is the most useful semantic for catching "agent summarized before searching" bugs.
A "last occurrence" variant could be added but wasn't needed for the core use cases.

## Performance

Benchmarked on Apple Silicon (M-series). 7 contracts checked per run, 1000 iterations:

| Session Size | p50 | p95 | p99 |
|-------------|-----|-----|-----|
| 5 tool calls | 3.7 us | 4.2 us | 16.6 us |
| 10 tool calls | 4.0 us | 4.5 us | 17.8 us |
| 25 tool calls | 5.0 us | 12.6 us | 15.8 us |
| 50 tool calls | 6.6 us | 8.1 us | 29.0 us |
| 100 tool calls | 10.2 us | 11.0 us | 12.5 us |

**That's microseconds, not milliseconds.** 7 contracts on a 100-call session in under 11 microseconds. No LLM calls. No network. Pure Python.

Run the benchmark yourself:
```bash
python benchmarks/run_benchmark.py
```

## Tests

```bash
pytest tests/ -v
# 73 passed in 0.06s
```

## Built With

- **Python 3.11** — core language
- **Pydantic** — data validation for `AgentSession` and `ToolCall` models
- **pytest** — test runner and assertion integration via `report.assert_all_pass()`
- **Anthropic SDK** — optional adapter for building sessions from Claude tool_use blocks
- **GitHub Actions** — CI pipeline (`.github/workflows/ci.yml`)

## What's Missing / What's Next

1. **Trace adapters** — currently you have to build `AgentSession` manually or use the Anthropic adapter. Ready-made adapters for LangSmith, LangChain callbacks, and OpenTelemetry would make this drop-in for any team.

2. **Probabilistic contracts** — `must_call_on_average(tool, n=2, over=100_runs)` for contracts that hold statistically but not deterministically. This would require running the agent multiple times and aggregating results.

3. **CI report format** — JUnit XML output so the results appear natively in GitHub Actions, CircleCI, and Jenkins test reports alongside regular pytest output.
