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

## Comparison (honest)

| | AgentSpec | DeepEval ToolCorrectness | LangChain AgentEvals | Promptfoo |
|---|---|---|---|---|
| **What it does** | Structural tool-call contracts | LLM output quality + tool correctness | Trajectory matching | General LLM eval |
| Tool ordering assertions | Yes (7 contract types) | Yes (should_consider_ordering) | Yes (strict mode) | No |
| Deterministic (no LLM needed) | **Yes - always** | Partial (uses LLM for optimization) | Partial (LLM-as-judge option) | No (LLM-based) |
| Dependencies | **0** | 26+ (openai, grpcio, sentry, etc.) | LangSmith platform | Node.js + many |
| Install size | **15 KB** | ~50 MB+ with deps | Platform-dependent | Heavy |
| Speed (7 contracts) | **9 us median** | ~100ms+ (async, LLM calls) | ~200ms+ (API calls) | ~500ms+ |
| Throughput | **296K ops/sec** | Hundreds/sec | Hundreds/sec | Tens/sec |
| Thread-safe | **Yes (tested with 20 threads)** | Not documented | Not documented | N/A |
| pytest native | **Yes (assert_all_pass)** | Yes (deepeval test) | No (SDK-based) | No (CLI-based) |
| Works offline | **Yes** | No (needs API key) | No (needs LangSmith) | No |
| Framework-agnostic | **Yes** | Partial | LangChain-first | Partial |
| Output quality scoring | No | **Yes** | **Yes** | **Yes** |
| Dashboards / monitoring | No | **Yes** | **Yes** | **Yes** |

**When to use AgentSpec:** You need fast, deterministic, zero-dependency assertions on tool-call behavior in CI/CD. You want pass/fail, not scores.

**When to use DeepEval instead:** You need LLM-graded output quality (hallucination, relevancy, coherence). You want a cloud dashboard.

**They are complementary, not competing.** Use AgentSpec for structural contracts in CI (did the agent call the right tools in the right order?) and DeepEval for quality evaluation (was the output good?).

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

Measured with [pytest-benchmark](https://github.com/ionelmc/pytest-benchmark) (industry standard). 7 contracts per run. Auto-calibrated rounds (43K-132K iterations). Apple Silicon.

| Session Size | Min | Median | Mean | Rounds | Ops/sec |
|-------------|-----|--------|------|--------|---------|
| 5 tool calls | 3.17 us | 3.38 us | 3.41 us | 43,165 | 293,275 |
| 10 tool calls | 3.42 us | 3.67 us | 3.70 us | 132,591 | 270,259 |
| 25 tool calls | 4.29 us | 4.54 us | 4.64 us | 123,077 | 215,730 |
| 50 tool calls | 5.79 us | 6.04 us | 6.15 us | 103,008 | 162,579 |
| 100 tool calls | 8.75 us | 9.04 us | 9.18 us | 72,506 | 108,958 |

**That's microseconds, not milliseconds.** 7 contracts on a 100-call session in 9 microseconds. 100K+ operations per second. No LLM calls. No network. Zero dependencies.

### Stress Tests

| Test | Result |
|------|--------|
| 10,000 sessions sequentially | 0.042s (**240K ops/sec**) |
| 100,000 sessions sequentially | 0.337s (**296K ops/sec**) |
| 1,000 tool calls per session | 63.5 us median |
| 5,000 tool calls per session | 303 us |
| 50 contracts on one session | 35.3 us median |
| 100 contracts on one session | 179 us |
| 10 threads x 1,000 checks each | 0.042s (thread-safe) |
| 20 threads x 500 checks each | 0.054s (**185K ops/sec concurrent**) |
| 10,000 spec create/discard cycles | No memory leak |

Run all benchmarks yourself:
```bash
pip install pytest-benchmark
pytest benchmarks/test_benchmark.py --benchmark-only   # precision benchmarks
pytest benchmarks/test_stress.py -v -s                  # stress tests
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
