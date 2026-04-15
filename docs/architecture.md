# AgentSpec — Architecture

## Core Concept

AgentSpec separates two things that AI engineers usually conflate:

| | What it checks | Pass/fail | Example tool |
|--|--|--|--|
| **Quality evals** | Is the output good? | LLM-graded | DeepEval, RAGAS |
| **Structural contracts** (AgentSpec) | Did the agent behave correctly? | Deterministic | **AgentSpec** |

Quality evals tell you if the *answer* was good. Structural contracts tell you if the *process* was correct. You need both.

```
Agent run
    │
    ▼
AgentSession (ordered list of ToolCalls)
    │
    ▼
ContractSet.check(session)
    │ runs each contract checker function
    ▼
ContractReport
    │
    ├── N passed
    ├── M failed
    ├── Overall: PASS or FAIL
    └── report.assert_all_pass()  → raises AssertionError in pytest
```

## Key Design Decisions

### 1. Pure Python, no platform dependency
Each contract is a function `(AgentSession) -> ContractResult`. No LLM calls. No external services.
This means: contracts run in CI in milliseconds, offline, with no API rate limits.

### 2. Contracts as composable method calls, not YAML
```python
spec.must_call("search")
spec.must_call_before("search", "summarize")
spec.must_not_call("delete_file")
```
This is more readable than YAML and more debuggable than a DSL.
Engineers already know Python. They don't need to learn another config format.

### 3. AgentSession is framework-agnostic
`AgentSession` is just a list of `ToolCall` objects. You can build it from:
- Anthropic tool_use blocks (live mode, see `agentspec/recorder.py`)
- LangSmith traces (convert the trace to ToolCall objects)
- OpenTelemetry spans (extract tool names from span attributes)
- Manually in tests (the most common case in CI)

The recorder in `agentspec/recorder.py` handles the Anthropic API case.
For other frameworks, you write a thin adapter that populates `AgentSession`.

### 4. report.assert_all_pass() bridges AgentSpec to pytest
This one method means AgentSpec drops into any existing pytest suite with zero friction:
```python
def test_my_agent():
    session = run_my_agent("task")
    spec.check(session).assert_all_pass()
```

## What This Is NOT

- Not a quality eval framework (use DeepEval/RAGAS for that)
- Not an observability platform (use Langfuse/Datadog for that)
- Not a simulation/sandbox (the agent needs to actually run)

AgentSpec sits in the gap: after the agent runs, before you ship.
