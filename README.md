![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)
![pytest](https://img.shields.io/badge/tested%20with-pytest-yellow)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-73%20passed-brightgreen)
![PyPI](https://img.shields.io/pypi/v/llm-agentspec)
![zero dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)

# AgentSpec

**Behavioral contracts for LLM agents. Like pytest, but for agent behavior.**

```
pip install llm-agentspec
```

---

## What is this?

If you're building AI agents that call tools (search the web, read files, query databases, send emails), you need a way to make sure the agent is doing the right things in the right order.

AgentSpec lets you write simple rules like:

- "The agent must search before it summarizes"
- "The agent must never call the delete function"
- "The agent can search at most 3 times (to control costs)"

And then it checks whether your agent actually followed those rules. If it didn't, you get a clear pass/fail result that works in your test suite.

No AI needed to run the checks. No API keys. No cloud platform. Just pure Python that runs in microseconds.

---

## Why does this exist?

I kept seeing the same problem in AI engineering communities in 2026:

> "My agent summarized before it even searched. The output looked fine, but the process was completely wrong."

> "We added a tool that can delete user data. How do I make sure the agent never calls it?"

> "We're paying $0.03 per search call. The agent searched 47 times on one query. We need a hard limit."

Tools like DeepEval and RAGAS can tell you if the agent's **output** is good (not hallucinated, relevant, coherent). But they can't tell you if the agent **behaved correctly** -- did it call the right tools, in the right order, the right number of times?

That's what AgentSpec does. It checks the **behavior**, not the output.

Anthropic's own engineering blog ["Demystifying evals for AI agents"](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) makes this exact point: you need to verify that agents followed the right process, not just that they produced a plausible answer.

---

## Quick start

### 1. Install

```bash
pip install llm-agentspec
```

That's it. Zero dependencies. Takes about 1 second.

### 2. Write your first contract

```python
from agentspec import ContractSet, AgentSession, ToolCall

# Say you have a research agent that should:
# 1. Search the web first
# 2. Then summarize what it found
# 3. Never delete any files
# 4. Not search more than 5 times (cost control)

spec = ContractSet("research_agent")
spec.must_call("search")                          # Must search at least once
spec.must_call_before("search", "summarize")      # Search BEFORE summarizing
spec.must_not_call("delete_file")                 # Never call this
spec.must_call_at_most("search", n=5)             # Cost control
```

### 3. Check your agent's behavior

```python
# After your agent runs, you have a record of what tools it called.
# Wrap that in an AgentSession:

session = AgentSession(tool_calls=[
    ToolCall(name="search", args={"query": "latest AI research"}),
    ToolCall(name="search", args={"query": "transformer architectures"}),
    ToolCall(name="summarize", args={"format": "bullets"}),
])

# Check all contracts at once
report = spec.check(session)
print(report)
```

Output:
```
  AgentSpec Report - 'research_agent'
  --------------------------------------------------
  [✓] must_call(search): 'search' was called
  [✓] must_call_before(search, summarize): 'search' (step 0) correctly precedes 'summarize' (step 2)
  [✓] must_not_call(delete_file): 'delete_file' was correctly not called
  [✓] must_call_at_most(search, n=5): 'search' called 2 time(s) (limit: 5)
  --------------------------------------------------
  4 passed, 0 failed
  Overall: PASS
```

### 4. Use it in pytest

```python
# test_my_agent.py

def test_research_agent_follows_rules():
    spec = ContractSet("research_agent")
    spec.must_call("search")
    spec.must_call_before("search", "summarize")
    spec.must_not_call("delete_file")

    session = run_my_agent("Summarize recent AI news")  # your agent function
    
    report = spec.check(session)
    report.assert_all_pass()  # raises AssertionError if anything failed
```

Run it:
```bash
pytest test_my_agent.py -v
```

If the agent misbehaves, you get a clear error:
```
AssertionError: AgentSpec 'research_agent' failed 2 contract(s):
  - 'search' was never called
  - 'delete_file' was called 1 time(s) but must never be called
```

---

## Real-world examples

### Example 1: Customer support agent

A company builds a support agent that handles refund requests. The agent has access to `lookup_order`, `check_refund_policy`, `process_refund`, `respond_to_customer`, and `delete_account`.

```python
spec = ContractSet("support_agent")
spec.must_call("lookup_order")                                    # Always check the order first
spec.must_call("check_refund_policy")                             # Always check policy
spec.must_call_before("check_refund_policy", "process_refund")    # Policy check BEFORE processing
spec.must_call_before("lookup_order", "respond_to_customer")      # Look up order BEFORE responding
spec.must_not_call("delete_account")                              # Never delete accounts
spec.must_call_at_most("respond_to_customer", n=1)                # Only respond once
```

Why this matters: Without these contracts, the agent might approve a $10,000 refund without checking the policy, or accidentally call `delete_account` because the LLM hallucinated.

### Example 2: Code review agent

An engineering team builds an agent that reviews pull requests.

```python
spec = ContractSet("code_review_agent")
spec.must_call("fetch_diff")                                  # Must actually read the code
spec.must_call_before("fetch_diff", "post_review")            # Read code BEFORE reviewing
spec.must_not_call("merge_pr")                                # Review agent should NEVER merge
spec.must_call_at_most("call_llm", n=10)                      # Budget: max 10 LLM calls per PR
spec.must_call_in_sequence("fetch_diff", "analyze", "post_review")
```

### Example 3: Cost control

Your agent calls an expensive API at $0.03 per call. You need a hard limit.

```python
spec = ContractSet("my_agent")
spec.must_call_at_most("expensive_api", n=3)   # Max $0.09 per run

session = run_my_agent(user_query)
report = spec.check(session)

if report.overall.value == "FAIL":
    alert_engineering_team(report)
```

---

## All available contracts

| Contract | What it checks | Example |
|----------|---------------|---------|
| `must_call(tool)` | Tool was called at least once | `spec.must_call("search")` |
| `must_not_call(tool)` | Tool was never called | `spec.must_not_call("delete_file")` |
| `must_call_before(a, b)` | Tool `a` was called before tool `b` | `spec.must_call_before("search", "summarize")` |
| `must_call_at_most(tool, n)` | Tool was called no more than n times | `spec.must_call_at_most("search", n=5)` |
| `must_call_at_least(tool, n)` | Tool was called at least n times | `spec.must_call_at_least("search", n=2)` |
| `must_not_call_after(tool, trigger)` | After `trigger` is called, `tool` is never called again | `spec.must_not_call_after("search", "write_report")` |
| `must_call_in_sequence(*tools)` | All tools appear in this exact order | `spec.must_call_in_sequence("search", "analyze", "report")` |

All contracts can be chained:

```python
spec = (ContractSet("my_agent")
    .must_call("search")
    .must_call_before("search", "summarize")
    .must_not_call("delete")
    .must_call_at_most("search", n=5))
```

---

## How it works under the hood

```
Your agent runs
    |
    v
Tool calls are recorded as a list of ToolCall objects
    |
    v
You wrap them in an AgentSession
    |
    v
ContractSet.check(session) runs every contract against the session
    (pure Python -- no LLM calls, no network, no API keys)
    |
    v
Returns a ContractReport with pass/fail for each contract
    |
    v
report.assert_all_pass() -- use this in pytest
```

The key thing: **every check is deterministic.** If the tool was called, it was called. There's no LLM judging anything, no probability scores, no flakiness. Your CI pipeline gets a reliable green/red signal.

---

## How is this different from DeepEval, LangSmith, etc?

Honest comparison. I'm not going to pretend AgentSpec replaces these tools. They do different things.

| | AgentSpec | DeepEval | LangSmith | Promptfoo |
|---|---|---|---|---|
| **What it does** | Checks agent behavior (tool calls) | Checks output quality (hallucination, relevancy) | Traces + evaluates agent runs | General LLM evaluation |
| **Needs an LLM to run?** | No, never | Yes (LLM-as-judge) | Sometimes | Yes |
| **Needs an API key?** | No | Yes | Yes | Yes |
| **Speed** | 9 microseconds | ~100+ milliseconds | ~200+ milliseconds | ~500+ milliseconds |
| **Dependencies** | Zero | 26+ packages | Platform | Node.js + many |
| **Install size** | 15 KB | ~50+ MB | Platform | Heavy |
| **Works offline** | Yes | No | No | No |
| **Can score output quality** | No | Yes | Yes | Yes |
| **Has dashboards** | No | Yes | Yes | Yes |

**When to use AgentSpec:** You want fast, deterministic pass/fail checks on what tools your agent called. You want this in CI. You don't want to pay for LLM calls just to run tests.

**When to use DeepEval:** You want to know if the agent's output is good (not hallucinated, relevant, coherent). You want a cloud dashboard.

**Use both together.** AgentSpec checks "did the agent do the right thing?" DeepEval checks "was the output good?" Both questions matter.

---

## Performance

These aren't made-up numbers. Measured with [pytest-benchmark](https://github.com/ionelmc/pytest-benchmark), the industry standard for Python benchmarking. Auto-calibrated with 43,000 to 132,000 iterations per test.

| Session size | Median time | Operations per second |
|-------------|------------|----------------------|
| 5 tool calls | 3.38 us | 293,275 |
| 10 tool calls | 3.67 us | 270,259 |
| 25 tool calls | 4.54 us | 215,730 |
| 50 tool calls | 6.04 us | 162,579 |
| 100 tool calls | 9.04 us | 108,958 |

That's **microseconds**, not milliseconds. 7 contracts checked on a 100-call session in 9 microseconds.

### Stress tested

| Test | Result |
|------|--------|
| 100,000 sessions in a row | 0.34 seconds (296K ops/sec) |
| 5,000 tool calls in one session | 303 microseconds |
| 100 contracts on one session | 179 microseconds |
| 20 threads running simultaneously | 185K ops/sec, zero errors |
| 10,000 create-and-discard cycles | No memory leaks |

Run the benchmarks yourself:
```bash
pip install pytest-benchmark
pytest benchmarks/test_benchmark.py --benchmark-only
pytest benchmarks/test_stress.py -v -s
```

---

## Building an AgentSession from your agent framework

AgentSpec doesn't care which framework you use. You just need to convert your agent's tool calls into a list of `ToolCall` objects.

### From Anthropic Claude API (tool_use)

```python
from agentspec import AgentSession, ToolCall

# After a Claude API call, response.content contains tool_use blocks
tool_calls = []
for block in response.content:
    if block.type == "tool_use":
        tool_calls.append(ToolCall(
            name=block.name,
            args=block.input,
        ))

session = AgentSession(tool_calls=tool_calls)
```

### From LangChain / LangGraph

```python
# If you have a list of AgentAction objects
tool_calls = [
    ToolCall(name=action.tool, args=action.tool_input)
    for action in agent_actions
]
session = AgentSession(tool_calls=tool_calls)
```

### From OpenAI function calling

```python
# From ChatCompletion response with tool_calls
tool_calls = [
    ToolCall(name=tc.function.name, args=json.loads(tc.function.arguments))
    for tc in response.choices[0].message.tool_calls
]
session = AgentSession(tool_calls=tool_calls)
```

### Manually in tests

```python
session = AgentSession(tool_calls=[
    ToolCall(name="search", args={"query": "test"}),
    ToolCall(name="summarize", args={}),
])
```

---

## Running the demo

```bash
git clone https://github.com/rakshithmuda22/agentspec.git
cd agentspec
pip install -e ".[dev]"
python demo/demo.py
```

The demo runs without any API key. It shows a well-behaved agent that passes all contracts and a broken agent that fails multiple contracts, side by side.

---

## Tests

```bash
pytest tests/ -v
# 73 passed in 0.06s
```

All tests run in mock mode. No API keys needed. No network calls.

---

## Project structure

```
agentspec/
  agentspec/           # The pip-installable package
    __init__.py        # Public API: ContractSet, AgentSession, ToolCall, etc.
    contracts.py       # Contract engine (all 7 contract types)
    recorder.py        # Helper to build AgentSession from different sources
    reporter.py        # Formatted report output (uses Python logging)
    config.py          # Configuration (reads from environment variables)
    pytest_plugin.py   # pytest fixtures (contract_spec, agent_session, etc.)
    py.typed           # PEP 561 marker for type checker support
  tests/               # 73 tests covering all contracts and edge cases
  benchmarks/          # pytest-benchmark performance tests + stress tests
  demo/                # Interactive demo (no API key needed)
  pyproject.toml       # Package metadata and build config
  LICENSE              # MIT
```

---

## Technical decisions

**Why pure Python with zero dependencies?**
Every dependency is a potential security risk, a version conflict, and install-time overhead. AgentSpec uses only Python standard library (dataclasses, enum, typing, logging). When you `pip install llm-agentspec`, nothing else gets installed. That's on purpose.

**Why method calls instead of YAML config?**
YAML works for simple cases. But when you need conditional logic ("search must be called at most `budget / cost_per_call` times"), YAML breaks down. Python is a language engineers already know. And you get IDE autocomplete for free.

**Why first-occurrence semantics for ordering?**
`must_call_before("search", "summarize")` checks whether the FIRST call to "search" came before the FIRST call to "summarize". This catches the most common bug: the agent summarized before it searched. If you need different semantics, the library is designed to be extended.

---

## Roadmap

Things I want to add next:

1. **Framework adapters** - ready-made converters for Anthropic, OpenAI, LangChain, and OpenTelemetry traces so you don't have to build AgentSession manually
2. **Probabilistic contracts** - `must_call_on_average(tool, n=2, over=100_runs)` for contracts that should hold statistically
3. **JUnit XML output** - so contract results show up natively in GitHub Actions and Jenkins
4. **VS Code extension** - inline visualization of contract results

---

## Contributing

Contributions are welcome. If you want to add a new contract type, a framework adapter, or fix a bug:

1. Fork the repo
2. Create a branch
3. Write tests for your change
4. Make sure `pytest tests/ -v` passes
5. Open a PR

---

## License

MIT. Use it however you want.

---

Built by [Sai Rakshith Muda](https://github.com/rakshithmuda22) while building AI agents and getting frustrated that there was no simple way to test their behavior.
