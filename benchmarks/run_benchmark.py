"""
Benchmark AgentSpec contract checking performance.

Measures p50, p95, p99, and mean execution times for contract checking
across 1000 runs with varying session sizes.

No API keys needed. Pure CPU benchmark.
"""
import sys
import time
import statistics

sys.path.insert(0, ".")
from agentspec import ContractSet, AgentSession, ToolCall


def build_session(n_calls: int) -> AgentSession:
    """Build a session with n tool calls."""
    tools = ["search", "fetch", "parse", "summarize", "write", "validate"]
    calls = [
        ToolCall(name=tools[i % len(tools)], args={"step": i}, result=f"result_{i}")
        for i in range(n_calls)
    ]
    return AgentSession(tool_calls=calls)


def build_spec() -> ContractSet:
    """Build a realistic contract set with 7 contracts."""
    spec = ContractSet("benchmark_agent")
    spec.must_call("search")
    spec.must_call("summarize")
    spec.must_call_before("search", "summarize")
    spec.must_call_before("fetch", "parse")
    spec.must_not_call("delete_file")
    spec.must_call_at_most("search", n=10)
    spec.must_call_at_least("search", n=1)
    return spec


def run_benchmark(n_runs: int = 1000, session_sizes: list[int] = None):
    if session_sizes is None:
        session_sizes = [5, 10, 25, 50, 100]

    print("=" * 65)
    print("AgentSpec Benchmark")
    print(f"Runs per config: {n_runs}")
    print(f"Contracts per spec: 7")
    print("=" * 65)
    print()

    for size in session_sizes:
        session = build_session(size)
        spec = build_spec()

        # Warmup
        for _ in range(10):
            spec.check(session)

        # Timed runs
        times_us = []
        for _ in range(n_runs):
            start = time.perf_counter()
            report = spec.check(session)
            elapsed = (time.perf_counter() - start) * 1_000_000  # microseconds
            times_us.append(elapsed)

        times_us.sort()
        p50 = times_us[n_runs // 2]
        p95 = times_us[int(n_runs * 0.95)]
        p99 = times_us[int(n_runs * 0.99)]
        mean = statistics.mean(times_us)
        std = statistics.stdev(times_us)

        print(f"Session size: {size:>3d} tool calls | "
              f"p50: {p50:>8.1f} us | "
              f"p95: {p95:>8.1f} us | "
              f"p99: {p99:>8.1f} us | "
              f"mean: {mean:>8.1f} us +/- {std:.1f}")

    # Also measure assert_all_pass overhead
    session = build_session(25)
    spec = build_spec()
    times_us = []
    for _ in range(n_runs):
        start = time.perf_counter()
        report = spec.check(session)
        report.assert_all_pass()
        elapsed = (time.perf_counter() - start) * 1_000_000
        times_us.append(elapsed)

    times_us.sort()
    p50 = times_us[n_runs // 2]
    p95 = times_us[int(n_runs * 0.95)]

    print()
    print(f"With assert_all_pass()  | "
          f"p50: {p50:>8.1f} us | p95: {p95:>8.1f} us")

    print()
    print("-" * 65)
    print("All times in microseconds (us). 1000 us = 1 ms.")
    print("7 contracts checked per run. No LLM calls. No network.")
    print(f"Measured on: {sys.platform}")
    print("-" * 65)


if __name__ == "__main__":
    run_benchmark(n_runs=1000)
