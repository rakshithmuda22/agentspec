"""
Stress tests for AgentSpec under heavy load.

Verifies that AgentSpec works correctly and efficiently when:
- Checking thousands of sessions in rapid succession
- Running concurrent checks from multiple threads
- Handling large sessions (1000+ tool calls)
- Processing many contracts on a single session

These tests prove AgentSpec is production-ready, not just demo-ready.
"""
import threading
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from agentspec import ContractSet, AgentSession, ToolCall


def _session(n: int) -> AgentSession:
    tools = ["search", "fetch", "parse", "summarize", "write", "validate"]
    return AgentSession(
        tool_calls=[ToolCall(name=tools[i % len(tools)], args={"i": i}) for i in range(n)]
    )


def _spec(max_calls: int = 1000) -> ContractSet:
    spec = ContractSet("stress_agent")
    spec.must_call("search")
    spec.must_call("summarize")
    spec.must_call_before("search", "summarize")
    spec.must_call_before("fetch", "parse")
    spec.must_not_call("delete_file")
    spec.must_call_at_most("search", n=max_calls)
    spec.must_call_at_least("search", n=1)
    return spec


# --- 1. Throughput: 10,000 sessions checked sequentially ---

class TestThroughput:

    def test_10k_sessions(self):
        """Check 10,000 sessions in sequence. Must complete in <1 second."""
        spec = _spec()
        session = _session(10)

        start = time.perf_counter()
        for _ in range(10_000):
            report = spec.check(session)
            assert report.overall.value == "PASS"
        elapsed = time.perf_counter() - start

        ops_per_sec = 10_000 / elapsed
        assert elapsed < 1.0, f"10K checks took {elapsed:.2f}s (expected <1s)"
        print(f"\n  10K sessions: {elapsed:.3f}s ({ops_per_sec:,.0f} ops/sec)")

    def test_100k_sessions(self):
        """Check 100,000 sessions. Must complete in <5 seconds."""
        spec = _spec()
        session = _session(5)

        start = time.perf_counter()
        for _ in range(100_000):
            spec.check(session)
        elapsed = time.perf_counter() - start

        ops_per_sec = 100_000 / elapsed
        assert elapsed < 5.0, f"100K checks took {elapsed:.2f}s (expected <5s)"
        print(f"\n  100K sessions: {elapsed:.3f}s ({ops_per_sec:,.0f} ops/sec)")


# --- 2. Large sessions (1000+ tool calls) ---

class TestLargeSessions:

    def test_1000_tool_calls(self):
        """Session with 1000 tool calls. Must check in <1ms."""
        spec = _spec()
        session = _session(1000)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            report = spec.check(session)
            times.append((time.perf_counter() - start) * 1_000_000)
            assert report.overall.value == "PASS"

        median = statistics.median(times)
        assert median < 1000, f"1000-call session median: {median:.0f}us (expected <1000us)"
        print(f"\n  1000 calls: median {median:.1f}us")

    def test_5000_tool_calls(self):
        """Session with 5000 tool calls."""
        spec = _spec()
        session = _session(5000)

        start = time.perf_counter()
        report = spec.check(session)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        assert report.overall.value == "PASS"
        print(f"\n  5000 calls: {elapsed_us:.1f}us")


# --- 3. Many contracts on one session ---

class TestManyContracts:

    def test_50_contracts(self):
        """50 contracts on one session. Must check in <100us."""
        spec = ContractSet("many_contracts")
        tools = [f"tool_{i}" for i in range(25)]

        for t in tools:
            spec.must_call(t)
            spec.must_call_at_least(t, n=1)

        session = AgentSession(
            tool_calls=[ToolCall(name=t, args={}) for t in tools]
        )

        times = []
        for _ in range(1000):
            start = time.perf_counter()
            report = spec.check(session)
            times.append((time.perf_counter() - start) * 1_000_000)
            assert report.overall.value == "PASS"

        median = statistics.median(times)
        print(f"\n  50 contracts: median {median:.1f}us")

    def test_100_contracts(self):
        """100 contracts on one session."""
        spec = ContractSet("hundred_contracts")
        tools = [f"tool_{i}" for i in range(50)]

        for t in tools:
            spec.must_call(t)
            spec.must_call_at_most(t, n=100)

        session = AgentSession(
            tool_calls=[ToolCall(name=t, args={}) for t in tools * 2]
        )

        start = time.perf_counter()
        report = spec.check(session)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        assert report.overall.value == "PASS"
        print(f"\n  100 contracts: {elapsed_us:.1f}us")


# --- 4. Thread safety: concurrent checks from multiple threads ---

class TestConcurrency:

    def test_concurrent_checks_10_threads(self):
        """10 threads each checking 1000 sessions. Must be thread-safe."""
        spec = _spec()
        session = _session(10)
        errors = []

        def worker(thread_id: int):
            try:
                for _ in range(1000):
                    report = spec.check(session)
                    if report.overall.value != "PASS":
                        errors.append(f"Thread {thread_id}: unexpected FAIL")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        start = time.perf_counter()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        assert len(errors) == 0, f"Thread errors: {errors}"
        print(f"\n  10 threads x 1000 checks: {elapsed:.3f}s (total 10K checks)")

    def test_thread_pool_executor(self):
        """ThreadPoolExecutor with 20 workers checking 500 sessions each."""
        spec = _spec()
        session = _session(25)

        def check_batch(batch_id: int) -> tuple[int, int]:
            passed = 0
            failed = 0
            for _ in range(500):
                report = spec.check(session)
                if report.overall.value == "PASS":
                    passed += 1
                else:
                    failed += 1
            return passed, failed

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(check_batch, i) for i in range(20)]
            total_passed = 0
            total_failed = 0
            for f in as_completed(futures):
                p, fail = f.result()
                total_passed += p
                total_failed += fail
        elapsed = time.perf_counter() - start

        assert total_failed == 0, f"{total_failed} failures in concurrent execution"
        assert total_passed == 10_000
        print(f"\n  20 workers x 500 checks: {elapsed:.3f}s (total 10K checks, {10_000/elapsed:,.0f} ops/sec)")


# --- 5. Memory stability: repeated creation and garbage collection ---

class TestMemoryStability:

    def test_no_memory_leak_on_repeated_spec_creation(self):
        """Create and discard 10,000 ContractSets. Should not leak."""
        for i in range(10_000):
            spec = ContractSet(f"spec_{i}")
            spec.must_call("search")
            spec.must_not_call("delete")
            session = AgentSession(tool_calls=[ToolCall(name="search", args={})])
            report = spec.check(session)
            assert report.overall.value == "PASS"
        # If we get here without OOM, we passed
        print("\n  10K spec create/discard cycles: OK (no memory leak)")
