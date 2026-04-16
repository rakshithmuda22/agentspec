"""Tests for the Anthropic adapter.

These tests use recorded-fixture style dicts shaped exactly like the
Anthropic Python SDK's `Message.model_dump()` output. That means the
same test works for live SDK objects AND for JSON fixtures committed
to CI, which is the whole point of the adapter (no SDK dep at import
time, no network at test time).
"""
from __future__ import annotations

from agentspec import ContractSet
from agentspec.adapters.anthropic import (
    from_anthropic_messages,
    from_anthropic_response,
)


def _text(s: str) -> dict:
    return {"type": "text", "text": s}


def _tool(name: str, args: dict, tu_id: str = "tu_1") -> dict:
    return {"type": "tool_use", "id": tu_id, "name": name, "input": args}


# ── from_anthropic_response ─────────────────────────────────────────


def test_extracts_single_tool_use():
    response = {
        "id": "msg_1",
        "role": "assistant",
        "content": [
            _text("I'll search for that."),
            _tool("search", {"query": "python testing"}),
        ],
    }
    session = from_anthropic_response(response)
    assert session.call_names() == ["search"]
    assert session.tool_calls[0].args == {"query": "python testing"}


def test_extracts_multiple_tool_uses_in_order():
    response = {
        "role": "assistant",
        "content": [
            _tool("search", {"q": "a"}, "tu_1"),
            _tool("search", {"q": "b"}, "tu_2"),
            _tool("summarize", {}, "tu_3"),
        ],
    }
    session = from_anthropic_response(response)
    assert session.call_names() == ["search", "search", "summarize"]


def test_ignores_text_only_response():
    response = {
        "role": "assistant",
        "content": [_text("No tool needed."), _text("Here is my answer.")],
    }
    session = from_anthropic_response(response)
    assert session.call_names() == []


def test_tolerates_empty_content():
    session = from_anthropic_response({"role": "assistant", "content": []})
    assert session.call_names() == []


def test_tolerates_missing_content():
    session = from_anthropic_response({"role": "assistant"})
    assert session.call_names() == []


def test_tool_with_no_input_becomes_empty_args():
    response = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "tu_1", "name": "ping"}],
    }
    session = from_anthropic_response(response)
    assert session.tool_calls[0].args == {}


def test_skips_tool_block_with_no_name():
    response = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "tu_1", "input": {"x": 1}}],
    }
    session = from_anthropic_response(response)
    assert session.call_names() == []


# ── SDK-object compatibility (duck-typed model_dump) ───────────────


class _FakeSDKBlock:
    """Mimics an Anthropic content block object (has model_dump)."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def model_dump(self) -> dict:
        return self._data


class _FakeSDKMessage:
    """Mimics an Anthropic Message object."""

    def __init__(self, content: list[dict]) -> None:
        self._content = [_FakeSDKBlock(b) for b in content]

    def model_dump(self) -> dict:
        return {"role": "assistant", "content": [b.model_dump() for b in self._content]}


def test_works_with_sdk_style_object():
    sdk_msg = _FakeSDKMessage([_tool("search", {"q": "test"})])
    session = from_anthropic_response(sdk_msg)
    assert session.call_names() == ["search"]


# ── from_anthropic_messages ────────────────────────────────────────


def test_conversation_captures_tool_uses_across_turns():
    messages = [
        {"role": "user", "content": "search for python docs"},
        {
            "role": "assistant",
            "content": [_tool("search", {"q": "python"}, "tu_1")],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}
            ],
        },
        {
            "role": "assistant",
            "content": [
                _tool("summarize", {}, "tu_2"),
                _text("Here's the summary."),
            ],
        },
    ]
    session = from_anthropic_messages(messages)
    assert session.call_names() == ["search", "summarize"]


def test_conversation_ignores_user_messages():
    messages = [
        {"role": "user", "content": "do something"},
        {"role": "user", "content": "again"},
    ]
    session = from_anthropic_messages(messages)
    assert session.call_names() == []


# ── End-to-end contract check against adapter output ───────────────


def test_contract_check_against_adapter_output():
    """The point of the adapter: feed a raw Anthropic response to a
    ContractSet without any manual session building."""
    response = {
        "role": "assistant",
        "content": [
            _tool("search", {"q": "docs"}, "tu_1"),
            _tool("search", {"q": "more docs"}, "tu_2"),
            _tool("summarize", {}, "tu_3"),
            _tool("write_report", {}, "tu_4"),
        ],
    }

    spec = (
        ContractSet("research_agent")
        .must_call("search")
        .must_call_before("search", "summarize")
        .must_call_before("summarize", "write_report")
        .must_not_call("delete_file")
        .must_call_at_most("search", n=3)
    )

    report = spec.check(from_anthropic_response(response))
    report.assert_all_pass()


def test_contract_catches_bad_adapter_output():
    """A broken agent: summarizes before searching. Contract should fail."""
    response = {
        "role": "assistant",
        "content": [
            _tool("summarize", {}, "tu_1"),  # WRONG: before search
            _tool("search", {"q": "docs"}, "tu_2"),
        ],
    }

    spec = ContractSet("research_agent").must_call_before("search", "summarize")
    report = spec.check(from_anthropic_response(response))
    assert report.failed == 1
    assert report.passed == 0
