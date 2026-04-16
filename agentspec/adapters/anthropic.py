"""
agentspec/adapters/anthropic.py — Convert Anthropic API responses into AgentSession.

The Anthropic SDK is optional. This adapter does not import it — it works on
the response dicts / objects that the SDK returns, so you can use AgentSpec
against recorded fixtures in CI without carrying a provider dependency.

Two entry points:

1. `from_anthropic_response(response)` — a single `Message` object (or its
   `model_dump()` dict) from a one-shot `client.messages.create(...)` call.

2. `from_anthropic_messages(messages)` — a full conversation list of
   `{"role": "assistant", "content": [...]}` dicts, such as what you'd
   send back through `client.messages.create(messages=...)` during a
   tool-use loop. Every `tool_use` block across every assistant turn
   is captured in order.

Example:
    from anthropic import Anthropic
    from agentspec import ContractSet
    from agentspec.adapters.anthropic import from_anthropic_response

    client = Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        tools=[...],
        messages=[{"role": "user", "content": "find docs about X"}],
    )

    session = from_anthropic_response(resp)

    spec = (ContractSet("docs_agent")
        .must_call("search")
        .must_not_call("delete_file"))

    report = spec.check(session)
    report.assert_all_pass()
"""
from __future__ import annotations

from typing import Any, Iterable

from agentspec.contracts import AgentSession, ToolCall


def _coerce(obj: Any) -> Any:
    """Return a JSON-like dict for Anthropic SDK objects or pass dicts through.

    The SDK returns Pydantic models that expose .model_dump() or .dict().
    We accept either so tests can use plain fixtures.
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _iter_content_blocks(message: Any) -> Iterable[dict]:
    """Yield content blocks from a Message, whether SDK object or dict."""
    msg = _coerce(message)
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    if not content:
        return
    for block in content:
        b = _coerce(block)
        if isinstance(b, dict):
            yield b


def _extract_tool_uses(message: Any) -> list[ToolCall]:
    """Pull every tool_use block out of a Message in order."""
    calls: list[ToolCall] = []
    for block in _iter_content_blocks(message):
        if block.get("type") != "tool_use":
            continue
        name = block.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = block.get("input")
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(name=name, args=args))
    return calls


def from_anthropic_response(response: Any, spec_name: str = "agent") -> AgentSession:
    """Convert a single Anthropic `Message` response into an AgentSession.

    Args:
        response: A `Message` object from `client.messages.create(...)`, or
            its `model_dump()` dict. Supports both the live SDK and recorded
            JSON fixtures.
        spec_name: Name tag applied to the session for reporting.

    Returns:
        An `AgentSession` with one `ToolCall` per `tool_use` block found in
        `response.content`, preserving order.
    """
    return AgentSession(tool_calls=_extract_tool_uses(response))


def from_anthropic_messages(messages: Iterable[Any], spec_name: str = "agent") -> AgentSession:
    """Convert a full conversation history into an AgentSession.

    Used when you're running a multi-turn tool-use loop and want to capture
    every tool call the model made across every assistant turn.

    Args:
        messages: An iterable of message dicts. Non-assistant messages (and
            assistant messages without tool_use blocks) are ignored.
        spec_name: Name tag applied to the session for reporting.

    Returns:
        An `AgentSession` containing every `tool_use` block in chronological
        order across all assistant messages.
    """
    calls: list[ToolCall] = []
    for msg in messages:
        m = _coerce(msg)
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "assistant":
            continue
        calls.extend(_extract_tool_uses(m))
    return AgentSession(tool_calls=calls)
