"""
core/recorder.py — Runs a Claude agent and records its tool calls.

Two modes:
  live: calls the real Claude API with your tools defined, records what it calls
  mock: returns pre-defined sample sessions (no API key needed)

The recorder produces an AgentSession that you then check against a ContractSet.
"""
from __future__ import annotations
from typing import Any, Callable

from agentspec.contracts import AgentSession, ToolCall
from agentspec import config


# ---------------------------------------------------------------------------
# Mock sessions — realistic agent traces for demo and testing
# ---------------------------------------------------------------------------

def make_research_agent_session() -> AgentSession:
    """A well-behaved research agent: searches first, then summarizes."""
    return AgentSession(
        tool_calls=[
            ToolCall(name="search", args={"query": "India AI policy 2026"}, result="[search results...]", step=0),
            ToolCall(name="search", args={"query": "NITI Aayog AI roadmap"}, result="[search results...]", step=1),
            ToolCall(name="summarize", args={}, result="India launched...", step=2),
            ToolCall(name="write_report", args={"title": "AI Policy Summary"}, result="Report written", step=3),
        ],
        metadata={"scenario": "research_agent_good", "model": "mock"}
    )


def make_broken_research_agent_session() -> AgentSession:
    """A broken agent: summarizes before searching, and calls delete_file."""
    return AgentSession(
        tool_calls=[
            ToolCall(name="summarize", args={}, result="[empty]", step=0),
            ToolCall(name="search", args={"query": "AI"}, result="[results]", step=1),
            ToolCall(name="delete_file", args={"path": "/tmp/cache"}, result="deleted", step=2),
            ToolCall(name="search", args={"query": "more"}, result="[results]", step=3),
            ToolCall(name="search", args={"query": "even more"}, result="[results]", step=4),
            ToolCall(name="search", args={"query": "and more"}, result="[results]", step=5),
        ],
        metadata={"scenario": "research_agent_broken", "model": "mock"}
    )


def make_customer_support_session() -> AgentSession:
    """A customer support agent that follows the right escalation flow."""
    return AgentSession(
        tool_calls=[
            ToolCall(name="lookup_customer", args={"id": "C123"}, result="{name: Sai}", step=0),
            ToolCall(name="lookup_order", args={"order_id": "O456"}, result="{status: delayed}", step=1),
            ToolCall(name="send_email", args={"template": "apology"}, result="sent", step=2),
        ],
        metadata={"scenario": "support_agent", "model": "mock"}
    )


# ---------------------------------------------------------------------------
# Live mode — actual Claude API with tool use
# ---------------------------------------------------------------------------

def run_agent_live(
    system_prompt: str,
    user_message: str,
    tools: list[dict],
    mock_tool_executor: Callable[[str, dict], Any] | None = None,
) -> AgentSession:
    """
    Runs a Claude agent with tool use and records every tool call into an AgentSession.

    Args:
        system_prompt: The agent's system prompt
        user_message: The task to give the agent
        tools: Anthropic-format tool definitions
        mock_tool_executor: Optional function that handles tool execution.
                           Signature: (tool_name, tool_args) -> result_str
                           If None, returns "[tool executed]" for all calls.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    messages = [{"role": "user", "content": user_message}]
    recorded_calls: list[ToolCall] = []
    step = 0
    max_steps = 10  # Safety limit — don't let agents run forever

    for _ in range(max_steps):
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        # Collect tool use blocks from this response turn
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            # No tool calls — agent is done
            break

        # Record each tool call
        tool_results = []
        for block in tool_use_blocks:
            args = block.input if isinstance(block.input, dict) else {}

            # Execute the tool (mock or real)
            if mock_tool_executor:
                result = str(mock_tool_executor(block.name, args))
            else:
                result = "[tool executed]"

            recorded_calls.append(ToolCall(
                name=block.name,
                args=args,
                result=result,
                step=step,
            ))
            step += 1

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        # Add assistant turn + tool results to conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        if response.stop_reason == "end_turn":
            break

    return AgentSession(
        tool_calls=recorded_calls,
        metadata={"model": config.CLAUDE_MODEL, "steps": step}
    )


def get_session(scenario: str = "good") -> AgentSession:
    """
    Entry point. Returns a session based on scenario name.
    In live mode, this would run the real agent.
    """
    if config.is_mock_mode():
        sessions = {
            "good": make_research_agent_session,
            "broken": make_broken_research_agent_session,
            "support": make_customer_support_session,
        }
        factory = sessions.get(scenario, make_research_agent_session)
        return factory()

    # In live mode — would call run_agent_live with real agent config
    # This is a placeholder; real usage passes system_prompt + tools
    raise NotImplementedError(
        "Live mode requires calling run_agent_live() directly with your agent config."
    )
