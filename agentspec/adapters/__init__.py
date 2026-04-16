"""Framework adapters — convert provider-specific traces into AgentSession."""
from agentspec.adapters.anthropic import from_anthropic_response, from_anthropic_messages

__all__ = ["from_anthropic_response", "from_anthropic_messages"]
