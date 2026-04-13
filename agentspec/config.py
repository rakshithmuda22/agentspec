"""
agentspec/config.py — Configuration for AgentSpec.

API keys are read from environment variables only. No .env files are auto-loaded.
Libraries should not mutate the global environment at import time.
"""
from __future__ import annotations

import os

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
AGENTSPEC_MODE: str = os.getenv("AGENTSPEC_MODE", "mock" if not ANTHROPIC_API_KEY else "live")


def is_mock_mode() -> bool:
    """Check if AgentSpec is running in mock mode (no API key needed)."""
    return AGENTSPEC_MODE == "mock" or not ANTHROPIC_API_KEY
