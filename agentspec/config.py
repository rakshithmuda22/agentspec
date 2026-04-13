"""
config.py — Single source of truth for all environment variables.
"""
import os
from pathlib import Path

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
AGENTSPEC_MODE: str = os.getenv("AGENTSPEC_MODE", "mock" if not ANTHROPIC_API_KEY else "live")
CLAUDE_MODEL: str = "claude-sonnet-4-6"


def is_mock_mode() -> bool:
    return AGENTSPEC_MODE == "mock" or not ANTHROPIC_API_KEY
