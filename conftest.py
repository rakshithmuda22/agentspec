"""Root conftest.py for AgentSpec test suite."""
import sys
from pathlib import Path

# Ensure the package is importable when running from source (not pip installed)
sys.path.insert(0, str(Path(__file__).parent))

# Register the pytest plugin for local/CI runs
# (when pip-installed, the entry_point in pyproject.toml handles this)
from agentspec.pytest_plugin import *  # noqa: F401, F403
