"""Root conftest.py for AgentSpec test suite."""
# Plugin is auto-registered via pyproject.toml entry_points when installed.
# For local development (not pip installed), add the package to sys.path:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
