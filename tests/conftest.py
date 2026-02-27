"""Global pytest configuration.

- Registers the `integration` marker.
- Loads environment variables from `.env` when python-dotenv is available.

Integration tests are intended to use real network calls and should be skipped
by default in CI (run with: `pytest -m integration`).
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (may use network)",
    )

    # Best-effort .env loading
    try:
        from dotenv import load_dotenv

        load_dotenv()  # loads from cwd/.env by default
    except Exception:
        # No dotenv installed (or other import/load issue). Ignore.
        return
