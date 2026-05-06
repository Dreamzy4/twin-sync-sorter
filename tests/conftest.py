"""Shared pytest fixtures and sys.path setup for the test suite.

Ensures the project root (one level above tests/) is importable so that
`import scene_config`, `import dashboard_server`, `import telemetry` work
from any test file without needing an installed package.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


@pytest.fixture
def cv_state_clean():
    """Yield CV_STATE with all per-run history cleared.

    ``CV_STATE.clear_history()`` resets cycle_log, both Twin-Sync series,
    phase timings, container counts, processed_xy, the cube list, and
    the reset_requested flag. Calling it before and after each test that
    uses this fixture keeps cross-test state from leaking through the
    shared dashboard CVState singleton.
    """
    from dashboard_server import CV_STATE

    CV_STATE.clear_history()
    yield CV_STATE
    CV_STATE.clear_history()


@pytest.fixture
def flask_client():
    """Flask test client for in-process dashboard route testing."""
    from dashboard_server import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
