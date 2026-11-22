from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "research_artifacts(*paths): requires ignored historical experiment artifacts; "
        "runs when every path is available and skips on a clean source checkout",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    marker = item.get_closest_marker("research_artifacts")
    if marker is None:
        return
    required = tuple(Path(str(path)) for path in marker.args)
    force_missing = os.environ.get("SEMANTICFLOW_TEST_WITHOUT_RESEARCH_ARTIFACTS") == "1"
    missing = tuple(
        path.as_posix() for path in required if force_missing or not path.exists()
    )
    if missing:
        pytest.skip("historical research artifact corpus is not installed: " + ", ".join(missing))
