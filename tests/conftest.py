"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def protect_production_output():
    """Monitor output/ directory to ensure tests don't modify it.

    This fixture runs once per test session and checks that the
    production output/ directory is not modified by any test.
    """
    output_dir = Path("output")

    if not output_dir.exists():
        # If output doesn't exist, skip protection
        yield
        return

    # Record timestamps before tests
    before = {}
    for file in output_dir.glob("*"):
        if file.is_file():
            before[file.name] = file.stat().st_mtime

    yield  # Run all tests

    # Check timestamps after tests
    for file in output_dir.glob("*"):
        if file.is_file() and file.name in before:
            after_mtime = file.stat().st_mtime
            if after_mtime != before[file.name]:
                pytest.fail(
                    f"PRODUCTION DATA MODIFIED: {file.name} was changed during tests! "
                    f"Tests must use tmp_path fixture, not write to output/ directory."
                )

    # Check for new files
    after_files = {f.name for f in output_dir.glob("*") if f.is_file()}
    new_files = after_files - set(before.keys())
    if new_files:
        pytest.fail(
            f"PRODUCTION DATA CREATED: {new_files} created in output/ during tests! "
            f"Tests must use tmp_path fixture, not write to output/ directory."
        )
