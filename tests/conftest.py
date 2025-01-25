"""
Unit tests for the configuration.
"""
import subprocess
import pytest

@pytest.fixture(scope="session", autouse=True)
def setup_dvc_data():
    """
    Ensure DVC-managed data is pulled before running tests.
    """
    try:
        subprocess.run(["dvc", "pull"], check=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to pull DVC data: {e}")
