import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def set_test_ports():
    """Override service ports for tests to avoid conflicts with other running services."""
    os.environ["QDRANT_PORT"] = "1000"
    os.environ["POSTGRES_PORT"] = "2000"
    os.environ["POSTGRES_PASSWORD"] = "postgres"
    yield
    os.environ.pop("QDRANT_PORT", None)
    os.environ.pop("POSTGRES_PORT", None)
    os.environ.pop("POSTGRES_PASSWORD", None)