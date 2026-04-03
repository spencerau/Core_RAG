import os
import pytest


@pytest.fixture(autouse=True)
def assert_no_routing_error(caplog):
    """Fail any test that triggers a query router JSON validation error."""
    import logging
    with caplog.at_level(logging.ERROR, logger="core_rag.retrieval.query_router"):
        yield
    routing_errors = [r for r in caplog.records if r.name == "core_rag.retrieval.query_router" and r.levelno >= logging.ERROR]
    assert not routing_errors, (
        "Query router failed to produce valid JSON output — "
        "check router LLM and structured output config.\n"
        f"Error: {routing_errors[0].getMessage()}"
    )


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