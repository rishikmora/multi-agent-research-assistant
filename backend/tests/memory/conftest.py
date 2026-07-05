"""
Shared fixtures for the memory test suite.

Setup for local testing:
  1. docker run -d -p 5432:5432 -e POSTGRES_USER=mars_test \\
       -e POSTGRES_PASSWORD=mars_test -e POSTGRES_DB=mars_test_db \\
       pgvector/pgvector:pg16
  2. pip install fakeredis pytest-asyncio --break-system-packages
  3. pytest tests/memory/ -v

CI setup: see .github/workflows/test.yml — spins up postgres+pgvector
as a service container, no local Docker needed.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_postgres: test needs a real PostgreSQL+pgvector instance"
    )
    config.addinivalue_line(
        "markers", "requires_embedding_model: test needs sentence-transformers loaded"
    )


@pytest.fixture(autouse=True)
def _reset_random_seed():
    """Ensure deterministic fake embeddings across test runs."""
    import random
    random.seed(42)
