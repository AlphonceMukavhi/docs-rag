import os

import pytest

from docsrag.chunking import Chunk
from docsrag.embedding import HashingEmbedder


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder(256)


@pytest.fixture
def dsn() -> str:
    value = os.environ.get("DOCSRAG_TEST_DSN")
    if not value:
        pytest.skip("set DOCSRAG_TEST_DSN (PostgreSQL with pgvector)")
    return value


SMALL_CORPUS = [
    Chunk(
        "git.md",
        "Undoing",
        0,
        "Use git revert to undo a pushed commit safely. Revert adds a new commit.",
    ),
    Chunk(
        "git.md",
        "Reflog",
        1,
        "The reflog lists where HEAD has been. "
        "Recover a lost commit with git switch -c rescue.",
    ),
    Chunk(
        "sql.md",
        "Indexes",
        0,
        "An index lets PostgreSQL find rows without scanning the whole table. "
        "Use EXPLAIN ANALYZE.",
    ),
    Chunk(
        "sql.md",
        "NULL",
        1,
        "Comparing with NULL yields unknown. Use IS NULL, never equals NULL.",
    ),
    Chunk(
        "np.md",
        "Broadcasting",
        0,
        "Broadcasting stretches arrays of different shapes. "
        "Align shapes from the right.",
    ),
]
