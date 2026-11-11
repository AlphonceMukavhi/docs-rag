"""Needs PostgreSQL with the pgvector extension available (DOCSRAG_TEST_DSN)."""

import pytest
from conftest import SMALL_CORPUS
from fastapi.testclient import TestClient

from docsrag.api import create_app
from docsrag.rag import ABSTAIN, ExtractiveGenerator
from docsrag.store import ChunkStore

pytestmark = pytest.mark.db
KEY = {"X-API-Key": "k"}


@pytest.fixture
def store(dsn, embedder):
    s = ChunkStore(dsn, embedder)
    s.rebuild(SMALL_CORPUS)
    yield s
    s.close()


def test_three_retrievers_find_the_right_chunk(store):
    assert store.count() == len(SMALL_CORPUS)
    q = "how do I recover a lost commit"
    assert store.vector_search(q, 1)[0].section == "Reflog"
    assert store.keyword_search(q, 1)[0].section == "Reflog"
    assert store.hybrid_search(q, 1)[0].section == "Reflog"


def test_hostile_query_is_treated_as_plain_words(store):
    hits = store.keyword_search("'; DROP TABLE chunks; --", 3)  # no exception
    assert all(h.section for h in hits)
    assert store.count() == len(SMALL_CORPUS)  # the table is still there


def test_corpus_version_changes_when_text_changes(store, embedder):
    before = store.corpus_version()
    store.rebuild(SMALL_CORPUS[:-1])
    assert store.corpus_version() != before


class FakeRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value


def test_api_answers_caches_and_validates(store):
    fake = FakeRedis()
    app = create_app(store, ExtractiveGenerator(), fake, api_keys={"k"})
    body = {"question": "how do I undo a pushed commit", "k": 3}
    with TestClient(app) as client:
        assert client.post("/ask", json=body).status_code == 401
        first = client.post("/ask", json=body, headers=KEY).json()
        second = client.post("/ask", json=body, headers=KEY).json()
        assert (
            client.post("/ask", json={"question": "hi"}, headers=KEY).status_code == 422
        )
        assert (
            client.post("/ask", json={**body, "k": 99}, headers=KEY).status_code == 422
        )
        off_topic = client.post(
            "/ask", json={"question": "capital of Zimbabwe"}, headers=KEY
        ).json()
        health = client.get("/health").json()
    assert first["cached"] is False and second["cached"] is True
    assert first["answer"] == second["answer"] and "[" in first["answer"]
    assert first["sources"][0]["section"] == "Undoing"
    assert off_topic["answer"] == ABSTAIN
    assert (
        health["status"] == "ok" and health["corpus_version"] == first["corpus_version"]
    )


def test_eval_quality_gate_passes_and_fails_on_threshold(dsn):
    from pathlib import Path

    from docsrag.cli import main

    corpus = str(Path(__file__).parents[1] / "corpus")
    assert main(["--dsn", dsn, "ingest", "--corpus", corpus]) == 0
    assert main(["--dsn", dsn, "eval", "--k", "5", "--min-hit-rate", "0.9"]) == 0
    assert main(["--dsn", dsn, "eval", "--k", "5", "--min-hit-rate", "0.999"]) == 1
