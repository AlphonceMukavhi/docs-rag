"""Metrics and failure handling with a stub store: no database needed."""

import httpx
from fastapi.testclient import TestClient

from docsrag.api import create_app
from docsrag.rag import ABSTAIN, ExtractiveGenerator
from docsrag.store import Hit

KEY = {"X-API-Key": "k"}


class StubStore:
    def __init__(self, hits):
        self.hits = hits

    def corpus_version(self):
        return "v1"

    def ping(self):
        return True

    def close(self):
        pass

    def hybrid_search(self, question, k):
        return self.hits[:k]

    vector_search = keyword_search = hybrid_search


class Cache:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value


class BrokenGenerator:
    def generate(self, question, hits):
        raise httpx.ConnectError("llm down")


def hit():
    return Hit(
        id=1,
        source="git.md",
        section="Undoing",
        content="Use git revert to undo a pushed commit safely.",
        score=1.0,
    )


def test_metrics_count_requests_cache_and_outcomes():
    app = create_app(StubStore([hit()]), ExtractiveGenerator(), Cache(), {"k"})
    body = {"question": "how do I undo a pushed commit"}
    with TestClient(app) as client:
        client.post("/ask", json=body, headers=KEY)
        client.post("/ask", json=body, headers=KEY)
        text = client.get("/metrics").text
    assert 'http_requests_total{method="POST",route="/ask",status="200"} 2.0' in text
    assert 'docsrag_answer_cache_total{result="hit"} 1.0' in text
    assert 'docsrag_answer_cache_total{result="miss"} 1.0' in text
    assert "docsrag_retrieval_seconds_bucket" in text
    assert 'docsrag_answers_total{outcome="answered"} 1.0' in text


def test_no_hits_abstains_and_is_counted():
    app = create_app(StubStore([]), ExtractiveGenerator(), Cache(), {"k"})
    with TestClient(app) as client:
        r = client.post("/ask", json={"question": "capital of Zimbabwe"}, headers=KEY)
        text = client.get("/metrics").text
    assert r.json()["answer"] == ABSTAIN
    assert 'docsrag_answers_total{outcome="abstained"} 1.0' in text


def test_llm_failure_is_503_counted_and_not_cached():
    cache = Cache()
    app = create_app(StubStore([hit()]), BrokenGenerator(), cache, {"k"})
    with TestClient(app) as client:
        r = client.post("/ask", json={"question": "undo a pushed commit"}, headers=KEY)
        text = client.get("/metrics").text
    assert r.status_code == 503
    assert cache.data == {}
    assert "docsrag_generation_errors_total 1.0" in text
    assert 'status="503"' in text
