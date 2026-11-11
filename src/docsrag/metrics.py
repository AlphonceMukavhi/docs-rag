"""Prometheus instrumentation for the RAG service."""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "http_requests_total",
            "HTTP requests",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency",
            ["method", "route"],
            buckets=BUCKETS,
            registry=self.registry,
        )
        self.retrieval = Histogram(
            "docsrag_retrieval_seconds",
            "Time spent retrieving chunks",
            ["mode"],
            buckets=BUCKETS,
            registry=self.registry,
        )
        self.generation = Histogram(
            "docsrag_generation_seconds",
            "Time spent in the generator",
            buckets=BUCKETS,
            registry=self.registry,
        )
        self.generation_errors = Counter(
            "docsrag_generation_errors_total",
            "Generator failures",
            registry=self.registry,
        )
        self.cache = Counter(
            "docsrag_answer_cache_total",
            "Answer cache lookups",
            ["result"],
            registry=self.registry,
        )
        self.answers = Counter(
            "docsrag_answers_total",
            "Answers by outcome",
            ["outcome"],
            registry=self.registry,
        )

    def observe_request(
        self, method: str, route: str, status: int, seconds: float
    ) -> None:
        self.requests.labels(method, route, str(status)).inc()
        self.duration.labels(method, route).observe(seconds)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
