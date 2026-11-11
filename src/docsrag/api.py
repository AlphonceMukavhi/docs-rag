"""HTTP API: POST /ask with retrieval, generation and an answer cache."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import httpx
import redis
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from .config import build_embedder, build_generator
from .metrics import Metrics
from .rag import ABSTAIN, Generator, answer
from .store import ChunkStore, Hit

log = logging.getLogger("docsrag")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)
    mode: Literal["vector", "keyword", "hybrid"] = "hybrid"


class Source(BaseModel):
    n: str
    source: str
    section: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    cached: bool
    corpus_version: str


def to_response(payload: dict[str, Any], cached: bool, version: str) -> AskResponse:
    return AskResponse(
        answer=payload["answer"],
        sources=[Source(**s) for s in payload["sources"]],
        cached=cached,
        corpus_version=version,
    )


def search(store: ChunkStore, mode: str, question: str, k: int) -> list[Hit]:
    return {
        "vector": store.vector_search,
        "keyword": store.keyword_search,
        "hybrid": store.hybrid_search,
    }[mode](question, k)


def create_app(
    store: ChunkStore | None = None,
    generator: Generator | None = None,
    cache_client: redis.Redis | None = None,
    api_keys: set[str] | None = None,
    cache_ttl: int = 600,
    metrics: Metrics | None = None,
) -> FastAPI:
    meter = metrics or Metrics()
    keys = api_keys or {
        k for k in os.environ.get("DOCSRAG_API_KEYS", "dev-key").split(",") if k
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = store or ChunkStore(
            os.environ["DOCSRAG_DSN"], build_embedder()
        )
        app.state.generator = generator or build_generator()
        url = os.environ.get("DOCSRAG_REDIS_URL")
        app.state.cache = cache_client or (redis.Redis.from_url(url) if url else None)
        app.state.version = app.state.store.corpus_version()
        yield
        if store is None:
            app.state.store.close()

    app = FastAPI(title="docsrag", version="0.2.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            template = route.path if route is not None else "unmatched"
            meter.observe_request(
                request.method, template, status, time.perf_counter() - start
            )

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint() -> Response:
        body, content_type = meter.render()
        return Response(content=body, media_type=content_type)

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if x_api_key is None or not any(
            secrets.compare_digest(x_api_key, k) for k in keys
        ):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        ok = request.app.state.store.ping()
        return {
            "status": "ok" if ok else "degraded",
            "corpus_version": request.app.state.version,
        }

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
    def ask(body: AskRequest, request: Request) -> AskResponse:
        state = request.app.state
        raw = f"{state.version}|{body.mode}|{body.k}|{body.question.strip().lower()}"
        cache_key = "docsrag:" + hashlib.sha256(raw.encode()).hexdigest()[:32]
        if state.cache is not None:
            try:
                hit = state.cache.get(cache_key)
                if hit is not None:
                    meter.cache.labels("hit").inc()
                    return AskResponse(
                        **json.loads(hit), cached=True, corpus_version=state.version
                    )
            except redis.RedisError:
                log.warning("cache read failed", exc_info=True)
        meter.cache.labels("miss").inc()
        t0 = time.perf_counter()
        hits = search(state.store, body.mode, body.question, body.k)
        meter.retrieval.labels(body.mode).observe(time.perf_counter() - t0)
        t1 = time.perf_counter()
        try:
            result = answer(body.question, hits, state.generator)
        except httpx.HTTPError:
            meter.generation_errors.inc()
            log.exception("generation failed")
            raise HTTPException(
                status_code=503, detail="language model temporarily unavailable"
            ) from None
        meter.generation.observe(time.perf_counter() - t1)
        meter.answers.labels(
            "abstained" if result.text == ABSTAIN else "answered"
        ).inc()
        payload = {"answer": result.text, "sources": result.sources}
        if state.cache is not None:
            try:
                state.cache.set(cache_key, json.dumps(payload), ex=cache_ttl)
            except redis.RedisError:
                log.warning("cache write failed", exc_info=True)
        return to_response(payload, False, state.version)

    return app


def app_factory() -> FastAPI:
    return create_app()
