"""PostgreSQL + pgvector storage and the three retrievers."""

from __future__ import annotations

import re
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from .chunking import Chunk
from .embedding import Embedder, tokens
from .fusion import reciprocal_rank_fusion

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS chunks;
CREATE TABLE chunks (
    id         integer PRIMARY KEY,
    source     text NOT NULL,
    section    text NOT NULL,
    ordinal    integer NOT NULL,
    content    text NOT NULL,
    embedding  vector({dim}) NOT NULL,
    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
"""

SAFE_TERM = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class Hit:
    id: int
    source: str
    section: str
    content: str
    score: float


class ChunkStore:
    def __init__(self, dsn: str, embedder: Embedder) -> None:
        self.embedder = embedder
        self.conn = psycopg.connect(dsn, autocommit=True)
        register_vector(self.conn)

    def close(self) -> None:
        self.conn.close()

    def rebuild(self, chunks: list[Chunk], batch: int = 64) -> int:
        """Drop and recreate the table, embed every chunk, and load it."""
        self.conn.execute(SCHEMA.format(dim=self.embedder.dim))
        with self.conn.transaction(), self.conn.cursor() as cur:
            for start in range(0, len(chunks), batch):
                part = chunks[start : start + batch]
                vectors = self.embedder.embed([c.content for c in part])
                cur.executemany(
                    "INSERT INTO chunks "
                    "(id, source, section, ordinal, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (
                            start + i,
                            c.source,
                            c.section,
                            c.ordinal,
                            c.content,
                            vectors[i],
                        )
                        for i, c in enumerate(part)
                    ],
                )
        self.conn.execute("ANALYZE chunks")
        return len(chunks)

    def _rows(self, sql: str, params: tuple[object, ...]) -> list[Hit]:
        rows = self.conn.execute(sql, params).fetchall()
        return [Hit(r[0], r[1], r[2], r[3], float(r[4])) for r in rows]

    def vector_search(self, query: str, k: int) -> list[Hit]:
        vec = self.embedder.embed([query])[0]
        return self._rows(
            "SELECT id, source, section, content, 1 - (embedding <=> %s) AS score "
            "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (vec, vec, k),
        )

    def keyword_search(self, query: str, k: int) -> list[Hit]:
        terms = [t for t in dict.fromkeys(tokens(query)) if SAFE_TERM.match(t)]
        if not terms:
            return []
        expr = " | ".join(terms)  # OR semantics; terms are sanitised above
        return self._rows(
            "SELECT id, source, section, content, ts_rank_cd(tsv, q) AS score "
            "FROM chunks, to_tsquery('english', %s) AS q WHERE tsv @@ q "
            "ORDER BY score DESC, id LIMIT %s",
            (expr, k),
        )

    def hybrid_search(self, query: str, k: int, pool: int = 30) -> list[Hit]:
        by_vec = self.vector_search(query, pool)
        by_kw = self.keyword_search(query, pool)
        order = reciprocal_rank_fusion([[h.id for h in by_vec], [h.id for h in by_kw]])
        lookup = {h.id: h for h in [*by_kw, *by_vec]}
        return [lookup[i] for i in order[:k]]

    def corpus_version(self) -> str:
        """Fingerprint of the indexed text; changes whenever the corpus changes."""
        row = self.conn.execute(
            "SELECT md5(coalesce(string_agg(content, '' ORDER BY id), '')) FROM chunks"
        ).fetchone()
        return str(row[0])[:12] if row else "empty"

    def ping(self) -> bool:
        try:
            self.conn.execute("SELECT 1")
            return True
        except psycopg.Error:
            return False

    def count(self) -> int:
        row = self.conn.execute("SELECT count(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0
