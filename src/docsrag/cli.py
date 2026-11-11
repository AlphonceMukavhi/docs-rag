"""Command line: ingest | ask | eval."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import psycopg

from .chunking import chunk_folder
from .config import build_embedder, build_generator
from .eval_set import EVAL_SET
from .evaluate import evaluate
from .rag import answer
from .store import ChunkStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docsrag", description=__doc__)
    parser.add_argument("--dsn", default=os.environ.get("DOCSRAG_DSN"))
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="chunk, embed and load a folder of .md files")
    ing.add_argument("--corpus", type=Path, default=Path("corpus"))
    ing.add_argument("--max-chars", type=int, default=900)
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=5)
    ask.add_argument(
        "--mode", choices=("vector", "keyword", "hybrid"), default="hybrid"
    )
    ev = sub.add_parser("eval", help="compare retrievers on the labelled question set")
    ev.add_argument("--k", type=int, default=5)
    ev.add_argument(
        "--min-hit-rate",
        type=float,
        default=None,
        help="exit 1 if the hybrid retriever's hit rate is below this (CI gate)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dsn:
        print("docsrag: error: set --dsn or DOCSRAG_DSN", file=sys.stderr)
        return 2
    try:
        store = ChunkStore(args.dsn, build_embedder())
    except psycopg.Error as exc:
        print(f"docsrag: error: cannot connect: {exc}", file=sys.stderr)
        return 2
    try:
        if args.command == "ingest":
            chunks = chunk_folder(args.corpus, max_chars=args.max_chars)
            if not chunks:
                print(f"docsrag: error: no .md files in {args.corpus}", file=sys.stderr)
                return 2
            print(f"ingested {store.rebuild(chunks)} chunks")
        elif args.command == "ask":
            hits = {
                "vector": store.vector_search,
                "keyword": store.keyword_search,
                "hybrid": store.hybrid_search,
            }[args.mode](args.question, args.k)
            result = answer(args.question, hits, build_generator())
            print(result.text)
            for s in result.sources:
                print(f"  [{s['n']}] {s['source']} :: {s['section']}")
        else:
            hybrid_rate = 0.0
            for name, fn in (
                ("vector", store.vector_search),
                ("keyword", store.keyword_search),
                ("hybrid", store.hybrid_search),
            ):
                r = evaluate(name, fn, EVAL_SET, args.k)
                print(f"{name:8s} hit@{r.k}={r.hit_rate:.2f}  mrr={r.mrr:.3f}")
                if name == "hybrid":
                    hybrid_rate = r.hit_rate
            if args.min_hit_rate is not None and hybrid_rate < args.min_hit_rate:
                print(
                    f"docsrag: quality gate failed: hybrid hit@{args.k} "
                    f"{hybrid_rate:.2f} < {args.min_hit_rate:.2f}",
                    file=sys.stderr,
                )
                return 1
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
