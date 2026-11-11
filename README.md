# docs-rag

A question-answering API over a knowledge base of Markdown documents, using retrieval-augmented generation (RAG). The sample knowledge base is an IT help desk for a primary school, covering printing, Wi-Fi, accounts, the website, backups and the computer lab. Staff ask a question in plain language, and the service finds the right sections and answers with citations, or says it does not know.

## How it works

```
Markdown files -> chunking -> embeddings -> PostgreSQL + pgvector
question -> full-text search + vector search -> reciprocal rank fusion -> answer with sources
```

- `chunking.py` splits documents by heading, with a little overlap between chunks.
- `embedding.py` holds pluggable embedders: a fast offline one for tests and CI, and an optional Sentence Transformers model.
- `store.py` uses PostgreSQL for both full-text search (`tsvector`) and vector similarity (pgvector, HNSW index).
- `fusion.py` combines the two rankings with reciprocal rank fusion (hybrid search).
- `rag.py` builds the prompt and generates the answer. It refuses to answer when nothing relevant is found.
- `api.py` is the FastAPI service, with API-key authentication, a Redis answer cache and Prometheus metrics.

## Run it

```bash
uv sync
export DOCSRAG_DSN=postgresql://user:pw@localhost:5432/docsrag   # needs the pgvector extension
uv run docsrag ingest --corpus corpus
uv run docsrag ask "How do I reset a staff password?"
uv run docsrag eval
```

```
In the Google Admin console, open the user, choose Reset password, ... [1]
  [1] accounts.md :: Resetting a staff password
```

A question the documents cannot answer, such as "What is the capital of France?", gets "I don't know based on the provided notes." instead of a made-up answer.

To run it as an API: `uv run uvicorn docsrag.api:app_factory --factory`, then `POST /ask` with an `X-API-Key` header.

## Measuring retrieval quality

`eval_set.py` holds 22 real help-desk questions, each labelled with the section that should answer it. `docsrag eval` compares the three retrievers:

| Retriever | Hit rate @5 | MRR |
|---|---|---|
| Vector only | 0.95 | 0.74 |
| Keyword only | 1.00 | 0.75 |
| Hybrid (fusion) | 0.95 | 0.80 |

Hybrid search ranks the right section highest on average (best MRR). CI fails the build if the hit rate drops below 0.9, so a change that hurts retrieval cannot be merged.

## Real models

`uv sync --extra hf` with `DOCSRAG_EMBEDDER=sentence-transformers` switches on a neural embedder. Set `DOCSRAG_LLM_URL` to any OpenAI-compatible endpoint to use an LLM for generation instead of the offline extractive answerer.

## Tests and CI

```bash
uv run pytest -q
DOCSRAG_TEST_DSN=postgresql://... uv run pytest -q    # database tests
```

## Why I built it

I wanted to understand how RAG systems work below the framework level: chunking, hybrid search and evaluation. I used a school IT help desk because that is the kind of support work I do day to day.
