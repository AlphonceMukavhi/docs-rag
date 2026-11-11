"""Build the embedder and generator from environment variables."""

from __future__ import annotations

import os

from .embedding import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from .rag import ExtractiveGenerator, Generator, OpenAICompatibleGenerator


def build_embedder() -> Embedder:
    if os.environ.get("DOCSRAG_EMBEDDER", "hashing") == "sentence-transformers":
        return SentenceTransformerEmbedder(
            os.environ.get("DOCSRAG_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
    return HashingEmbedder(int(os.environ.get("DOCSRAG_HASH_DIM", "512")))


def build_generator() -> Generator:
    url = os.environ.get("DOCSRAG_LLM_URL")
    if url:
        return OpenAICompatibleGenerator(
            url,
            os.environ.get("DOCSRAG_LLM_MODEL", "default"),
            os.environ.get("DOCSRAG_LLM_API_KEY", ""),
        )
    return ExtractiveGenerator()
