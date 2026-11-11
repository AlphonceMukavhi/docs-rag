"""Split documents into overlapping, section-aware chunks."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
COMMENT = re.compile(r"^\s*<!--.*-->\s*$")
QUOTE = re.compile(r"^>\s?")


@dataclass(frozen=True)
class Chunk:
    source: str  # file name, e.g. printing.md
    section: str  # nearest heading
    ordinal: int  # position within the source
    content: str


def clean(line: str) -> str | None:
    """Drop HTML comments and blockquote markers; keep readable text."""
    if COMMENT.match(line):
        return None
    return QUOTE.sub("", line)


def sections(text: str) -> Iterator[tuple[str, list[str]]]:
    """Yield (heading, paragraphs) pairs. Code fences stay inside their paragraph."""
    heading = "Introduction"
    paragraphs: list[str] = []
    current: list[str] = []
    in_fence = False
    for raw in text.splitlines():
        if raw.startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(raw)
        if match:
            if current:
                paragraphs.append("\n".join(current))
                current = []
            if paragraphs:
                yield heading, paragraphs
            heading, paragraphs = match.group(2).strip(), []
            continue
        line = clean(raw)
        if line is None:
            continue
        if not line.strip() and not in_fence:
            if current:
                paragraphs.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append("\n".join(current))
    if paragraphs:
        yield heading, paragraphs


def chunk_document(
    source: str, text: str, max_chars: int = 900, overlap_paragraphs: int = 1
) -> list[Chunk]:
    """Pack paragraphs into chunks of at most ~max_chars, repeating the last
    paragraph(s) at the start of the next chunk so context is not cut mid-thought."""
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    chunks: list[Chunk] = []
    for heading, paragraphs in sections(text):
        window: list[str] = []
        size = 0
        for para in paragraphs:
            if window and size + len(para) > max_chars:
                chunks.append(Chunk(source, heading, len(chunks), "\n\n".join(window)))
                window = window[-overlap_paragraphs:] if overlap_paragraphs else []
                size = sum(len(p) for p in window)
            window.append(para)
            size += len(para)
        if window:
            chunks.append(Chunk(source, heading, len(chunks), "\n\n".join(window)))
    return chunks


def chunk_folder(folder: Path, **kwargs: int) -> list[Chunk]:
    out: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        out.extend(
            chunk_document(path.name, path.read_text(encoding="utf-8"), **kwargs)
        )
    return out
