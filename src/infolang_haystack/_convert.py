"""Pure conversions between InfoLang SDK models and Haystack dataclasses.

Kept free of I/O so the sync and async component paths share identical shaping
and can never drift.
"""

from __future__ import annotations

from typing import Any

from haystack import Document
from infolang import Chunk, RecallResult


def chunk_to_document(chunk: Chunk, *, namespace: str | None = None) -> Document:
    """Map an InfoLang :class:`~infolang.Chunk` onto a Haystack ``Document``."""

    return Document(
        content=chunk.text,
        score=chunk.score,
        meta={
            "infolang_id": chunk.id,
            "tags": chunk.tags,
            "namespace": namespace,
        },
    )


def recall_to_documents(
    result: RecallResult, *, score_threshold: float | None = None
) -> list[Document]:
    """Convert a recall result to documents, dropping chunks below the threshold.

    A chunk with no score is always kept; the threshold only filters chunks that
    report a score under it.
    """

    docs: list[Document] = []
    for chunk in result.chunks:
        if (
            score_threshold is not None
            and chunk.score is not None
            and chunk.score < score_threshold
        ):
            continue
        docs.append(chunk_to_document(chunk, namespace=result.namespace))
    return docs


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """A compact, JSON-friendly view of a chunk (id/text/score/tags)."""

    return {
        "id": chunk.id,
        "text": chunk.text,
        "score": chunk.score,
        "tags": chunk.tags,
    }


def format_context(
    result: RecallResult,
    *,
    header: str,
    score_threshold: float | None = None,
    max_chunks: int | None = None,
) -> str | None:
    """Render recalled chunks into a single injectable context block.

    Returns ``None`` when there is nothing worth injecting.
    """

    lines: list[str] = []
    for chunk in result.chunks:
        if (
            score_threshold is not None
            and chunk.score is not None
            and chunk.score < score_threshold
        ):
            continue
        text = (chunk.text or "").strip()
        if not text:
            continue
        lines.append(f"- {text}")
        if max_chunks is not None and len(lines) >= max_chunks:
            break
    if not lines:
        return None
    return header + "\n" + "\n".join(lines)
