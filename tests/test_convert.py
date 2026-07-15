"""Tests for the pure conversion helpers."""

from __future__ import annotations

from infolang_haystack._convert import (
    chunk_to_dict,
    chunk_to_document,
    format_context,
    recall_to_documents,
)

from .conftest import make_chunk, make_recall_result


def test_chunk_to_document_maps_fields() -> None:
    doc = chunk_to_document(make_chunk(id="x", text="body", score=0.7, tags="t"), namespace="ns")
    assert doc.content == "body"
    assert doc.score == 0.7
    assert doc.meta["infolang_id"] == "x"
    assert doc.meta["tags"] == "t"
    assert doc.meta["namespace"] == "ns"


def test_recall_to_documents_keeps_all_without_threshold() -> None:
    result = make_recall_result([make_chunk(score=0.9), make_chunk(id="c2", score=0.1)])
    docs = recall_to_documents(result)
    assert len(docs) == 2


def test_recall_to_documents_filters_below_threshold() -> None:
    result = make_recall_result([make_chunk(score=0.9), make_chunk(id="c2", score=0.1)])
    docs = recall_to_documents(result, score_threshold=0.5)
    assert len(docs) == 1
    assert docs[0].score == 0.9


def test_recall_to_documents_keeps_none_score() -> None:
    result = make_recall_result([make_chunk(id="c2", score=None)])
    docs = recall_to_documents(result, score_threshold=0.5)
    assert len(docs) == 1


def test_chunk_to_dict() -> None:
    d = chunk_to_dict(make_chunk(id="c1", text="hi", score=0.5, tags="a"))
    assert d == {"id": "c1", "text": "hi", "score": 0.5, "tags": "a"}


def test_format_context_builds_block() -> None:
    result = make_recall_result([make_chunk(text="first"), make_chunk(id="c2", text="second")])
    block = format_context(result, header="Memory:")
    assert block is not None
    assert block.startswith("Memory:")
    assert "- first" in block
    assert "- second" in block


def test_format_context_empty_returns_none() -> None:
    assert format_context(make_recall_result([]), header="Memory:") is None


def test_format_context_all_below_threshold_returns_none() -> None:
    result = make_recall_result([make_chunk(score=0.1)])
    assert format_context(result, header="Memory:", score_threshold=0.5) is None


def test_format_context_skips_blank_text() -> None:
    result = make_recall_result([make_chunk(text="   "), make_chunk(id="c2", text="real")])
    block = format_context(result, header="Memory:")
    assert block is not None
    assert "real" in block
    assert block.count("- ") == 1


def test_format_context_respects_max_chunks() -> None:
    result = make_recall_result(
        [make_chunk(text="a"), make_chunk(id="c2", text="b"), make_chunk(id="c3", text="c")]
    )
    block = format_context(result, header="Memory:", max_chunks=2)
    assert block is not None
    assert block.count("- ") == 2
