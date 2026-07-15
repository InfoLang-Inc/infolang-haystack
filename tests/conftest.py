"""Shared fixtures: offline fake InfoLang clients and model builders.

The whole suite runs offline by default — no network, no credentials — by
injecting these fakes into the components. Live tests carry the ``live`` marker
and are deselected unless explicitly requested.
"""

from __future__ import annotations

from typing import Any

import pytest
from infolang import Chunk, RecallResult, RememberResult


def make_chunk(
    id: str = "c1",
    text: str = "hello world",
    score: float | None = 0.9,
    tags: str | None = None,
) -> Chunk:
    return Chunk(id=id, text=text, score=score, tags=tags)


def make_recall_result(
    chunks: list[Chunk] | None = None, namespace: str | None = "ns"
) -> RecallResult:
    return RecallResult(chunks=chunks or [], namespace=namespace)


def make_remember_result(
    memory_id: str | None = "m1", namespace: str | None = "ns"
) -> RememberResult:
    return RememberResult.model_validate({"id": memory_id, "namespace": namespace})


class RecordingSyncClient:
    """A stand-in for :class:`infolang.InfoLang` that records calls."""

    def __init__(
        self,
        recall_result: RecallResult | None = None,
        remember_result: RememberResult | None = None,
        batch_results: list[RememberResult] | None = None,
    ) -> None:
        self.recall_result = recall_result or make_recall_result([make_chunk()])
        self.remember_result = remember_result or make_remember_result()
        self.batch_results = batch_results
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []

    def recall(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        verbose: bool | None = None,
    ) -> RecallResult:
        self.calls.append(
            ("recall", query, {"namespace": namespace, "top_k": top_k, "filters": filters})
        )
        return self.recall_result

    def remember(
        self,
        text: str,
        *,
        namespace: str | None = None,
        source: str | None = None,
        tags: str | None = None,
    ) -> RememberResult:
        self.calls.append(
            ("remember", text, {"namespace": namespace, "source": source, "tags": tags})
        )
        return self.remember_result

    def remember_batch(
        self,
        items: list[Any],
        *,
        namespace: str | None = None,
        source: str | None = None,
    ) -> list[RememberResult]:
        self.calls.append(
            ("remember_batch", items, {"namespace": namespace, "source": source})
        )
        if self.batch_results is not None:
            return self.batch_results
        return [make_remember_result(memory_id=f"m{i}") for i, _ in enumerate(items)]


class RecordingAsyncClient(RecordingSyncClient):
    """Async mirror of :class:`RecordingSyncClient`."""

    async def recall(self, query: str, **kwargs: Any) -> RecallResult:  # type: ignore[override]
        return super().recall(query, **kwargs)

    async def remember(self, text: str, **kwargs: Any) -> RememberResult:  # type: ignore[override]
        return super().remember(text, **kwargs)

    async def remember_batch(self, items: list[Any], **kwargs: Any) -> list[RememberResult]:  # type: ignore[override]
        return super().remember_batch(items, **kwargs)


@pytest.fixture
def sync_client() -> RecordingSyncClient:
    return RecordingSyncClient()


@pytest.fixture
def async_client() -> RecordingAsyncClient:
    return RecordingAsyncClient()
