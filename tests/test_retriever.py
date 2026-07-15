"""Tests for :class:`InfoLangRetriever`."""

from __future__ import annotations

import pytest
from haystack import Document
from haystack.utils import Secret

from infolang_haystack import InfoLangRetriever

from .conftest import (
    RecordingAsyncClient,
    RecordingSyncClient,
    make_chunk,
    make_recall_result,
)


def test_run_returns_documents_and_chunks(sync_client: RecordingSyncClient) -> None:
    sync_client.recall_result = make_recall_result(
        [make_chunk(text="a"), make_chunk(id="c2", text="b")]
    )
    retriever = InfoLangRetriever(namespace="ns", client=sync_client)
    out = retriever.run(query="hello")
    assert [d.content for d in out["documents"]] == ["a", "b"]
    assert len(out["chunks"]) == 2
    assert isinstance(out["documents"][0], Document)


def test_run_passes_defaults(sync_client: RecordingSyncClient) -> None:
    retriever = InfoLangRetriever(namespace="ns", top_k=7, filters={"k": "v"}, client=sync_client)
    retriever.run(query="q")
    _, query, kwargs = sync_client.calls[0]
    assert query == "q"
    assert kwargs == {"namespace": "ns", "top_k": 7, "filters": {"k": "v"}}


def test_run_overrides_win(sync_client: RecordingSyncClient) -> None:
    retriever = InfoLangRetriever(namespace="ns", top_k=7, filters={"k": "v"}, client=sync_client)
    retriever.run(query="q", top_k=2, filters={"other": 1}, namespace="other-ns")
    _, _, kwargs = sync_client.calls[0]
    assert kwargs == {"namespace": "other-ns", "top_k": 2, "filters": {"other": 1}}


def test_run_empty_filters_override_is_respected(sync_client: RecordingSyncClient) -> None:
    retriever = InfoLangRetriever(filters={"k": "v"}, client=sync_client)
    retriever.run(query="q", filters={})
    _, _, kwargs = sync_client.calls[0]
    assert kwargs["filters"] == {}


def test_score_threshold_filters_documents(sync_client: RecordingSyncClient) -> None:
    sync_client.recall_result = make_recall_result(
        [make_chunk(score=0.9), make_chunk(id="c2", score=0.2)]
    )
    retriever = InfoLangRetriever(score_threshold=0.5, client=sync_client)
    out = retriever.run(query="q")
    assert len(out["documents"]) == 1
    assert len(out["chunks"]) == 2  # chunks always returned unfiltered


def test_invalid_top_k_raises() -> None:
    with pytest.raises(ValueError):
        InfoLangRetriever(top_k=0)


def test_warm_up_uses_injected_client(sync_client: RecordingSyncClient) -> None:
    retriever = InfoLangRetriever(client=sync_client)
    retriever.warm_up()
    assert retriever._sync() is sync_client


def test_lazy_build_when_no_client(monkeypatch) -> None:
    fake = RecordingSyncClient()
    monkeypatch.setattr(
        "infolang_haystack.retriever.make_sync_client", lambda **kw: fake
    )
    retriever = InfoLangRetriever(namespace="ns")
    out = retriever.run(query="q")
    assert out["documents"]
    assert fake.calls[0][0] == "recall"


def test_to_dict_from_dict_round_trip() -> None:
    retriever = InfoLangRetriever(namespace="app", top_k=3, score_threshold=0.4)
    data = retriever.to_dict()
    assert data["type"].endswith("InfoLangRetriever")
    restored = InfoLangRetriever.from_dict(data)
    assert restored.namespace == "app"
    assert restored.top_k == 3
    assert restored.score_threshold == 0.4
    assert isinstance(restored.api_key, Secret)


def test_from_dict_without_secret() -> None:
    data = {
        "type": "infolang_haystack.retriever.InfoLangRetriever",
        "init_parameters": {"namespace": "app", "api_key": None},
    }
    restored = InfoLangRetriever.from_dict(data)
    assert restored.namespace == "app"


async def test_run_async(async_client: RecordingAsyncClient) -> None:
    async_client.recall_result = make_recall_result([make_chunk(text="z")])
    retriever = InfoLangRetriever(namespace="ns", async_client=async_client)
    out = await retriever.run_async(query="hi")
    assert out["documents"][0].content == "z"
    assert async_client.calls[0][0] == "recall"


async def test_run_async_lazy_build(monkeypatch) -> None:
    fake = RecordingAsyncClient()
    monkeypatch.setattr(
        "infolang_haystack.retriever.make_async_client", lambda **kw: fake
    )
    retriever = InfoLangRetriever(namespace="ns")
    out = await retriever.run_async(query="q")
    assert out["documents"]
