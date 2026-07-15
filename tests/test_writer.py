"""Tests for :class:`InfoLangWriter`."""

from __future__ import annotations

from haystack import Document
from haystack.utils import Secret

from infolang_haystack import InfoLangWriter

from .conftest import RecordingAsyncClient, RecordingSyncClient


def test_run_with_documents(sync_client: RecordingSyncClient) -> None:
    writer = InfoLangWriter(namespace="ns", source="ingest", client=sync_client)
    out = writer.run(documents=[Document(content="a"), Document(content="b")])
    assert out["memory_ids"] == ["m0", "m1"]
    assert len(out["results"]) == 2
    op, items, kwargs = sync_client.calls[0]
    assert op == "remember_batch"
    assert items == [{"text": "a", "source": "ingest"}, {"text": "b", "source": "ingest"}]
    assert kwargs == {"namespace": "ns", "source": "ingest"}


def test_run_with_texts(sync_client: RecordingSyncClient) -> None:
    writer = InfoLangWriter(client=sync_client)
    out = writer.run(texts=["one", "two"])
    assert out["memory_ids"] == ["m0", "m1"]
    _, items, _ = sync_client.calls[0]
    assert items == [{"text": "one"}, {"text": "two"}]


def test_run_with_documents_and_texts(sync_client: RecordingSyncClient) -> None:
    writer = InfoLangWriter(client=sync_client)
    writer.run(documents=[Document(content="d")], texts=["t"])
    _, items, _ = sync_client.calls[0]
    assert items == [{"text": "d"}, {"text": "t"}]


def test_run_empty_does_not_call_client(sync_client: RecordingSyncClient) -> None:
    writer = InfoLangWriter(client=sync_client)
    out = writer.run()
    assert out == {"memory_ids": [], "results": []}
    assert sync_client.calls == []


def test_empty_content_documents_skipped(sync_client: RecordingSyncClient) -> None:
    writer = InfoLangWriter(client=sync_client)
    writer.run(documents=[Document(content=""), Document(content="ok")], texts=[""])
    _, items, _ = sync_client.calls[0]
    assert items == [{"text": "ok"}]


def test_init_tags_source_override_doc_meta(sync_client: RecordingSyncClient) -> None:
    writer = InfoLangWriter(source="init_src", tags="init_tag", client=sync_client)
    doc = Document(content="x", meta={"tags": "doc_tag", "source": "doc_src"})
    writer.run(documents=[doc])
    _, items, _ = sync_client.calls[0]
    assert items == [{"text": "x", "tags": "init_tag", "source": "init_src"}]


def test_run_args_override_everything(sync_client: RecordingSyncClient) -> None:
    writer = InfoLangWriter(source="init_src", tags="init_tag", client=sync_client)
    doc = Document(content="x", meta={"tags": "doc_tag"})
    writer.run(documents=[doc], source="run_src", tags="run_tag")
    _, items, kwargs = sync_client.calls[0]
    assert items == [{"text": "x", "tags": "run_tag", "source": "run_src"}]
    assert kwargs["source"] == "run_src"


def test_doc_meta_used_when_no_init(sync_client: RecordingSyncClient) -> None:
    writer = InfoLangWriter(client=sync_client)
    doc = Document(content="x", meta={"tags": "doc_tag", "source": "doc_src"})
    writer.run(documents=[doc])
    _, items, _ = sync_client.calls[0]
    assert items == [{"text": "x", "tags": "doc_tag", "source": "doc_src"}]


def test_texts_inherit_init_tags_and_source(sync_client: RecordingSyncClient) -> None:
    writer = InfoLangWriter(source="s", tags="t", client=sync_client)
    writer.run(texts=["a"])
    _, items, _ = sync_client.calls[0]
    assert items == [{"text": "a", "tags": "t", "source": "s"}]


def test_from_dict_without_secret() -> None:
    data = {
        "type": "infolang_haystack.writer.InfoLangWriter",
        "init_parameters": {"namespace": "app", "api_key": None},
    }
    restored = InfoLangWriter.from_dict(data)
    assert restored.namespace == "app"


def test_to_dict_from_dict_round_trip() -> None:
    writer = InfoLangWriter(namespace="app", source="s", tags="t")
    data = writer.to_dict()
    restored = InfoLangWriter.from_dict(data)
    assert restored.namespace == "app"
    assert restored.source == "s"
    assert restored.tags == "t"
    assert isinstance(restored.api_key, Secret)


def test_warm_up(sync_client: RecordingSyncClient) -> None:
    writer = InfoLangWriter(client=sync_client)
    writer.warm_up()
    assert writer._sync() is sync_client


def test_lazy_build(monkeypatch) -> None:
    fake = RecordingSyncClient()
    monkeypatch.setattr("infolang_haystack.writer.make_sync_client", lambda **kw: fake)
    writer = InfoLangWriter()
    writer.run(texts=["hi"])
    assert fake.calls[0][0] == "remember_batch"


async def test_run_async(async_client: RecordingAsyncClient) -> None:
    writer = InfoLangWriter(namespace="ns", async_client=async_client)
    out = await writer.run_async(texts=["a", "b"])
    assert out["memory_ids"] == ["m0", "m1"]
    assert async_client.calls[0][0] == "remember_batch"


async def test_run_async_empty(async_client: RecordingAsyncClient) -> None:
    writer = InfoLangWriter(async_client=async_client)
    out = await writer.run_async()
    assert out == {"memory_ids": [], "results": []}
    assert async_client.calls == []


async def test_run_async_lazy_build(monkeypatch) -> None:
    fake = RecordingAsyncClient()
    monkeypatch.setattr("infolang_haystack.writer.make_async_client", lambda **kw: fake)
    writer = InfoLangWriter()
    await writer.run_async(texts=["hi"])
    assert fake.calls[0][0] == "remember_batch"
