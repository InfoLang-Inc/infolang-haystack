"""Tests for :class:`InfoLangChatMemory`."""

from __future__ import annotations

import pytest
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils import Secret

from infolang_haystack import InfoLangChatMemory

from .conftest import (
    RecordingAsyncClient,
    RecordingSyncClient,
    make_chunk,
    make_recall_result,
)


def test_inject_after_leading_system(sync_client: RecordingSyncClient) -> None:
    sync_client.recall_result = make_recall_result([make_chunk(text="fact")])
    memory = InfoLangChatMemory(namespace="ns", auto_retain=False, client=sync_client)
    out = memory.run(
        messages=[ChatMessage.from_system("sys"), ChatMessage.from_user("question")]
    )
    msgs = out["messages"]
    assert len(msgs) == 3
    assert msgs[0].is_from(ChatRole.SYSTEM) and msgs[0].text == "sys"
    assert msgs[1].is_from(ChatRole.SYSTEM) and "fact" in (msgs[1].text or "")
    assert msgs[2].is_from(ChatRole.USER)
    assert out["documents"][0].content == "fact"


def test_inject_at_start_without_system(sync_client: RecordingSyncClient) -> None:
    sync_client.recall_result = make_recall_result([make_chunk(text="fact")])
    memory = InfoLangChatMemory(auto_retain=False, client=sync_client)
    out = memory.run(messages=[ChatMessage.from_user("q")])
    msgs = out["messages"]
    assert msgs[0].is_from(ChatRole.SYSTEM)
    assert msgs[1].is_from(ChatRole.USER)


def test_auto_recall_false_skips_recall(sync_client: RecordingSyncClient) -> None:
    memory = InfoLangChatMemory(auto_recall=False, auto_retain=True, client=sync_client)
    out = memory.run(messages=[ChatMessage.from_user("q")])
    assert len(out["messages"]) == 1
    assert out["documents"] == []
    ops = [c[0] for c in sync_client.calls]
    assert ops == ["remember"]


def test_auto_retain_false_skips_remember(sync_client: RecordingSyncClient) -> None:
    memory = InfoLangChatMemory(auto_recall=True, auto_retain=False, client=sync_client)
    memory.run(messages=[ChatMessage.from_user("q")])
    ops = [c[0] for c in sync_client.calls]
    assert ops == ["recall"]


def test_retain_passes_query_and_config(sync_client: RecordingSyncClient) -> None:
    memory = InfoLangChatMemory(
        namespace="ns", source="src", tags="t", auto_recall=False, client=sync_client
    )
    memory.run(messages=[ChatMessage.from_user("remember me")])
    op, text, kwargs = sync_client.calls[0]
    assert op == "remember"
    assert text == "remember me"
    assert kwargs == {"namespace": "ns", "source": "src", "tags": "t"}


def test_no_user_message_is_noop(sync_client: RecordingSyncClient) -> None:
    memory = InfoLangChatMemory(client=sync_client)
    msgs = [ChatMessage.from_system("only system")]
    out = memory.run(messages=msgs)
    assert out["messages"] == msgs
    assert out["documents"] == []
    assert sync_client.calls == []


def test_no_chunks_leaves_messages_but_retains(sync_client: RecordingSyncClient) -> None:
    sync_client.recall_result = make_recall_result([])
    memory = InfoLangChatMemory(client=sync_client)
    out = memory.run(messages=[ChatMessage.from_user("q")])
    assert len(out["messages"]) == 1
    ops = [c[0] for c in sync_client.calls]
    assert ops == ["recall", "remember"]


def test_score_threshold(sync_client: RecordingSyncClient) -> None:
    sync_client.recall_result = make_recall_result(
        [make_chunk(text="hi", score=0.9), make_chunk(id="c2", text="lo", score=0.1)]
    )
    memory = InfoLangChatMemory(score_threshold=0.5, auto_retain=False, client=sync_client)
    out = memory.run(messages=[ChatMessage.from_user("q")])
    assert len(out["documents"]) == 1
    block = out["messages"][0].text or ""
    assert "hi" in block and "lo" not in block


def test_last_user_text_picks_last_non_blank(sync_client: RecordingSyncClient) -> None:
    memory = InfoLangChatMemory(auto_recall=False, client=sync_client)
    memory.run(
        messages=[
            ChatMessage.from_user("first"),
            ChatMessage.from_assistant("reply"),
            ChatMessage.from_user("second"),
            ChatMessage.from_user("   "),
        ]
    )
    _, text, _ = sync_client.calls[0]
    assert text == "second"


def test_invalid_top_k_raises() -> None:
    with pytest.raises(ValueError):
        InfoLangChatMemory(top_k=-1)


def test_to_dict_from_dict_round_trip() -> None:
    memory = InfoLangChatMemory(
        namespace="app", top_k=3, auto_recall=False, auto_retain=True, header="H:"
    )
    data = memory.to_dict()
    restored = InfoLangChatMemory.from_dict(data)
    assert restored.namespace == "app"
    assert restored.top_k == 3
    assert restored.auto_recall is False
    assert restored.header == "H:"
    assert isinstance(restored.api_key, Secret)


def test_warm_up(sync_client: RecordingSyncClient) -> None:
    memory = InfoLangChatMemory(client=sync_client)
    memory.warm_up()
    assert memory._sync() is sync_client


def test_lazy_build(monkeypatch) -> None:
    fake = RecordingSyncClient()
    monkeypatch.setattr("infolang_haystack.memory.make_sync_client", lambda **kw: fake)
    memory = InfoLangChatMemory(auto_retain=False)
    memory.run(messages=[ChatMessage.from_user("q")])
    assert fake.calls[0][0] == "recall"


async def test_run_async(async_client: RecordingAsyncClient) -> None:
    async_client.recall_result = make_recall_result([make_chunk(text="afact")])
    memory = InfoLangChatMemory(namespace="ns", async_client=async_client)
    out = await memory.run_async(messages=[ChatMessage.from_user("q")])
    ops = [c[0] for c in async_client.calls]
    assert ops == ["recall", "remember"]
    assert any("afact" in (m.text or "") for m in out["messages"])


async def test_run_async_no_user(async_client: RecordingAsyncClient) -> None:
    memory = InfoLangChatMemory(async_client=async_client)
    out = await memory.run_async(messages=[ChatMessage.from_system("s")])
    assert out["documents"] == []
    assert async_client.calls == []


async def test_run_async_lazy_build(monkeypatch) -> None:
    fake = RecordingAsyncClient()
    monkeypatch.setattr("infolang_haystack.memory.make_async_client", lambda **kw: fake)
    memory = InfoLangChatMemory(auto_retain=False)
    await memory.run_async(messages=[ChatMessage.from_user("q")])
    assert fake.calls[0][0] == "recall"
