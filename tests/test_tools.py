"""Tests for the recall/remember Tool factories."""

from __future__ import annotations

from haystack.tools import Tool
from haystack.utils import Secret

from infolang_haystack import recall_tool, remember_tool

from .conftest import RecordingSyncClient, make_chunk, make_recall_result


def test_recall_tool_shape_and_call() -> None:
    client = RecordingSyncClient(recall_result=make_recall_result([make_chunk(text="ctx")]))
    tool = recall_tool(client=client, namespace="ns")
    assert isinstance(tool, Tool)
    assert tool.name == "infolang_recall"
    assert tool.parameters["required"] == ["query"]
    result = tool.function(query="q")
    assert result["chunks"][0]["text"] == "ctx"
    op, query, kwargs = client.calls[0]
    assert op == "recall"
    assert query == "q"
    assert kwargs["namespace"] == "ns"
    assert kwargs["top_k"] == 5


def test_recall_tool_custom_top_k_default() -> None:
    client = RecordingSyncClient()
    tool = recall_tool(client=client, top_k=3)
    tool.function(query="q")
    assert client.calls[0][2]["top_k"] == 3


def test_recall_tool_explicit_top_k() -> None:
    client = RecordingSyncClient()
    tool = recall_tool(client=client, top_k=3)
    tool.function(query="q", top_k=9)
    assert client.calls[0][2]["top_k"] == 9


def test_recall_tool_custom_name_description() -> None:
    tool = recall_tool(client=RecordingSyncClient(), name="mem_search", description="d")
    assert tool.name == "mem_search"
    assert tool.description == "d"


def test_remember_tool_shape_and_call() -> None:
    client = RecordingSyncClient()
    tool = remember_tool(client=client, namespace="ns", source="src")
    assert isinstance(tool, Tool)
    assert tool.name == "infolang_remember"
    assert tool.parameters["required"] == ["text"]
    result = tool.function(text="note", tags="a,b")
    assert result["memory_id"] == "m1"
    op, text, kwargs = client.calls[0]
    assert op == "remember"
    assert text == "note"
    assert kwargs == {"namespace": "ns", "source": "src", "tags": "a,b"}


def test_remember_tool_default_source() -> None:
    client = RecordingSyncClient()
    tool = remember_tool(client=client, source="default_src")
    tool.function(text="note")
    assert client.calls[0][2]["source"] == "default_src"


def test_recall_tool_builds_client_when_absent(monkeypatch) -> None:
    fake = RecordingSyncClient()
    monkeypatch.setattr("infolang_haystack.tools.make_sync_client", lambda **kw: fake)
    tool = recall_tool(api_key=Secret.from_token("il_live_x"), namespace="ns")
    tool.function(query="q")
    assert fake.calls[0][0] == "recall"


def test_remember_tool_builds_client_when_absent(monkeypatch) -> None:
    fake = RecordingSyncClient()
    monkeypatch.setattr("infolang_haystack.tools.make_sync_client", lambda **kw: fake)
    tool = remember_tool(api_key=Secret.from_token("il_live_x"))
    tool.function(text="t")
    assert fake.calls[0][0] == "remember"
