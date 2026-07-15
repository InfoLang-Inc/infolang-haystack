"""Tests for the client-construction helpers."""

from __future__ import annotations

from haystack.utils import Secret
from infolang import AsyncInfoLang, InfoLang

from infolang_haystack._client import (
    API_KEY_ENV_VAR,
    default_api_key,
    make_async_client,
    make_sync_client,
    resolve_api_key,
)


def test_resolve_api_key_none() -> None:
    assert resolve_api_key(None) is None


def test_resolve_api_key_token() -> None:
    assert resolve_api_key(Secret.from_token("il_live_abc")) == "il_live_abc"


def test_default_api_key_reads_env(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "il_live_env")
    assert resolve_api_key(default_api_key()) == "il_live_env"


def test_default_api_key_missing_is_none(monkeypatch) -> None:
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    assert resolve_api_key(default_api_key()) is None


def test_make_sync_client_with_key() -> None:
    client = make_sync_client(
        api_key=Secret.from_token("il_live_x"),
        namespace="ns",
        workspace="ws",
        base_url=None,
    )
    assert isinstance(client, InfoLang)
    assert client.namespace == "ns"
    assert client.workspace == "ws"


def test_make_async_client_with_key() -> None:
    client = make_async_client(
        api_key=Secret.from_token("il_live_x"),
        namespace="ns",
        workspace=None,
        base_url="https://api.example",
    )
    assert isinstance(client, AsyncInfoLang)
    assert client.namespace == "ns"


def test_make_sync_client_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "il_live_env")
    client = make_sync_client(
        api_key=None, namespace="fromarg", workspace=None, base_url=None
    )
    assert isinstance(client, InfoLang)
    assert client.namespace == "fromarg"


def test_make_async_client_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "il_live_env")
    client = make_async_client(
        api_key=None, namespace=None, workspace=None, base_url=None
    )
    assert isinstance(client, AsyncInfoLang)
