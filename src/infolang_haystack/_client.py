"""Client construction helpers shared by the components and tools.

Everything here builds on the **published** InfoLang Python SDK
(``infolang>=0.2,<0.3``). We never speak HTTP directly or reach into runtime
internals — the SDK is the only contract surface.
"""

from __future__ import annotations

from typing import cast

from haystack.utils import Secret
from infolang import AsyncInfoLang, InfoLang

API_KEY_ENV_VAR = "INFOLANG_API_KEY"


def default_api_key() -> Secret:
    """A non-strict env-var secret so components construct even without a key set.

    Resolution is deferred until a real recall/remember call, which keeps
    serialization and offline (mocked-client) tests free of credentials.
    """

    return Secret.from_env_var(API_KEY_ENV_VAR, strict=False)


def resolve_api_key(api_key: Secret | None) -> str | None:
    """Resolve a :class:`~haystack.utils.Secret` to its string value, if any."""

    if api_key is None:
        return None
    return cast("str | None", api_key.resolve_value())


def make_sync_client(
    *,
    api_key: Secret | None,
    namespace: str | None,
    workspace: str | None,
    base_url: str | None,
) -> InfoLang:
    """Build a synchronous :class:`infolang.InfoLang` from resolved config."""

    key = resolve_api_key(api_key)
    if key:
        return InfoLang.from_api_key(
            key, namespace=namespace, workspace=workspace, base_url=base_url
        )
    # Fall back to the SDK's own env/credential resolution (e.g. INFOLANG_API_KEY,
    # INFOLANG_DEV_KEY, INFOLANG_NAMESPACE) when no explicit key is provided.
    return InfoLang(namespace=namespace, workspace=workspace, base_url=base_url)


def make_async_client(
    *,
    api_key: Secret | None,
    namespace: str | None,
    workspace: str | None,
    base_url: str | None,
) -> AsyncInfoLang:
    """Build an asynchronous :class:`infolang.AsyncInfoLang` from resolved config."""

    key = resolve_api_key(api_key)
    if key:
        return AsyncInfoLang.from_api_key(
            key, namespace=namespace, workspace=workspace, base_url=base_url
        )
    return AsyncInfoLang(namespace=namespace, workspace=workspace, base_url=base_url)
