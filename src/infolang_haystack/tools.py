"""Haystack ``Tool`` factories for agent-style recall and remember.

These build runtime :class:`haystack.tools.Tool` objects an agent/LLM can call.
Each tool closes over an InfoLang client, so pass a preconfigured ``client`` or
credentials for it to build one.
"""

from __future__ import annotations

from typing import Any

from haystack.tools import Tool
from haystack.utils import Secret
from infolang import InfoLang

from ._client import make_sync_client

DEFAULT_RECALL_DESCRIPTION = (
    "Search long-term semantic memory (InfoLang) for context relevant to a query. "
    "Returns the most similar remembered chunks with their similarity scores."
)
DEFAULT_REMEMBER_DESCRIPTION = (
    "Save a fact or note to long-term semantic memory (InfoLang) so it can be "
    "recalled later. Provide the text to remember and optional comma-separated tags."
)


def _client_or_build(
    client: InfoLang | None,
    api_key: Secret | None,
    namespace: str | None,
    workspace: str | None,
    base_url: str | None,
) -> InfoLang:
    if client is not None:
        return client
    return make_sync_client(
        api_key=api_key, namespace=namespace, workspace=workspace, base_url=base_url
    )


def recall_tool(
    *,
    client: InfoLang | None = None,
    api_key: Secret | None = None,
    namespace: str | None = None,
    workspace: str | None = None,
    base_url: str | None = None,
    top_k: int = 5,
    name: str = "infolang_recall",
    description: str | None = None,
) -> Tool:
    """Build a Tool that recalls context from InfoLang memory."""

    resolved = _client_or_build(client, api_key, namespace, workspace, base_url)
    default_top_k = top_k

    def _recall(query: str, top_k: int = default_top_k) -> dict[str, Any]:
        result = resolved.recall(query, namespace=namespace, top_k=top_k)
        return {
            "chunks": [
                {"id": c.id, "text": c.text, "score": c.score, "tags": c.tags}
                for c in result.chunks
            ]
        }

    return Tool(
        name=name,
        description=description or DEFAULT_RECALL_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search memory for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of chunks to return.",
                    "default": default_top_k,
                },
            },
            "required": ["query"],
        },
        function=_recall,
    )


def remember_tool(
    *,
    client: InfoLang | None = None,
    api_key: Secret | None = None,
    namespace: str | None = None,
    workspace: str | None = None,
    base_url: str | None = None,
    source: str | None = None,
    name: str = "infolang_remember",
    description: str | None = None,
) -> Tool:
    """Build a Tool that stores a note in InfoLang memory."""

    resolved = _client_or_build(client, api_key, namespace, workspace, base_url)
    default_source = source

    def _remember(text: str, tags: str | None = None) -> dict[str, Any]:
        result = resolved.remember(
            text, namespace=namespace, source=default_source, tags=tags
        )
        return {"memory_id": result.memory_id, "namespace": result.namespace}

    return Tool(
        name=name,
        description=description or DEFAULT_REMEMBER_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The fact or note to remember.",
                },
                "tags": {
                    "type": "string",
                    "description": "Optional comma-separated tags.",
                },
            },
            "required": ["text"],
        },
        function=_remember,
    )
