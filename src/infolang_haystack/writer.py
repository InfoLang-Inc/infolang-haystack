"""``InfoLangWriter`` — persist documents/text into InfoLang memory."""

from __future__ import annotations

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret
from infolang import AsyncInfoLang, InfoLang

from ._client import default_api_key, make_async_client, make_sync_client


@component
class InfoLangWriter:
    """Write ``Document`` objects (or raw strings) into InfoLang memory.

    Uses the SDK's batched ``remember_batch`` so a list of documents becomes a
    single round-trip.

    ### Usage

    ```python
    from haystack import Document
    from infolang_haystack import InfoLangWriter

    writer = InfoLangWriter(namespace="my-app", source="ingest")
    writer.run(documents=[Document(content="Auth uses JWT in a cookie.")])
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret | None = None,
        namespace: str | None = None,
        workspace: str | None = None,
        base_url: str | None = None,
        source: str | None = None,
        tags: str | None = None,
        client: InfoLang | None = None,
        async_client: AsyncInfoLang | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else default_api_key()
        self.namespace = namespace
        self.workspace = workspace
        self.base_url = base_url
        self.source = source
        self.tags = tags
        self._client = client
        self._async_client = async_client

    # --- lifecycle ------------------------------------------------------

    def warm_up(self) -> None:
        self._sync()

    def _sync(self) -> InfoLang:
        if self._client is None:
            self._client = make_sync_client(
                api_key=self.api_key,
                namespace=self.namespace,
                workspace=self.workspace,
                base_url=self.base_url,
            )
        return self._client

    def _async(self) -> AsyncInfoLang:
        if self._async_client is None:
            self._async_client = make_async_client(
                api_key=self.api_key,
                namespace=self.namespace,
                workspace=self.workspace,
                base_url=self.base_url,
            )
        return self._async_client

    # --- serialization --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict() if self.api_key else None,
            namespace=self.namespace,
            workspace=self.workspace,
            base_url=self.base_url,
            source=self.source,
            tags=self.tags,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InfoLangWriter:
        init_params = data.get("init_parameters", {})
        secret = init_params.get("api_key")
        if isinstance(secret, dict):
            init_params["api_key"] = Secret.from_dict(secret)
        return default_from_dict(cls, data)

    # --- item building --------------------------------------------------

    def _build_items(
        self,
        documents: list[Document] | None,
        texts: list[str] | None,
        source: str | None,
        tags: str | None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        eff_source = source or self.source
        eff_tags = tags if tags is not None else self.tags
        for doc in documents or []:
            if not doc.content:
                continue
            meta = doc.meta or {}
            item: dict[str, Any] = {"text": doc.content}
            item_tags = eff_tags if eff_tags is not None else meta.get("tags")
            if item_tags is not None:
                item["tags"] = item_tags
            item_source = eff_source if eff_source is not None else meta.get("source")
            if item_source is not None:
                item["source"] = item_source
            items.append(item)
        for text in texts or []:
            if not text:
                continue
            item = {"text": text}
            if eff_tags is not None:
                item["tags"] = eff_tags
            if eff_source is not None:
                item["source"] = eff_source
            items.append(item)
        return items

    # --- run ------------------------------------------------------------

    @component.output_types(memory_ids=list[str], results=list[dict[str, Any]])
    def run(
        self,
        documents: list[Document] | None = None,
        texts: list[str] | None = None,
        namespace: str | None = None,
        source: str | None = None,
        tags: str | None = None,
    ) -> dict[str, Any]:
        """Store ``documents`` and/or ``texts`` in memory."""

        items = self._build_items(documents, texts, source, tags)
        if not items:
            return {"memory_ids": [], "results": []}
        results = self._sync().remember_batch(
            items, namespace=namespace or self.namespace, source=source or self.source
        )
        return {
            "memory_ids": [r.memory_id for r in results if r.memory_id],
            "results": [r.model_dump() for r in results],
        }

    @component.output_types(memory_ids=list[str], results=list[dict[str, Any]])
    async def run_async(
        self,
        documents: list[Document] | None = None,
        texts: list[str] | None = None,
        namespace: str | None = None,
        source: str | None = None,
        tags: str | None = None,
    ) -> dict[str, Any]:
        """Async mirror of :meth:`run`."""

        items = self._build_items(documents, texts, source, tags)
        if not items:
            return {"memory_ids": [], "results": []}
        results = await self._async().remember_batch(
            items, namespace=namespace or self.namespace, source=source or self.source
        )
        return {
            "memory_ids": [r.memory_id for r in results if r.memory_id],
            "results": [r.model_dump() for r in results],
        }
