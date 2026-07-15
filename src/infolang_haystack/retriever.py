"""``InfoLangRetriever`` — a Haystack retriever backed by InfoLang recall."""

from __future__ import annotations

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret
from infolang import AsyncInfoLang, InfoLang

from ._client import default_api_key, make_async_client, make_sync_client
from ._convert import chunk_to_dict, recall_to_documents


@component
class InfoLangRetriever:
    """Retrieve documents from InfoLang semantic memory.

    Wraps the published SDK's ``recall`` (``InfoLang.recall``) and returns
    Haystack ``Document`` objects, so it drops into any 2.x pipeline where a
    retriever is expected.

    ### Usage

    ```python
    from infolang_haystack import InfoLangRetriever

    retriever = InfoLangRetriever(namespace="my-app", top_k=5)
    docs = retriever.run(query="How does auth middleware work?")["documents"]
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret | None = None,
        namespace: str | None = None,
        workspace: str | None = None,
        base_url: str | None = None,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        client: InfoLang | None = None,
        async_client: AsyncInfoLang | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")
        self.api_key = api_key if api_key is not None else default_api_key()
        self.namespace = namespace
        self.workspace = workspace
        self.base_url = base_url
        self.top_k = top_k
        self.filters = filters
        self.score_threshold = score_threshold
        self._client = client
        self._async_client = async_client

    # --- lifecycle ------------------------------------------------------

    def warm_up(self) -> None:
        """Eagerly build the sync client (optional; ``run`` builds lazily)."""

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
            top_k=self.top_k,
            filters=self.filters,
            score_threshold=self.score_threshold,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InfoLangRetriever:
        init_params = data.get("init_parameters", {})
        secret = init_params.get("api_key")
        if isinstance(secret, dict):
            init_params["api_key"] = Secret.from_dict(secret)
        return default_from_dict(cls, data)

    # --- run ------------------------------------------------------------

    @component.output_types(documents=list[Document], chunks=list[dict[str, Any]])
    def run(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Recall documents for ``query``."""

        result = self._sync().recall(
            query,
            namespace=namespace or self.namespace,
            top_k=top_k or self.top_k,
            filters=filters if filters is not None else self.filters,
        )
        return {
            "documents": recall_to_documents(result, score_threshold=self.score_threshold),
            "chunks": [chunk_to_dict(c) for c in result.chunks],
        }

    @component.output_types(documents=list[Document], chunks=list[dict[str, Any]])
    async def run_async(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Async mirror of :meth:`run`."""

        result = await self._async().recall(
            query,
            namespace=namespace or self.namespace,
            top_k=top_k or self.top_k,
            filters=filters if filters is not None else self.filters,
        )
        return {
            "documents": recall_to_documents(result, score_threshold=self.score_threshold),
            "chunks": [chunk_to_dict(c) for c in result.chunks],
        }
