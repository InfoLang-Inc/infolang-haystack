"""``InfoLangChatMemory`` — auto-recall context and auto-retain user turns.

Drop this component in front of a ChatGenerator: it recalls memory for the
latest user turn, injects it as a system message, and (optionally) remembers
the user turn so future runs can recall it.
"""

from __future__ import annotations

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils import Secret
from infolang import AsyncInfoLang, InfoLang

from ._client import default_api_key, make_async_client, make_sync_client
from ._convert import format_context, recall_to_documents

DEFAULT_HEADER = "Relevant memory (InfoLang):"


@component
class InfoLangChatMemory:
    """Inject recalled context into a chat and retain the latest user turn.

    ### Usage

    ```python
    from haystack.dataclasses import ChatMessage
    from infolang_haystack import InfoLangChatMemory

    memory = InfoLangChatMemory(namespace="session-42")
    out = memory.run(messages=[ChatMessage.from_user("What did we decide on auth?")])
    enriched = out["messages"]  # feed into a ChatGenerator
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
        auto_recall: bool = True,
        auto_retain: bool = True,
        header: str = DEFAULT_HEADER,
        source: str | None = "haystack",
        tags: str | None = None,
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
        self.auto_recall = auto_recall
        self.auto_retain = auto_retain
        self.header = header
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
            top_k=self.top_k,
            filters=self.filters,
            score_threshold=self.score_threshold,
            auto_recall=self.auto_recall,
            auto_retain=self.auto_retain,
            header=self.header,
            source=self.source,
            tags=self.tags,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InfoLangChatMemory:
        init_params = data.get("init_parameters", {})
        secret = init_params.get("api_key")
        if isinstance(secret, dict):
            init_params["api_key"] = Secret.from_dict(secret)
        return default_from_dict(cls, data)

    # --- helpers --------------------------------------------------------

    @staticmethod
    def _last_user_text(messages: list[ChatMessage]) -> str | None:
        for message in reversed(messages):
            if message.is_from(ChatRole.USER):
                text = message.text
                if text and text.strip():
                    return text
        return None

    @staticmethod
    def _inject(messages: list[ChatMessage], block: str) -> list[ChatMessage]:
        """Insert a system message after any leading system messages."""

        context_message = ChatMessage.from_system(block)
        index = 0
        for message in messages:
            if message.is_from(ChatRole.SYSTEM):
                index += 1
            else:
                break
        return [*messages[:index], context_message, *messages[index:]]

    # --- run ------------------------------------------------------------

    @component.output_types(messages=list[ChatMessage], documents=list[Document])
    def run(
        self, messages: list[ChatMessage], namespace: str | None = None
    ) -> dict[str, Any]:
        """Recall + inject context for, and optionally retain, the latest user turn."""

        ns = namespace or self.namespace
        query = self._last_user_text(messages)
        out = list(messages)
        documents: list[Document] = []
        if self.auto_recall and query:
            result = self._sync().recall(
                query, namespace=ns, top_k=self.top_k, filters=self.filters
            )
            documents = recall_to_documents(result, score_threshold=self.score_threshold)
            block = format_context(
                result, header=self.header, score_threshold=self.score_threshold
            )
            if block:
                out = self._inject(out, block)
        if self.auto_retain and query:
            self._sync().remember(query, namespace=ns, source=self.source, tags=self.tags)
        return {"messages": out, "documents": documents}

    @component.output_types(messages=list[ChatMessage], documents=list[Document])
    async def run_async(
        self, messages: list[ChatMessage], namespace: str | None = None
    ) -> dict[str, Any]:
        """Async mirror of :meth:`run`."""

        ns = namespace or self.namespace
        query = self._last_user_text(messages)
        out = list(messages)
        documents: list[Document] = []
        if self.auto_recall and query:
            result = await self._async().recall(
                query, namespace=ns, top_k=self.top_k, filters=self.filters
            )
            documents = recall_to_documents(result, score_threshold=self.score_threshold)
            block = format_context(
                result, header=self.header, score_threshold=self.score_threshold
            )
            if block:
                out = self._inject(out, block)
        if self.auto_retain and query:
            await self._async().remember(
                query, namespace=ns, source=self.source, tags=self.tags
            )
        return {"messages": out, "documents": documents}
