"""InfoLang integration for Haystack 2.x.

Components and tools that connect Haystack pipelines to InfoLang semantic
memory through the published ``infolang`` Python SDK.

Quickstart::

    from infolang_haystack import InfoLangRetriever

    retriever = InfoLangRetriever(namespace="my-app", top_k=5)
    docs = retriever.run(query="How does auth middleware work?")["documents"]
"""

from __future__ import annotations

from ._version import __version__
from .memory import InfoLangChatMemory
from .retriever import InfoLangRetriever
from .tools import recall_tool, remember_tool
from .writer import InfoLangWriter

__all__ = [
    "__version__",
    "InfoLangRetriever",
    "InfoLangWriter",
    "InfoLangChatMemory",
    "recall_tool",
    "remember_tool",
]
