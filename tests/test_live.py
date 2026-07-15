"""Live round trip against a real InfoLang endpoint.

Deselected by default (``-m 'not live'``). Run explicitly with credentials:

    INFOLANG_API_KEY=il_live_... pytest -m live
"""

from __future__ import annotations

import os
import uuid

import pytest

from infolang_haystack import InfoLangRetriever, InfoLangWriter

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    not os.getenv("INFOLANG_API_KEY"), reason="INFOLANG_API_KEY not set"
)
def test_live_remember_then_recall() -> None:
    namespace = os.getenv("INFOLANG_TEST_NAMESPACE", "infolang-haystack-live")
    sentinel = f"live-sentinel-{uuid.uuid4().hex[:8]}"

    writer = InfoLangWriter(namespace=namespace, source="live-test")
    written = writer.run(texts=[f"The InfoLang Haystack live sentinel is {sentinel}."])
    assert written["memory_ids"]

    retriever = InfoLangRetriever(namespace=namespace, top_k=5)
    docs = retriever.run(query="What is the Haystack live sentinel?")["documents"]
    assert isinstance(docs, list)
