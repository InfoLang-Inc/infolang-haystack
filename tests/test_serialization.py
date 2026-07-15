"""Pipeline-level serialization round trips prove the components are real 2.x components."""

from __future__ import annotations

from haystack import Pipeline

from infolang_haystack import InfoLangChatMemory, InfoLangRetriever, InfoLangWriter


def test_retriever_in_pipeline_roundtrip() -> None:
    pipe = Pipeline()
    pipe.add_component("recall", InfoLangRetriever(namespace="x", top_k=3))
    restored = Pipeline.from_dict(pipe.to_dict())
    assert "recall" in restored.to_dict()["components"]


def test_writer_in_pipeline_roundtrip() -> None:
    pipe = Pipeline()
    pipe.add_component("write", InfoLangWriter(namespace="x", source="s"))
    restored = Pipeline.from_dict(pipe.to_dict())
    assert "write" in restored.to_dict()["components"]


def test_memory_in_pipeline_roundtrip() -> None:
    pipe = Pipeline()
    pipe.add_component("memory", InfoLangChatMemory(namespace="x"))
    restored = Pipeline.from_dict(pipe.to_dict())
    assert "memory" in restored.to_dict()["components"]


def test_pipeline_yaml_roundtrip() -> None:
    pipe = Pipeline()
    pipe.add_component("recall", InfoLangRetriever(namespace="x", top_k=2))
    yaml_str = pipe.dumps()
    restored = Pipeline.loads(yaml_str)
    comp = restored.get_component("recall")
    assert isinstance(comp, InfoLangRetriever)
    assert comp.top_k == 2
