# infolang-haystack

First-party [InfoLang](https://infolang.ai) semantic-memory integration for
[Haystack](https://haystack.deepset.ai) 2.x.

It gives Haystack pipelines and agents long-term memory through the published
`infolang` Python SDK:

- **`InfoLangRetriever`** — recall memory as Haystack `Document`s.
- **`InfoLangWriter`** — persist `Document`s / text into memory (batched).
- **`InfoLangChatMemory`** — auto-recall context into a chat and auto-retain the
  user turn, in front of any ChatGenerator.
- **`recall_tool` / `remember_tool`** — `haystack.tools.Tool`s for agents.

Every component has a synchronous `run` and an asynchronous `run_async`, and is
fully serializable (`to_dict` / `from_dict`, pipeline YAML).

## Install

```bash
pip install infolang-haystack
```

This pulls the published SDK (`infolang>=0.2,<0.3`) and `haystack-ai>=2.9`.

## Configure

Set your key once; components read it from the environment by default:

```bash
export INFOLANG_API_KEY="il_live_..."
# optional:
export INFOLANG_NAMESPACE="my-app"     # default bank
export INFOLANG_WORKSPACE="acme"       # tenant
```

Scoping follows the InfoLang contract: `workspace` is your tenant, `namespace`
is the memory bank. Managed API keys honor `namespace` on both reads and writes.

## Quickstart

### Recall

```python
from infolang_haystack import InfoLangRetriever

retriever = InfoLangRetriever(namespace="my-app", top_k=5)
docs = retriever.run(query="How does auth middleware work?")["documents"]
for d in docs:
    print(d.score, d.content)
```

### Remember

```python
from haystack import Document
from infolang_haystack import InfoLangWriter

writer = InfoLangWriter(namespace="my-app", source="ingest")
writer.run(documents=[Document(content="Auth uses a JWT stored in a cookie.")])
# or: writer.run(texts=["Deploys run on Fridays."])
```

### Retrieval-augmented pipeline

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from infolang_haystack import InfoLangRetriever

pipe = Pipeline()
pipe.add_component("memory", InfoLangRetriever(namespace="my-app", top_k=5))
pipe.add_component("prompt", PromptBuilder(
    template="Context:\n{% for d in documents %}- {{ d.content }}\n{% endfor %}\nQ: {{ query }}"
))
pipe.connect("memory.documents", "prompt.documents")

result = pipe.run({"memory": {"query": "auth?"}, "prompt": {"query": "auth?"}})
print(result["prompt"]["prompt"])
```

### Chat memory (auto-recall + auto-retain)

```python
from haystack.dataclasses import ChatMessage
from infolang_haystack import InfoLangChatMemory

memory = InfoLangChatMemory(namespace="session-42")
out = memory.run(messages=[ChatMessage.from_user("What did we decide about auth?")])
enriched = out["messages"]   # inject the recalled context, then feed a ChatGenerator
```

`InfoLangChatMemory` injects recalled context as a system message and remembers
the latest user turn. For full-turn retention (user **and** assistant), place an
`InfoLangWriter` after your generator, or use the InfoLang LiteLLM proxy
integration, which sees both sides of the call.

### Tools for agents

```python
from infolang_haystack import recall_tool, remember_tool

tools = [recall_tool(namespace="my-app"), remember_tool(namespace="my-app")]
# pass `tools` to a Haystack Agent / ToolInvoker
```

### Async

```python
retriever = InfoLangRetriever(namespace="my-app")
docs = (await retriever.run_async(query="auth?"))["documents"]
```

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest            # offline; live tests are deselected by default
```

Live end-to-end tests hit a real endpoint and are opt-in:

```bash
INFOLANG_API_KEY=il_live_... pytest -m live
```

## License

Apache-2.0. See [LICENSE](LICENSE).
