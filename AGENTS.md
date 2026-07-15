# infolang-haystack — agent instructions

InfoLang semantic-memory integration for **Haystack 2.x**. Package import name:
`infolang_haystack`; PyPI name: `infolang-haystack`.

## Frozen contract

- Depend **only** on the published public SDK: `infolang>=0.2,<0.3` (PyPI) and
  `haystack-ai>=2.9,<3`. Never reimplement HTTP, import runtime/engine
  internals, or reference core-ip.
- SDK surface used: `from infolang import InfoLang, AsyncInfoLang`;
  `recall(query, namespace=, top_k=, filters=)`, `remember(text, namespace=,
  source=, tags=)`, `remember_batch(items, namespace=, source=)`.
- Scoping: `workspace` = tenant, `namespace` = bank.

## Architecture

- `src/infolang_haystack/_client.py` — builds sync/async SDK clients from a
  Haystack `Secret` + config.
- `src/infolang_haystack/_convert.py` — pure Chunk→Document / context-block shaping
  shared by the sync and async paths.
- `retriever.py` — `InfoLangRetriever` (recall → Documents).
- `writer.py` — `InfoLangWriter` (Documents/text → memory via `remember_batch`).
- `memory.py` — `InfoLangChatMemory` (auto-recall inject + auto-retain).
- `tools.py` — `recall_tool` / `remember_tool` → `haystack.tools.Tool`.

## Rules

- Sync `run` and async `run_async` must stay in lockstep — share shaping in
  `_convert.py`, never duplicate it.
- Tests mock the InfoLang client (offline default). Live tests carry the
  `live` marker and are deselected unless explicitly run.

## Commands

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest
```
