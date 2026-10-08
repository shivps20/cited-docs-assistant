# Architecture

How a question is answered, from the document manifest to a cited answer. For commands see [commands.md](commands.md), for settings [configuration.md](configuration.md).

## Overview

```
Ingestion (offline)                         Query (per request)
-------------------                         -------------------
documents + manifest                        question
  -> Docling parse                            -> condense follow-up, route
  -> structure-aware chunks                   -> hybrid search in Qdrant (dense + sparse, RRF)
  -> metadata from manifest                   -> filters: access groups, release, latest revision
  -> bge-m3 dense + sparse                    -> rerank -> confidence gate -> assemble context
  -> Qdrant (chunks) + SQLite (sections)      -> LLM answer with [n] citations
```

Every query is traced stage by stage into SQLite, and a golden question set (`eval/golden.json`, local; example in [eval/golden.example.json](../eval/golden.example.json)) measures retrieval and answer quality.

## Retrieval

```bash
uv run kb search "Which port does the application server use?"            # hybrid search + rerank, public docs
uv run kb search "..." --groups internal --release R2024x     # with access group and release filter
uv run kb search "..." --mode dense --no-rerank --context        # compare modes, print assembled context
```

Query embedding and reranking run on the CPU (the GPU stays free for the LLM). Every search is traced in the `traces` / `trace_stages` tables.

Search retrieves 30 candidates (dense + sparse, fused), reranks the first `KB_RERANK_TOP` of them with the cross-encoder, then assembles up to 6 context units (3,000 tokens) for the LLM; near-identical sections (the same section in two variants of a guide) are sent once, and the copy is named in the source line ("· same text: …") so both documents are cited.

### Tuning flags

| Flag | Default | Meaning |
|---|---|---|
| `--rerank-top N` | `KB_RERANK_TOP` (20) | Rerank only the first N candidates; the rest keep search order. `0` reranks all. |
| `--max-length N` | 512 | Reranker input length in tokens; longer chunks are truncated. |
| `--min-score X` | off | Drop context chunks whose rerank score is below X (0–1). Unreranked chunks are dropped too. |

Change the default depth in `.env` with `KB_RERANK_TOP=<N>`.

## Answering

```bash
uv run kb ask "How do I enable single sign-on?"
uv run kb ask "..." --groups internal --release R2025x --show-context
```

`kb ask` runs retrieval, then:

1. **Gate:** if nothing was found, or the best rerank score is below `KB_NOT_FOUND_SCORE` (0.1), it replies "not found" without calling the LLM.
2. **Prompt:** the context goes to the LLM as numbered sources (document, release, section, pages). The rules: answer only from the sources, cite `[n]` after every statement, copy article numbers, URLs, commands and queries exactly, give procedures as numbered steps, and reply with a fixed "not found" sentence when the sources do not contain the answer (this catches on-topic near misses that the gate lets through).
3. **Provider:** Ollama (`LLM_MODEL`) by default. With `OPENAI_API_KEY` and `OPENAI_MODEL` set, `KB_LLM_PROVIDER=auto` uses OpenAI only when **every** source has `external_ok = true`; otherwise it answers locally and says why. If OpenAI fails, it falls back to Ollama.
4. **Citations:** `[n]` markers that do not match a source are removed; the **Sources** list under the answer is built from the cited numbers, never written by the LLM.

The answer streams as it is generated (~25 tokens/s with qwen2.5 7B on the 6 GB GPU; the first question after a pause also loads the model, 5–25 s). The whole run (retrieval, gate, generation, citations) is one trace with `route = 'answer'`.
