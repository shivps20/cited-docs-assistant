# Knowledge-Base Assistant

A local, retrieval-augmented question-answering assistant over technical documentation (PDF, PPTX, DOCX). It answers single-fact lookups, how-to questions and cross-document comparisons, with every answer citing document, section and page.

Everything runs on one workstation: Qdrant in Docker, bge-m3 embeddings and the bge-reranker in-process, and a local LLM through Ollama. Other models (OpenAI, Mistral, Gemini, Claude) can be added in `config/models.yaml` and chosen per question; they only ever see documents cleared for external use (`external_ok` in the manifest).

> **Status:** ingestion, retrieval, cited answers (including comparisons across documents), evaluation (including calibration of the "not found" threshold), chat API, web UI, a configurable choice of language model and an answer cache for repeated questions are working. See [Documentation](#documentation) for the commands, settings, API and evaluation. Project notes are kept locally (`docs/local/`), not in the repository.

## How it works

```
Ingestion (offline)                         Query (per request)
-------------------                         -------------------
documents + manifest                        question
  -> Docling parse                            -> condense follow-up; asked before? answer from the cache
                                              -> route (comparison: one search per side)
  -> structure-aware chunks                   -> hybrid search in Qdrant (dense + sparse, RRF)
  -> metadata from manifest                   -> filters: access groups, release, latest revision
  -> bge-m3 dense + sparse                    -> rerank -> confidence gate -> assemble context
  -> Qdrant (chunks) + SQLite (sections)      -> LLM answer with [n] citations
                                              -> "not found"? once more with the best 3 sources
```

Every query is traced stage by stage into SQLite, and a golden question set (`eval/golden.json`, local; example in [eval/golden.example.json](eval/golden.example.json)) measures retrieval and answer quality.

## Requirements

<table style="width:100%">
<colgroup><col style="width:23%"><col style="width:21%"><col style="width:56%"></colgroup>
<thead><tr><th>Component</th><th>Version used</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>Windows 11 / Linux</td><td></td><td>Developed on Windows 11</td></tr>
<tr><td>NVIDIA GPU + driver</td><td>RTX A1000 6 GB, driver 596 (CUDA 13.2)</td><td>GPU used for ingestion; queries run on CPU + Ollama</td></tr>
<tr><td><a href="https://www.docker.com/products/docker-desktop/">Docker Desktop</a></td><td>4.93</td><td>Runs Qdrant</td></tr>
<tr><td><a href="https://docs.astral.sh/uv/">uv</a></td><td>0.11</td><td>Python and dependency management</td></tr>
<tr><td><a href="https://ollama.com/">Ollama</a></td><td>0.30</td><td>Local LLM <code>qwen2.5:7b-instruct</code></td></tr>
<tr><td>Disk</td><td>~10 GB</td><td>Models (~6 GB) + CUDA torch</td></tr>
</tbody>
</table>

## Setup

```bash
git clone <repository-url> knowledgebase_assistant
cd knowledgebase_assistant
```

1. **Python environment** (Python 3.13 and CUDA torch from the PyTorch index, see `pyproject.toml`):
   ```bash
   uv sync
   ```
2. **Configuration:**
   ```bash
   cp .env.example .env
   ```
3. **Qdrant** (REST + dashboard on `127.0.0.1:6444`, data in Docker named volumes):
   ```bash
   docker compose -f deploy/docker-compose.yml up -d
   ```
4. **Collection and database:**
   ```bash
   uv run python scripts/init_qdrant.py
   uv run python scripts/init_db.py
   ```
5. **Models** (~6 GB into `./models`, then verified offline):
   ```bash
   uv run python scripts/download_models.py
   ```
6. **LLM:**
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```
7. **Documents and local project data:** put documents in `data/documents/` (or set `KB_DOCS_DIR`), or leave them where they are and add a folder with `uv run kb manifest scan --folder PATH`; review the drafted rows in `config/manifest.csv`. PDF, PPTX and DOCX are read directly; `.ppt` / `.doc` need LibreOffice (converted once before parsing). Organisation-specific data is never committed; start from the fictional examples:
   ```bash
   cp config/manifest.example.csv config/manifest.csv
   cp config/users.example.yaml config/users.yaml
   cp config/domain.example.yaml config/domain.yaml
   cp config/models.example.yaml config/models.yaml     # optional: language models (keys go in .env)
   cp eval/golden.example.json eval/golden.json
   ```
   See [docs/configuration.md](docs/configuration.md).
8. **Check everything:**
   ```bash
   uv run python scripts/check_env.py
   ```
   `--deep` also loads the LLM and generates a few tokens.

## Daily use

1. Docker Desktop running (Qdrant restarts with it).
2. Ollama running.
3. `uv run python scripts/check_env.py`: all green.

Qdrant dashboard: <http://127.0.0.1:6444/dashboard>. Chat UI: `uv run kb serve`, then <http://127.0.0.1:8000>. Browse the SQLite database (`data/kb.db`) read-only with DB Browser for SQLite.

The 6 GB GPU can't hold the embedding model and the LLM at once: run ingestion while Ollama has no model loaded.

## Documentation

<table style="width:100%">
<colgroup><col style="width:40%"><col style="width:60%"></colgroup>
<thead><tr><th>Document</th><th>Contents</th></tr></thead>
<tbody>
<tr><td><a href="docs/architecture.md">docs/architecture.md</a></td><td>Ingestion, retrieval and answering pipelines; design decisions</td></tr>
<tr><td><a href="docs/commands.md">docs/commands.md</a></td><td>Every <code>kb</code> command with options and examples, typical workflows</td></tr>
<tr><td><a href="docs/configuration.md">docs/configuration.md</a></td><td><code>.env</code> settings, <code>config/</code> files, organisation-specific rules, the document manifest</td></tr>
<tr><td><a href="docs/api.md">docs/api.md</a></td><td><code>kb serve</code>, HTTP endpoints, chat events, the web UI</td></tr>
<tr><td><a href="docs/evaluation.md">docs/evaluation.md</a></td><td>Golden set, <code>kb eval</code>, <code>kb eval-answers</code>, <code>kb calibrate</code>, results</td></tr>
</tbody>
</table>

## Project layout

```
config/            settings files: *.example.* committed; manifest.csv, users.yaml, domain.yaml local (git-ignored)
deploy/            docker-compose.yml (Qdrant)
docs/              architecture, commands, configuration, API, evaluation (docs/local/: personal notes, git-ignored)
eval/              golden.example.json (golden.json local) · check_golden.py
scripts/           init_qdrant, init_db, download_models, check_env
src/kb/
  core/            config (settings) · db (SQLite schema) · tracing · domain (organisation rules) · perf · progress
  ingest/          manifest · parse (Docling) · structure (sections) · chunk · index (Qdrant)
  store/           embed (bge-m3 dense + sparse) · vectorstore (Qdrant collection)
  retrieve/        search (filters, hybrid) · rerank · assemble · pipeline · gate · release
  llm/             catalogue (models.yaml) · providers (adapters: Ollama, OpenAI-compatible, Anthropic) · registry (roles, privacy, fallbacks) · prompts · condense · judge
  agent/           route (comparison?) · tools (search_kb, get_section, outline, read_section; access-checked) · compare (split, one search per side; read step per model)
  answer/          pipeline: cache -> route -> retrieve -> gate -> LLM -> citations (-> retry on a refusal), in one trace · cache
  evaluation/      coverage · retrieval (kb eval) · answers (kb eval-answers) · calibration (kb calibrate)
  api/             app (FastAPI) · chat (SSE turn) · services · health · users · sessions · static/ (chat UI)
  cli/             main (entry point) · ingest · search · evaluation · models · cache · serve
tests/             mirrors src/kb (core, ingest, retrieve, agent, answer, evaluation, api)
data/, models/     local runtime data (documents, parse cache, kb.db, reports) and models: git-ignored
```

## Development

```bash
uv run pytest
uv run ruff check src tests scripts eval
uv run python eval/check_golden.py   # after editing eval/golden.json
```

Database schema changes go into `kb.core.db.MIGRATIONS` as a new entry (never edit an applied one); `scripts/init_db.py` applies them.
