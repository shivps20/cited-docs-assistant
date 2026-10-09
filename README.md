# Knowledge-Base Assistant

A local, retrieval-augmented question-answering assistant over technical documentation (PDF, PPTX, DOCX). It answers single-fact lookups, how-to questions and cross-document comparisons, with every answer citing document, section and page.

Everything runs on one workstation: Qdrant in Docker, bge-m3 embeddings and the bge-reranker in-process, and a local LLM through Ollama (OpenAI optional, per-document opt-in).

> **Status:** ingestion, retrieval, cited answers, evaluation, chat API and web UI are working. See [Documentation](#documentation) for the commands, settings, API and evaluation. Project notes are kept locally (`docs/local/`), not in the repository.

## How it works

```
Ingestion (offline)                         Query (per request)
-------------------                         -------------------
documents + manifest                        question
  -> Docling parse                            -> condense follow-up, route (comparison: one search per side)
  -> structure-aware chunks                   -> hybrid search in Qdrant (dense + sparse, RRF)
  -> metadata from manifest                   -> filters: access groups, release, latest revision
  -> bge-m3 dense + sparse                    -> rerank -> confidence gate -> assemble context
  -> Qdrant (chunks) + SQLite (sections)      -> LLM answer with [n] citations
                                              -> "not found"? once more with the best 3 sources
```

Every query is traced stage by stage into SQLite, and a golden question set (`eval/golden.json`, local; example in [eval/golden.example.json](eval/golden.example.json)) measures retrieval and answer quality.

## Requirements

| Component | Version used | Notes |
|---|---|---|
| Windows 11 / Linux | | Developed on Windows 11 |
| NVIDIA GPU + driver | RTX A1000 6 GB, driver 596 (CUDA 13.2) | GPU used for ingestion; queries run on CPU + Ollama |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | 4.93 | Runs Qdrant |
| [uv](https://docs.astral.sh/uv/) | 0.11 | Python and dependency management |
| [Ollama](https://ollama.com/) | 0.30 | Local LLM `qwen2.5:7b-instruct` |
| Disk | ~10 GB | Models (~6 GB) + CUDA torch |

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
7. **Documents and local project data:** put the source files in `data/documents/` (or set `KB_DOCS_DIR`) and list them in `config/manifest.csv`. Organisation-specific data is never committed; start from the fictional examples:
   ```bash
   cp config/manifest.example.csv config/manifest.csv
   cp config/users.example.yaml config/users.yaml
   cp config/domain.example.yaml config/domain.yaml
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

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Ingestion, retrieval and answering pipelines; design decisions |
| [docs/commands.md](docs/commands.md) | Every `kb` command with options and examples, typical workflows |
| [docs/configuration.md](docs/configuration.md) | `.env` settings, `config/` files, organisation-specific rules, the document manifest |
| [docs/api.md](docs/api.md) | `kb serve`, HTTP endpoints, chat events, the web UI |
| [docs/evaluation.md](docs/evaluation.md) | Golden set, `kb eval`, `kb eval-answers`, results |

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
  llm/             catalogue (models.yaml) · providers (Ollama, OpenAI) · prompts (prompt, citations) · condense (follow-ups) · judge
  agent/           route (comparison?) · tools (search_kb, get_section, outline, read_section; access-checked) · compare (split, one search per side; read step per model)
  answer/          pipeline: route -> retrieve -> gate -> LLM -> citations (-> retry on a refusal), in one trace
  evaluation/      coverage · retrieval (kb eval) · answers (kb eval-answers)
  api/             app (FastAPI) · chat (SSE turn) · services · health · users · sessions · static/ (chat UI)
  cli/             main (entry point) · ingest · search · evaluation · serve
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
