# Configuration

Settings (`.env`), the organisation-specific files in `config/`, and the document manifest.

## Settings (.env)

All settings come from `.env` (see [.env.example](../.env.example)) through `kb.core.config.Settings`.

| Variable | Default | |
|---|---|---|
| `QDRANT_URL`, `QDRANT_COLLECTION` | `http://127.0.0.1:6444`, `kb_chunks` | Vector store |
| `KB_DB_PATH` | `data/kb.db` | SQLite database |
| `KB_DOCS_DIR`, `KB_MANIFEST_PATH` | `data/documents`, `config/manifest.csv` | Source documents and the manifest |
| `EMBED_MODEL_PATH`, `RERANK_MODEL_PATH`, `DOCLING_ARTIFACTS_PATH` | `models/...` | Local models |
| `HF_HUB_OFFLINE` | `1` | Never download models at runtime |
| `KB_RERANK_TOP` | `20` | Candidates reranked per search (see [architecture.md](architecture.md#retrieval)) |
| `KB_NOT_FOUND_SCORE` | `0.1` | Reply "not found" without the LLM below this top rerank score (see [architecture.md](architecture.md#answering)) |
| `KB_COMPARE_READ` | — | Disabled in code (commented out): comparisons would let the local LLM pick up to 2 more sections per side from each guide's table of contents; did not help with a 7B model |
| `KB_REFUSAL_RETRY` | `true` | When the model replies "not found", ask once more with only the best-matching sources (3, or 2 per side of a comparison) |
| `KB_LLM_PROVIDER` | `auto` | `auto` (the catalogue's answer role), `ollama` (the local fallback) or `openai`; superseded by `config/models.yaml` roles |
| `KB_USERS_PATH` | `config/users.yaml` | Chat API users and their access groups |
| `KB_API_HOST`, `KB_API_PORT` | `127.0.0.1`, `8000` | Where `kb serve` listens (keep loopback: no authentication) |
| `KB_DOMAIN_PATH` | `config/domain.yaml` | Organisation-specific text rules (see below) |
| `KB_GOLDEN_PATH` | `eval/golden.json` | Golden question set used by `kb coverage`, `kb eval`, `kb eval-answers` |
| `KB_MODELS_PATH` | `config/models.yaml` | Model catalogue: which language models exist and which job each one does (below). Without the file: `LLM_MODEL`, plus OpenAI when `OPENAI_*` is set |
| `LLM_TEMPERATURE`, `LLM_MAX_TOKENS` | `0`, `1500` | Ollama sampling temperature; answer length cap (both providers) |
| `OLLAMA_HOST`, `LLM_MODEL`, `LLM_NUM_CTX` | `127.0.0.1:11434`, `qwen2.5:7b-instruct`, `8192` | Local LLM |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | *(empty)* | Optional external LLM when there is no `config/models.yaml` (with a catalogue, OpenAI is a catalogue entry) |

## Organisation-specific data (kept local)

Nothing that comes from the ingested documents, and no rule that names their publisher, is committed. These files live only on your machine (git-ignored); the repository has fictional `*.example.*` versions to copy:

| Local file | Example in the repository | Holds |
|---|---|---|
| `config/manifest.csv` | [config/manifest.example.csv](../config/manifest.example.csv) | Which documents are ingested, their titles, releases, access groups |
| `config/users.yaml` | [config/users.example.yaml](../config/users.example.yaml) | Chat API users and their access groups |
| `config/domain.yaml` | [config/domain.example.yaml](../config/domain.example.yaml) | Text rules specific to whose documents you ingest (below) |
| `config/models.yaml` | [config/models.example.yaml](../config/models.example.yaml) | Language models (local, OpenAI, Mistral, Gemini, Claude) and the job of each (below) |
| `eval/golden.json` | [eval/golden.example.json](../eval/golden.example.json) | Golden questions with expected answers taken from the documents |

`config/domain.yaml` (loaded by `kb.core.domain`; every key optional, a key you set replaces the generic default):

- `boilerplate_patterns`: legal and footer lines removed wherever they appear (copyright, confidentiality notices).
- `command_patterns`: your products' command-prompt lines (e.g. `ACME>`), merged into code blocks when chunking, in addition to the generic SQL, shell, XML and web-server patterns.
- `reference_patterns`, `reference_label`, `reference_example`: the format of knowledge-base article numbers; the prompt asks the LLM to copy them, and article numbers in cited sections are appended to answers when the model leaves them out.
- `synonym_examples`: examples of "different words for the same thing" given to the LLM.

Changing `boilerplate_patterns` or `command_patterns` changes sections and chunks: run `kb chunk` and `kb index` afterwards. The tests never read the local `domain.yaml`; they use a fixed fictional domain (`tests/conftest.py`).

`config/models.yaml` (loaded by `kb.llm.catalogue`; check it with `uv run kb models list`):

- `models`: one entry per model, chosen by name. `adapter` (`ollama`, `openai_compatible` for OpenAI / Mistral / Gemini / local OpenAI-compatible servers, `anthropic` for Claude), `model` (the provider's model id), `location` (`local` or `external`), `base_url`, `api_key_env` (the name of the variable in `.env` that holds the key; keys never go in this file), `context_tokens`, `max_output_tokens`, `temperature` or `effort` (Claude), `timeout`, `json_mode`, `refusal_retry`, `compare_read`, `fallback`.
- `roles`: which model does which job: `answer`, `planner` (comparison split), `condenser` (follow-up rewrite), `judge` (faithfulness). A role left out uses the fallback.
- `fallback`: a local model; it answers whenever an external model may not see the context.

An external model only ever sees sources whose manifest row has `external_ok = true`. Planner and condenser see the raw question and the chat history; `kb models list` warns when either is external.

## The document manifest

`config/manifest.csv` (example: [config/manifest.example.csv](../config/manifest.example.csv)) holds, for every source file, the metadata that can't be read reliably from the file itself. It is copied into every chunk and drives retrieval filters.

| Column | Example | Purpose |
|---|---|---|
| `doc_id` | `install-guide` | Stable ID used in citations and re-ingestion |
| `path` | `Acme_Platform_Installation_Guide.pdf` | File, relative to `KB_DOCS_DIR` |
| `title` | Acme Platform Installation Guide | Shown in answer citations |
| `family`, `version` | `install-guide`, `2.0` | The highest version per family is the latest revision |
| `release_min`, `release_max` | `R2024x`, *(blank)* | Release range the document applies to; blank = open-ended |
| `allowed_groups` | `all` or `internal` | `;`-separated access groups |
| `external_ok` | `true` | May its text be sent to OpenAI |
| `category` | `installation` | One of: installation, administration, authentication, infrastructure, upgrade, performance, applications, functional, troubleshooting (defined in `kb.manifest.CATEGORIES`) |

Adding documents:

```bash
uv run kb manifest scan       # appends draft rows for new files (release guessed from filename)
uv run kb manifest validate   # reports every problem with its line number
```

Fill in title, category, groups and release range between the two commands.
