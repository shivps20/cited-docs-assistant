# Configuration

Settings (`.env`), the organisation-specific files in `config/`, and the document manifest.

## Settings (.env)

All settings come from `.env` (see [.env.example](../.env.example)) through `kb.core.config.Settings`.

| Variable | Default | |
|---|---|---|
| `QDRANT_URL`, `QDRANT_COLLECTION` | `http://127.0.0.1:6444`, `kb_chunks` | Vector store |
| `KB_DB_PATH` | `data/kb.db` | SQLite database |
| `KB_DOCS_DIR`, `KB_MANIFEST_PATH` | `data/documents`, `config/manifest.csv` | The documents folder (subfolders included; manifest paths are relative to it) and the manifest. Documents elsewhere are added with `kb manifest scan --folder PATH` and read in place (absolute paths in the manifest) |
| `KB_DOC_TYPES` | `pdf,pptx,ppt,docx,doc` | File types picked up by `kb manifest scan` and accepted in the manifest. `.ppt` / `.doc` (old binary formats) are converted once to `.pptx` / `.docx` with LibreOffice before parsing, cached in `data/parsed/converted/` |
| `KB_SOFFICE` | *(found automatically)* | Path to LibreOffice's `soffice` when it is not on the PATH or in `C:/Program Files/LibreOffice` |
| `KB_SOURCE_PATH` | `full` | How every source shows its file (in `kb ask`, `kb search`, the chat's Sources and context panel, evaluation reports): `full` = the absolute path; `relative` = the path relative to `KB_DOCS_DIR`, or only the file name for a document kept elsewhere (for a server: no disk layout shown). Display only: stored answers, traces and the answer cache keep the manifest form, so switching it changes nothing else |
| `EMBED_MODEL_PATH`, `RERANK_MODEL_PATH`, `DOCLING_ARTIFACTS_PATH` | `models/...` | Local models |
| `HF_HUB_OFFLINE` | `1` | Never download models at runtime |
| `KB_RERANK_TOP` | `20` | Candidates reranked per search (see [architecture.md](architecture.md#retrieval)) |
| `KB_NOT_FOUND_SCORE` | `0.1` | Reply "not found" without the LLM below this top rerank score (see [architecture.md](architecture.md#answering)); check a value with `kb calibrate` ([evaluation.md](evaluation.md#kb-calibrate--the-not-found-threshold)) |
| `KB_REFUSAL_RETRY` | `true` | When the model replies "not found", ask once more with only the best-matching sources (3, or 2 per side of a comparison) |
| `KB_ANSWER_CACHE` | `true` | Answer a repeated question from the answer cache (same question, groups, release, corpus, model and prompt; see [architecture.md](architecture.md#answer-cache)) |
| `KB_LLM_PROVIDER` | `auto` | `auto` (the catalogue's answer role), `ollama` (the local fallback) or `openai`; superseded by `config/models.yaml` roles |
| `KB_USERS_PATH` | `config/users.yaml` | Chat API users and their access groups |
| `KB_API_HOST`, `KB_API_PORT` | `127.0.0.1`, `8000` | Where `kb serve` listens (keep loopback: no authentication) |
| `KB_DOMAIN_PATH` | `config/domain.yaml` | Organisation-specific text rules (see below) |
| `KB_GOLDEN_PATH` | `eval/golden.json` | Golden question set used by `kb coverage`, `kb eval`, `kb eval-answers`, `kb calibrate` and `eval/check_golden.py` |
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

`kb serve` re-reads `models.yaml` when the file changes (no restart); an edit with errors is ignored and logged, and the previous catalogue stays. An external model only ever sees sources whose manifest row has `external_ok = true`. Planner and condenser see the raw question and the chat history; `kb models list` warns when either is external.

**Per-model behaviour.** The context sent with a question is sized for the model that answers: a quarter of what its window leaves after the answer, never less than the defaults and at most 12,000 tokens. An 8k model with a 1,500-token answer gets the defaults (6 sources, 3,000 tokens; comparisons 3 per side, 4,000 tokens); a 200k model gets up to 16 sources, 12,000 tokens (8 per side, 16,000 for comparisons). When a smaller model answers instead (privacy fallback, or the chosen model failed), it gets the best-ranked part of the same context, with a notice. `refusal_retry` (ask again with the best sources after a "not found") and `compare_read` (the comparison read step, TO-5.10) are switched per model; `KB_REFUSAL_RETRY=false` turns the retry off for every model.

## The document manifest

`config/manifest.csv` (example: [config/manifest.example.csv](../config/manifest.example.csv)) holds, for every source file, the metadata that can't be read reliably from the file itself. It is copied into every chunk and drives retrieval filters.

| Column | Example | Purpose |
|---|---|---|
| `doc_id` | `install-guide` | Stable ID used in citations and re-ingestion |
| `path` | `Install/Acme_Platform_Installation_Guide.pdf` | File, relative to `KB_DOCS_DIR` (may include subfolders), or an absolute path such as `E:/Docs/New/Guide.pptx` for a document kept outside it |
| `title` | Acme Platform Installation Guide | Shown in answer citations |
| `family`, `version` | `install-guide`, `2.0` | The highest version per family is the latest revision |
| `release_min`, `release_max` | `R2024x`, *(blank)* | Release range the document applies to; blank = open-ended |
| `allowed_groups` | `all` or `internal` | `;`-separated access groups |
| `external_ok` | `true` | May its text be sent to an external LLM (OpenAI, Mistral, Gemini, Claude) |
| `category` | `installation` | One of: installation, administration, authentication, infrastructure, upgrade, performance, applications, functional, troubleshooting (defined in `kb.manifest.CATEGORIES`) |
| `added` | `2026-10-10` | Date the row was added (`YYYY-MM-DD`, blank allowed); filled by `kb manifest scan` and, for older rows, `kb manifest backfill` |
| `review` | `category guessed from the name` | What still needs a human check; blank = reviewed. `kb manifest validate` counts and lists these rows |

How `kb manifest scan` drafts a row (every guess is written into `review`):

- **doc_id / family** from the whole file name without its version (no cut at 60 characters; names over 100 characters are shortened with a short hash): separators do not matter, so `Configuring-Secure_Socket` and `Configuring Secure Socket` are the same name.
- **version** from a suffix at the end of the name (`_V2.0`, `-v1`, ` V3.1 Internal`, ` Rev 3`), so revisions of a guide share a family: `…_V1.0.pdf` and `…_V2.0.pdf` → family `…`, versions 1.0 and 2.0. "ENOVIA V6" / "CATIA V5" are product generations, not versions. A file without a version beside versioned ones is drafted as the newest; a file whose name matches a family already in the manifest gets a family of its own (joining would hide the listed document from search).
- **release** from the name (`R2021x`, `V6R2021x`, `2021x`, `21x`, `R2017xFP1705`, `R15xGA`) or, without one, from the PDF's first two pages; always as "from that release on": `release_min` set, `release_max` left open (set an end yourself when a guide stops applying). Several releases on the first pages are only noted.
- **category** guessed from words in the name and folders (85% agreement with the hand-made categories of the current corpus).
- **access**: `allowed_groups=all`, `external_ok=false`: review before `kb index`.

Older rows (without `added`, or without a release) are filled by `uv run kb manifest backfill` (shows the changes) and `uv run kb manifest backfill --write` (applies them, with a backup in `data/`): `added` from the first ingestion in `kb.db` (else the file date), the release from the name or the first pages, marked in `review`. Category, access and versions are not touched. Close the manifest in Excel first: Windows locks an open file. Saving in Excel rewrites the `added` dates in the machine's short-date format (e.g. `10/8/2026`); they are still accepted (US month/day and European day.month.year) and the next rewrite by `kb manifest scan` or `backfill --write` stores them as `YYYY-MM-DD` again.

Adding documents:

```bash
uv run kb manifest scan       # appends draft rows for new files (version, release and category guessed; see below)
uv run kb manifest validate   # reports every problem with its line number
```

Fill in title, category, groups and release range between the two commands.
