# Configuration

Settings (`.env`), the organisation-specific files in `config/`, and the document manifest.

## Settings (.env)

All settings come from `.env` (see [.env.example](../.env.example)) through `kb.core.config.Settings`.

<table style="width:100%">
<colgroup><col style="width:15%"><col style="width:16%"><col style="width:69%"></colgroup>
<thead><tr><th>Variable</th><th>Default</th><th></th></tr></thead>
<tbody>
<tr><td><code>QDRANT_URL</code>, <code>QDRANT_COLLECTION</code></td><td><code>http://127.0.0.1:6444</code>, <code>kb_chunks</code></td><td>Vector store</td></tr>
<tr><td><code>KB_DB_PATH</code></td><td><code>data/kb.db</code></td><td>SQLite database</td></tr>
<tr><td><code>KB_DOCS_DIR</code>, <code>KB_MANIFEST_PATH</code></td><td><code>data/documents</code>, <code>config/manifest.csv</code></td><td>The documents folder (subfolders included; manifest paths are relative to it) and the manifest. Documents elsewhere are added with <code>kb manifest scan --folder PATH</code> and read in place (absolute paths in the manifest)</td></tr>
<tr><td><code>KB_DOC_TYPES</code></td><td><code>pdf,pptx,ppt,docx,doc</code></td><td>File types picked up by <code>kb manifest scan</code> and accepted in the manifest. <code>.ppt</code> / <code>.doc</code> (old binary formats) are converted once to <code>.pptx</code> / <code>.docx</code> with LibreOffice before parsing, cached in <code>data/parsed/converted/</code></td></tr>
<tr><td><code>KB_SOFFICE</code></td><td><em>(found automatically)</em></td><td>Path to LibreOffice's <code>soffice</code> when it is not on the PATH or in <code>C:/Program Files/LibreOffice</code></td></tr>
<tr><td><code>KB_SOURCE_PATH</code></td><td><code>full</code></td><td>How every source shows its file (in <code>kb ask</code>, <code>kb search</code>, the chat's Sources and context panel, evaluation reports): <code>full</code> = the absolute path; <code>relative</code> = the path relative to <code>KB_DOCS_DIR</code>, or only the file name for a document kept elsewhere (for a server: no disk layout shown). Display only: stored answers, traces and the answer cache keep the manifest form, so switching it changes nothing else</td></tr>
<tr><td><code>EMBED_MODEL_PATH</code>, <code>RERANK_MODEL_PATH</code>, <code>DOCLING_ARTIFACTS_PATH</code></td><td><code>models/...</code></td><td>Local models</td></tr>
<tr><td><code>HF_HUB_OFFLINE</code></td><td><code>1</code></td><td>Never download models at runtime</td></tr>
<tr><td><code>KB_RERANK_TOP</code></td><td><code>20</code></td><td>Candidates reranked per search (see <a href="architecture.md#retrieval">architecture.md</a>)</td></tr>
<tr><td><code>KB_NOT_FOUND_SCORE</code></td><td><code>0.1</code></td><td>Reply &quot;not found&quot; without the LLM below this top rerank score (see <a href="architecture.md#answering">architecture.md</a>); check a value with <code>kb calibrate</code> (<a href="evaluation.md#kb-calibrate--the-not-found-threshold">evaluation.md</a>)</td></tr>
<tr><td><code>KB_REFUSAL_RETRY</code></td><td><code>true</code></td><td>When the model replies &quot;not found&quot;, ask once more with only the best-matching sources (3, or 2 per side of a comparison)</td></tr>
<tr><td><code>KB_ANSWER_CACHE</code></td><td><code>true</code></td><td>Answer a repeated question from the answer cache (same question, groups, release, corpus, model and prompt; see <a href="architecture.md#answer-cache">architecture.md</a>)</td></tr>
<tr><td><code>KB_LLM_PROVIDER</code></td><td><code>auto</code></td><td><code>auto</code> (the catalogue's answer role), <code>ollama</code> (the local fallback) or <code>openai</code>; superseded by <code>config/models.yaml</code> roles</td></tr>
<tr><td><code>KB_USERS_PATH</code></td><td><code>config/users.yaml</code></td><td>Chat API users and their access groups</td></tr>
<tr><td><code>KB_API_HOST</code>, <code>KB_API_PORT</code></td><td><code>127.0.0.1</code>, <code>8000</code></td><td>Where <code>kb serve</code> listens (keep loopback: no authentication)</td></tr>
<tr><td><code>KB_DOMAIN_PATH</code></td><td><code>config/domain.yaml</code></td><td>Organisation-specific text rules (see below)</td></tr>
<tr><td><code>KB_GOLDEN_PATH</code></td><td><code>eval/golden.json</code></td><td>Golden question set used by <code>kb coverage</code>, <code>kb eval</code>, <code>kb eval-answers</code>, <code>kb calibrate</code> and <code>eval/check_golden.py</code></td></tr>
<tr><td><code>KB_MODELS_PATH</code></td><td><code>config/models.yaml</code></td><td>Model catalogue: which language models exist and which job each one does (below). Without the file: <code>LLM_MODEL</code>, plus OpenAI when <code>OPENAI_*</code> is set</td></tr>
<tr><td><code>LLM_TEMPERATURE</code>, <code>LLM_MAX_TOKENS</code></td><td><code>0</code>, <code>1500</code></td><td>Ollama sampling temperature; answer length cap (both providers)</td></tr>
<tr><td><code>OLLAMA_HOST</code>, <code>LLM_MODEL</code>, <code>LLM_NUM_CTX</code></td><td><code>127.0.0.1:11434</code>, <code>qwen2.5:7b-instruct</code>, <code>8192</code></td><td>Local LLM</td></tr>
<tr><td><code>OPENAI_API_KEY</code>, <code>OPENAI_MODEL</code></td><td><em>(empty)</em></td><td>Optional external LLM when there is no <code>config/models.yaml</code> (with a catalogue, OpenAI is a catalogue entry)</td></tr>
</tbody>
</table>

## Organisation-specific data (kept local)

Nothing that comes from the ingested documents, and no rule that names their publisher, is committed. These files live only on your machine (git-ignored); the repository has fictional `*.example.*` versions to copy:

<table style="width:100%">
<colgroup><col style="width:17%"><col style="width:27%"><col style="width:56%"></colgroup>
<thead><tr><th>Local file</th><th>Example in the repository</th><th>Holds</th></tr></thead>
<tbody>
<tr><td><code>config/manifest.csv</code></td><td><a href="../config/manifest.example.csv">config/manifest.example.csv</a></td><td>Which documents are ingested, their titles, releases, access groups</td></tr>
<tr><td><code>config/users.yaml</code></td><td><a href="../config/users.example.yaml">config/users.example.yaml</a></td><td>Chat API users and their access groups</td></tr>
<tr><td><code>config/domain.yaml</code></td><td><a href="../config/domain.example.yaml">config/domain.example.yaml</a></td><td>Text rules specific to whose documents you ingest (below)</td></tr>
<tr><td><code>config/models.yaml</code></td><td><a href="../config/models.example.yaml">config/models.example.yaml</a></td><td>Language models (local, OpenAI, Mistral, Gemini, Claude) and the job of each (below)</td></tr>
<tr><td><code>eval/golden.json</code></td><td><a href="../eval/golden.example.json">eval/golden.example.json</a></td><td>Golden questions with expected answers taken from the documents</td></tr>
</tbody>
</table>

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

<table style="width:100%">
<colgroup><col style="width:10%"><col style="width:25%"><col style="width:65%"></colgroup>
<thead><tr><th>Column</th><th>Example</th><th>Purpose</th></tr></thead>
<tbody>
<tr><td><code>doc_id</code></td><td><code>install-guide</code></td><td>Stable ID used in citations and re-ingestion</td></tr>
<tr><td><code>path</code></td><td><code>Install/Acme_Platform_Installation_Guide.pdf</code></td><td>File, relative to <code>KB_DOCS_DIR</code> (may include subfolders), or an absolute path such as <code>E:/Docs/New/Guide.pptx</code> for a document kept outside it</td></tr>
<tr><td><code>title</code></td><td>Acme Platform Installation Guide</td><td>Shown in answer citations</td></tr>
<tr><td><code>family</code>, <code>version</code></td><td><code>install-guide</code>, <code>2.0</code></td><td>The highest version per family is the latest revision</td></tr>
<tr><td><code>release_min</code>, <code>release_max</code></td><td><code>R2024x</code>, <em>(blank)</em></td><td>Release range the document applies to; blank = open-ended</td></tr>
<tr><td><code>allowed_groups</code></td><td><code>all</code> or <code>internal</code></td><td><code>;</code>-separated access groups</td></tr>
<tr><td><code>external_ok</code></td><td><code>true</code></td><td>May its text be sent to an external LLM (OpenAI, Mistral, Gemini, Claude)</td></tr>
<tr><td><code>category</code></td><td><code>installation</code></td><td>One of: installation, administration, authentication, infrastructure, upgrade, performance, applications, functional, troubleshooting (defined in <code>kb.manifest.CATEGORIES</code>)</td></tr>
<tr><td><code>added</code></td><td><code>2026-10-10</code></td><td>Date the row was added (<code>YYYY-MM-DD</code>, blank allowed); filled by <code>kb manifest scan</code> and, for older rows, <code>kb manifest backfill</code></td></tr>
<tr><td><code>review</code></td><td><code>category guessed from the name</code></td><td>What still needs a human check; blank = reviewed. <code>kb manifest validate</code> counts and lists these rows</td></tr>
</tbody>
</table>

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
