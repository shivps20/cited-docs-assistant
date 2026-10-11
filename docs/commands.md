# Command reference

Every `kb` command with its options and examples. Run them from the project root; `uv run` uses the project's virtual environment, so no activation is needed.

## Ingestion at a glance

```bash
uv run kb ingest [--folder PATH] [--retry-failed] [--dry-run]   # parse + chunk + index what needs it
uv run kb parse [--doc DOC_ID] [--force]   # parse with Docling; results cached in data/parsed/
uv run kb status                           # status and parse statistics per document
uv run kb inspect DOC_ID [--details]       # rebuilt section tree (numbers, pages, size)
uv run kb inspect DOC_ID --section 3.1.7   # content of one section as it will be chunked
uv run kb inspect DOC_ID --chunks [--section 3.1.7]   # chunks as they will be embedded (preview)
uv run kb chunk [--doc DOC_ID] [--force]   # sections + chunks in SQLite (new or changed documents only)
uv run kb coverage                         # are golden facts present in the stored chunks?
uv run kb index [--doc DOC_ID] [--force] [--prune]   # embed (bge-m3, GPU) and write to Qdrant
```

`kb index` re-embeds only documents that were (re)chunked; if only manifest values changed (groups, external_ok, release range, latest revision) it updates the Qdrant payload in place. `--prune` removes documents that are no longer in the manifest. Stop any loaded Ollama model first (`ollama stop <model>`): the 6 GB GPU cannot hold both.

Status and statistics are also stored in the `documents` table of `data/kb.db`.

### `kb ingest` — parse, chunk and index in one run

Runs the three steps below for every manifest document that needs them, and skips what is up to date: a document is parsed only when its file changed or its parse is missing (not even the cached parse is loaded otherwise), chunked when `kb chunk` would rebuild it, and embedded when it was (re)chunked or is not yet indexed; manifest-only changes (groups, release, latest revision) are applied to the index in place. So an interrupted or failed run is continued by simply running it again.

A document that fails a step is left out of the later steps and written to the failure log (`data/logs/ingest_failures.csv`, `KB_FAILURE_LOG`: run, time, step, document, error); the run ends with a list of the failures. Fix the cause, then `--retry-failed` runs only the documents that failed in the latest run. `kb status` shows the latest run's failures too. The parser is unloaded before bge-m3 loads; stop any loaded Ollama model first (`ollama stop <model>`).

<table style="width:100%">
<colgroup><col style="width:40%"><col style="width:60%"></colgroup>
<thead><tr><th>Option</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>--doc DOC_ID</code></td><td>Only this document (repeatable)</td></tr>
<tr><td><code>--folder PATH</code></td><td>Only manifest documents in this folder and its subfolders (e.g. a batch drafted with <code>kb manifest scan --folder</code>)</td></tr>
<tr><td><code>--retry-failed</code></td><td>Only the documents that failed in the latest run</td></tr>
<tr><td><code>--dry-run</code></td><td>Show how many documents each step would process, and which, without changing anything</td></tr>
</tbody>
</table>

```bash
uv run kb ingest --dry-run                    # what would be done
uv run kb ingest                              # everything that needs it
uv run kb ingest --folder "E:/Docs/New"       # one batch
uv run kb ingest --retry-failed               # after fixing the causes of the last run's failures
```

For rebuilds after code or `config/domain.yaml` changes use `kb chunk --force` (then `kb ingest` or `kb index`); `kb ingest` has no `--force`, so a run can never re-parse the whole corpus by accident. The single-step commands below remain for one document or one step.

> **Editing in Excel:** Excel rewrites versions such as `1.0` as `1` and `1.10` as `1.1`, which changes which revision counts as latest. Format the `version` column as Text, or edit the CSV in a text editor.

### Infrastructure

**Qdrant (Docker)**

```bash
docker compose -f deploy/docker-compose.yml up -d          # start Qdrant (also restarts with Docker Desktop)
docker compose -f deploy/docker-compose.yml ps             # is it running?
docker compose -f deploy/docker-compose.yml logs -f qdrant # follow the log (Ctrl+C to stop following)
docker compose -f deploy/docker-compose.yml down           # stop; data stays in the named volumes
docker compose -f deploy/docker-compose.yml pull           # fetch the image version pinned in the file
```

Dashboard: <http://127.0.0.1:6444/dashboard>. To change the port, edit the host side of `ports` in `deploy/docker-compose.yml` (`"127.0.0.1:<new>:6333"`), set `QDRANT_URL` in `.env` to match, then `docker compose -f deploy/docker-compose.yml up -d`.

**Ollama**

```bash
ollama pull qwen2.5:7b-instruct   # download the LLM (once)
ollama ps                         # models loaded right now (and whether on GPU)
ollama stop qwen2.5:7b-instruct   # unload it, freeing the GPU before ingestion
```

### Setup scripts

<table style="width:100%">
<colgroup><col style="width:19%"><col style="width:28%"><col style="width:53%"></colgroup>
<thead><tr><th>Command</th><th>Options</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>uv sync</code></td><td></td><td>Create <code>.venv</code> and install dependencies (CUDA torch from the PyTorch index)</td></tr>
<tr><td><code>uv run python scripts/init_qdrant.py</code></td><td><code>--recreate</code> delete and recreate the collection (drops all points); <code>--yes</code> skip the confirmation</td><td>Create the <code>kb_chunks</code> collection (dense 1024 cosine + sparse) and its payload indexes; safe to re-run</td></tr>
<tr><td><code>uv run python scripts/init_db.py</code></td><td><code>--reset</code> delete and recreate the database; <code>--yes</code> skip the confirmation</td><td>Create <code>data/kb.db</code> or apply pending schema migrations (needed after pulling code with a new migration: the <code>kb</code> commands and the server refuse an out-of-date schema)</td></tr>
<tr><td><code>uv run python scripts/download_models.py</code></td><td><code>--verify-only</code> skip downloads, only run the smoke tests</td><td>Download bge-m3, bge-reranker-v2-m3 and Docling models into <code>./models</code>, then verify they load offline</td></tr>
<tr><td><code>uv run python scripts/check_env.py</code></td><td><code>--deep</code> also load the LLM and generate a token</td><td>Health check: .env, Qdrant, SQLite schema, Ollama, GPU, models, manifest, disk</td></tr>
</tbody>
</table>

```bash
uv run python scripts/check_env.py --deep
uv run python scripts/init_qdrant.py --recreate --yes   # start the index from scratch, then `kb index`
uv run python scripts/init_db.py --reset --yes          # empty database, then parse/chunk/index again
```

### `kb manifest`

<table style="width:100%">
<colgroup><col style="width:40%"><col style="width:60%"></colgroup>
<thead><tr><th>Command</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>uv run kb manifest scan [--folder PATH] [--dry-run]</code></td><td>Draft rows for the files (types in <code>KB_DOC_TYPES</code>) in <code>KB_DOCS_DIR</code>, or in <code>--folder</code> (any folder, read where it is: rows get absolute paths), and its subfolders that are not in the manifest: a count per subfolder, files whose content is already listed (or found twice) skipped and reported, a PDF whose deck or Word file of the same name is listed or found skipped too (only the preferred format is drafted: <code>.pptx</code>, then <code>.pdf</code>, then <code>.docx</code>), doc ids kept unique across subfolders (same file name in two folders gets the folder name as prefix). <code>--dry-run</code> shows what would be drafted and writes nothing</td></tr>
<tr><td><code>uv run kb manifest validate</code></td><td>Check every row (columns, categories, releases, dates, duplicate IDs, missing files, two rows with the same family and version); lists all problems with line numbers and how to fix a family clash; counts and lists rows still marked for review. Then warnings, which do not fail the check but should be looked at before <code>kb ingest</code>: one document in two formats (which line to keep), the same file name in several folders, rows open to <code>all</code> whose path says internal, confidential or restricted, a family whose highest version has an older release than an earlier version, and likely editions of one document in different families (all searched as latest); warnings are shown even when the manifest has errors, since they often explain them</td></tr>
<tr><td><code>uv run kb manifest backfill [--write]</code></td><td>Fill what older rows lack: the <code>added</code> date (first ingestion in <code>kb.db</code>, else the file date) and, for rows without a release, the release from the file name or the PDF's first pages. Shows the changes; <code>--write</code> applies them (backup in <code>data/</code>) and notes each in the <code>review</code> column</td></tr>
</tbody>
</table>

Run `validate` after every edit to `config/manifest.csv`.

**Documents where they already are.** Keep `KB_DOCS_DIR=data/documents` and add any other folder with `uv run kb manifest scan --folder "E:/Docs/New"`: its files are read where they are (the manifest holds their absolute path), so nothing has to be copied. Start with `--dry-run` to see what a folder holds; files whose content is already in the manifest are skipped. `.ppt` / `.doc` are converted with LibreOffice when parsed (once, cached). Drafted rows give access to everyone (`allowed_groups=all`) and keep `external_ok=false`: review access, category, title, version and release range before `kb index`.

### `kb parse` — Docling parsing

<table style="width:100%">
<colgroup><col style="width:40%"><col style="width:60%"></colgroup>
<thead><tr><th>Option</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>--doc DOC_ID</code></td><td>Only this document; repeatable</td></tr>
<tr><td><code>--force</code></td><td>Re-parse even if a cached result exists in <code>data/parsed/</code></td></tr>
</tbody>
</table>

```bash
uv run kb parse                                       # all new or changed documents
uv run kb parse --doc sso-setup --force
```

Parsing typically takes 0.1–0.5 s per page on the GPU (a 214-page guide in about 30 s). Long batches (`ingest`, `parse`, `chunk`, `index`, `coverage`, `eval`, `eval-answers`) keep Windows awake and the display on while they run (with the screen off, Modern Standby can pause the process even when plugged in); keep a laptop plugged in with the lid open. Documents are parsed in a separate worker process that is replaced every 50 documents (the parser leaves threads behind; a worker that crashes, or a document still parsing after 30 minutes, fails only that document), so Docling's "Loading weights" line appears again every 50 documents. In a CMD or PowerShell window, selecting text pauses the program writing to it until you press Esc: for long runs, send the output to a file (`uv run kb ingest > data\logs\ingest-run.log 2>&1`) and follow it in another window (`powershell -Command "Get-Content data\logs\ingest-run.log -Wait -Tail 20"`). `kb parse` prints a WARN for any document that took over 2 minutes while using under a tenth of that in CPU time (the machine was asleep or throttled); re-parse such a document with `--doc DOC_ID --force` to record its real time.

### `kb status`

Status per document (parsed / chunked / indexed), parse statistics (pages, seconds, headings, tables, pictures, empty pages) and chunk counts, then a summary per status and the latest `kb ingest` run with its failures per step. No options.

### `kb inspect` — review a document's structure and chunks

<table style="width:100%">
<colgroup><col style="width:40%"><col style="width:60%"></colgroup>
<thead><tr><th>Option</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>DOC_ID</code></td><td>Document to show (required)</td></tr>
<tr><td><code>--section NUMBER</code></td><td>Print one section's content, e.g. <code>3.1.7</code></td></tr>
<tr><td><code>--details</code></td><td>Also list unmatched TOC entries, demoted headings and removed repeated lines</td></tr>
<tr><td><code>--chunks</code></td><td>Show the chunks as they will be embedded (all, or of <code>--section</code>)</td></tr>
</tbody>
</table>

```bash
uv run kb inspect install-guide                 # section tree with pages and sizes
uv run kb inspect install-guide --details       # what the structure builder could not place
uv run kb inspect install-guide --section 3.1           # one section's text
uv run kb inspect install-guide --chunks --section 3.1  # that section's chunks
```

For a slide deck exported to PDF (landscape pages) whose PDF structure would be misread (one section with more than 40 sub-sections: a running heading or an agenda slide taken as a chapter), the output starts with "Built like a deck": such PDFs get one section per page, titled by the page's heading or first line, and their section numbers are page numbers.

### `kb chunk` — build and store sections and chunks

<table style="width:100%">
<colgroup><col style="width:40%"><col style="width:60%"></colgroup>
<thead><tr><th>Option</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>--doc DOC_ID</code></td><td>Only this document; repeatable</td></tr>
<tr><td><code>--force</code></td><td>Rebuild documents that are up to date too: needed after changes to the structure or chunking code or to <code>config/domain.yaml</code></td></tr>
</tbody>
</table>

Builds sections and chunks from the cached parse and replaces the document's sections and chunks in SQLite; the next `kb index` then re-embeds it. A document is skipped when it is already chunked (or indexed) from the same parse, with the current chunker version, and its stored chunk headers still carry the manifest's current title and release; a changed file (re-parsed), a new title or release, or an older chunker version rebuilds it. So a plain `kb chunk` after adding documents only chunks the new ones, and `kb index` only embeds those.

```bash
uv run kb chunk                                      # new or changed documents
uv run kb chunk --doc sso-setup --doc role-overview  # these two (if not up to date)
uv run kb chunk --force                              # everything, after a code or domain.yaml change
```

### `kb coverage` — golden facts present in the chunks?

<table style="width:100%">
<colgroup><col style="width:40%"><col style="width:60%"></colgroup>
<thead><tr><th>Option</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>--all</code></td><td>List passing questions too, not only failures</td></tr>
</tbody>
</table>

Checks that every `must_include` string of the golden set appears in a stored chunk of the cited document and pages. Expected output: `40/40 answerable golden questions fully covered`.

### `kb index` — embed and write to Qdrant

<table style="width:100%">
<colgroup><col style="width:40%"><col style="width:60%"></colgroup>
<thead><tr><th>Option</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>--doc DOC_ID</code></td><td>Only this document; repeatable</td></tr>
<tr><td><code>--force</code></td><td>Re-embed even if already indexed</td></tr>
<tr><td><code>--prune</code></td><td>Delete points and rows of documents no longer in the manifest</td></tr>
</tbody>
</table>

Re-embeds only (re)chunked documents; manifest-only changes update the Qdrant payload in place. Uses the GPU: run `ollama stop <model>` first.

```bash
uv run kb index
uv run kb index --doc install-guide --force
uv run kb index --prune          # after removing rows from config/manifest.csv
```

### `kb search` — retrieval without the LLM

<table style="width:100%">
<colgroup><col style="width:32%"><col style="width:8%"><col style="width:60%"></colgroup>
<thead><tr><th>Option</th><th>Default</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>QUERY</code></td><td></td><td>The question (required, in quotes)</td></tr>
<tr><td><code>--groups GROUP</code></td><td><code>all</code></td><td>User access group(s); repeatable or comma-separated</td></tr>
<tr><td><code>--release RELEASE</code></td><td>any</td><td>Only documents that apply to this release, e.g. <code>R2024x</code></td></tr>
<tr><td><code>--mode</code></td><td><code>hybrid</code></td><td><code>hybrid</code>, <code>dense</code> or <code>sparse</code></td></tr>
<tr><td><code>--no-rerank</code></td><td></td><td>Skip the cross-encoder (fast; search order only)</td></tr>
<tr><td><code>--candidates N</code></td><td>30</td><td>Chunks retrieved before reranking</td></tr>
<tr><td><code>--rerank-top N</code></td><td><code>KB_RERANK_TOP</code> (20)</td><td>Rerank only the first N candidates; <code>0</code> = all</td></tr>
<tr><td><code>--max-length N</code></td><td>512</td><td>Reranker input length in tokens</td></tr>
<tr><td><code>--min-score X</code></td><td>off</td><td>Drop context chunks with rerank score below X</td></tr>
<tr><td><code>--show N</code></td><td>10</td><td>Ranked results to print</td></tr>
<tr><td><code>--context</code></td><td></td><td>Print the full assembled context the LLM would receive</td></tr>
</tbody>
</table>

```bash
uv run kb search "Which port does the application server use?"
uv run kb search "Which port does the application server use?" --mode dense --no-rerank   # fast, no reranker
uv run kb search "Which tool captures traffic for a performance analysis?"                       # restricted doc hidden
uv run kb search "Which tool captures traffic for a performance analysis?" --groups internal  # visible with access
uv run kb search "Which SQL Server version is supported?" --release R2025x      # release filter
uv run kb search "How do I enable single sign-on?" --context --show 5         # see what the LLM gets
uv run kb search "..." --rerank-top 0 --min-score 0.3                           # rerank all, drop weak context
```

### `kb models` — model catalogue

<table style="width:100%">
<colgroup><col style="width:40%"><col style="width:60%"></colgroup>
<thead><tr><th>Command</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>uv run kb models list</code></td><td>Validate <code>config/models.yaml</code> (every problem listed at once) and show each model: adapter, local / external, whether its key is set, context and output size, model id and the roles it has</td></tr>
<tr><td><code>uv run kb models check [--model NAME]</code></td><td>Send every configured model (or the named ones) a tiny text prompt and a JSON prompt: reachable, seconds, token counts, whether JSON works. Models without a key are skipped; costs a few tokens per model</td></tr>
</tbody>
</table>

Without `config/models.yaml` it shows the catalogue built from the older settings (`LLM_MODEL`, `OPENAI_*`).

### `kb cache` — answer cache

<table style="width:100%">
<colgroup><col style="width:40%"><col style="width:60%"></colgroup>
<thead><tr><th>Command</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>uv run kb cache stats</code></td><td>Entries (of the current corpus and stale), answers served from the cache, the most-served questions</td></tr>
<tr><td><code>uv run kb cache clear</code></td><td>Remove every entry</td></tr>
<tr><td><code>uv run kb cache clear --stale</code></td><td>Remove only entries of an older corpus (they can no longer be hit; storing a new answer also removes them)</td></tr>
</tbody>
</table>

### `kb ask` — answer a question with citations

<table style="width:100%">
<colgroup><col style="width:26%"><col style="width:14%"><col style="width:60%"></colgroup>
<thead><tr><th>Option</th><th>Default</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>QUERY</code></td><td></td><td>The question (required, in quotes)</td></tr>
<tr><td><code>--groups GROUP</code></td><td><code>all</code></td><td>User access group(s); repeatable or comma-separated</td></tr>
<tr><td><code>--release RELEASE</code></td><td>any</td><td>Only documents that apply to this release, e.g. <code>R2024x</code></td></tr>
<tr><td><code>--model NAME</code></td><td>the answer role in <code>config/models.yaml</code></td><td>Answer model by catalogue name (<code>auto</code>, <code>ollama</code>, <code>openai</code> still accepted). External models only see documents with <code>external_ok</code>; otherwise the local fallback answers, with a notice</td></tr>
<tr><td><code>--mode</code></td><td><code>hybrid</code></td><td><code>hybrid</code>, <code>dense</code> or <code>sparse</code></td></tr>
<tr><td><code>--no-rerank</code></td><td></td><td>Skip the reranker; without a rerank score the gate always passes</td></tr>
<tr><td><code>--rerank-top N</code></td><td><code>KB_RERANK_TOP</code> (20)</td><td>Rerank only the first N candidates; <code>0</code> = all</td></tr>
<tr><td><code>--show-context</code></td><td></td><td>Print the context sent to the LLM before the answer</td></tr>
<tr><td><code>--no-stream</code></td><td></td><td>Print the answer only when it is complete</td></tr>
<tr><td><code>--no-compare</code></td><td></td><td>Answer comparison questions with one search instead of one search per side</td></tr>
<tr><td><code>--no-refusal-retry</code></td><td><code>KB_REFUSAL_RETRY</code> (on)</td><td>No second attempt with the best-matching sources when the model finds no answer</td></tr>
<tr><td><code>--no-cache</code></td><td><code>KB_ANSWER_CACHE</code> (on)</td><td>Neither read nor store the answer cache: always search and ask the model</td></tr>
</tbody>
</table>

After the answer it prints the sources (each with its file path below it, full or relative per `KB_SOURCE_PATH`), any notes (removed citations, provider choice, a comparison answered with one search, an answer found on the second attempt), the status (`answered`, or `not found` by the gate or by the LLM), timings and the trace ID.

```bash
uv run kb ask "How do I enable single sign-on?"
uv run kb ask "Which port does the application server use?" --show-context              # see what the LLM was given
uv run kb ask "Which tool captures traffic for a performance analysis?" --groups internal   # restricted document
uv run kb ask "Which SQL Server version is supported?" --release R2026x
uv run kb ask "How do I configure NGINX as a reverse proxy?"                   # expect "not found"
uv run kb ask "How does the database setup differ between MSSQL and Oracle?"   # comparison: one search per side
uv run kb ask "..." --model claude-opus                                      # a model from config/models.yaml (key in .env)
```

### `kb trace` — behind the scenes of one answer

```bash
uv run kb trace                      # the latest answer
uv run kb trace 7b93a3da             # by trace id or its first characters (as shown under each answer)
uv run kb trace 7b93a3da --prompt    # also print the exact messages sent to the model
uv run kb trace 7b93a3da --all       # every search candidate, not only the reranked ones
uv run kb trace 7b93a3da --faithfulness   # run the faithfulness judge on it (one LLM call; stored with the trace)
uv run kb trace 7b93a3da --json      # the whole view as JSON
```

Shows, from the answer's trace: the question and filters; the gate (top rerank score against the threshold); every candidate with its rerank rank and score, its search rank and retrieval (fusion) score, and whether it reached the context; the context units sent to the model; each generation's model, tokens, speed and fallbacks, and the prompt; the checks (citations kept and removed, the refusal retry, commands not found word for word in the context, the faithfulness verdict if checked); the stage timings. `--faithfulness --force` judges again. Answers traced before 2026-10-11 show what was recorded then: the top 10 rerank scores only, the context from the assembly step, and no prompt. The chat page shows the same in **Behind the scenes** under each answer.

### `kb serve` — chat API

Starts the chat API and web UI; endpoints, the chat turn and the UI are described in [api.md](api.md).

<table style="width:100%">
<colgroup><col style="width:12%"><col style="width:29%"><col style="width:59%"></colgroup>
<thead><tr><th>Option</th><th>Default</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>--host</code></td><td><code>KB_API_HOST</code> (127.0.0.1)</td><td>Interface to listen on; keep the loopback address (users are not authenticated)</td></tr>
<tr><td><code>--port</code></td><td><code>KB_API_PORT</code> (8000)</td><td>Port</td></tr>
</tbody>
</table>

```bash
uv run kb serve                 # http://127.0.0.1:8000 ; models load once (~20 s)
uv run kb serve --port 8012     # a second instance, e.g. while another one runs
```

### `kb eval`, `kb eval-answers` — evaluation

Options, metrics and results are described in [evaluation.md](evaluation.md).

```bash
uv run kb eval --configs dense,hybrid+rr10 --misses        # retrieval only (~7 min with the reranker)
uv run kb eval-answers --questions q012,q043 --details     # a few questions with the judge (~1 min)
uv run kb eval-answers --types compare                     # one question type
uv run kb eval-answers --no-judge                          # whole set without the judge (~10 s per question)
uv run kb eval-answers                                     # whole set with the judge (~1 h on the GPU)
uv run kb eval-answers --model claude-opus --judge-model local-qwen   # another answer model, same judge
```

Reports go to `data/eval/` (`retrieval-*.json`, `answers-*.json`).

### `kb calibrate` — the "not found" threshold

Replays candidate gate thresholds (`KB_NOT_FOUND_SCORE`) on the latest full `kb eval-answers` report per answer model, without running any model, and recommends one. Details in [evaluation.md](evaluation.md#kb-calibrate--the-not-found-threshold).

<table style="width:100%">
<colgroup><col style="width:17%"><col style="width:24%"><col style="width:59%"></colgroup>
<thead><tr><th>Option</th><th>Default</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><code>--report PATH</code></td><td>latest full run per answer model</td><td>Answer evaluation report(s) to use; repeatable</td></tr>
<tr><td><code>--thresholds LIST</code></td><td>0 to 0.9</td><td>Candidate thresholds, comma-separated</td></tr>
<tr><td><code>--all</code></td><td>reviewed questions only</td><td>Count every golden question</td></tr>
<tr><td><code>--details</code></td><td></td><td>List the questions lost or unmeasured per threshold</td></tr>
</tbody>
</table>

```bash
uv run kb calibrate                                   # after a full kb eval-answers
uv run kb calibrate --all --details                   # every question, with the questions behind each number
uv run kb calibrate --report data/eval/answers-<timestamp>.json --thresholds 0.05,0.1,0.15
```

The output ends with a recommended threshold, the refusals that are generation problems rather than the gate's, the number of real questions the gate refuses, and the 👎 reasons from the chat; the result is also saved as `data/eval/calibration-<timestamp>.json`. To change the threshold, set `KB_NOT_FOUND_SCORE` in `.env` and run `kb eval-answers` again.

### Typical workflows

**New or changed document**

```bash
uv run kb manifest scan                # draft a manifest row for each new file; then review access, groups and the review notes
uv run kb manifest validate            # check every row (errors must be fixed) and show warnings (formats, folders, access)
ollama stop qwen2.5:7b-instruct        # free the GPU: parsing and embedding need it
uv run kb ingest                       # parse, chunk and index only what is new or changed; failures go to the failure log
uv run kb inspect <doc_id> --details   # check the new document's section structure (titles, pages, what was removed)
uv run kb coverage                     # confirm the golden questions' facts are still present in the stored chunks
```

**A large batch (hundreds of documents): parse first, check, then finish.** Parsing is the long step (hours) and the one most likely to fail, so stop after it and look at the results before anything is embedded.

```bash
uv run kb manifest scan --folder "E:/Docs/New"   # draft rows; review access, groups and the review notes
uv run kb manifest validate                       # no errors; look at every warning (formats, folders, access)
uv run kb ingest --dry-run                        # how many documents each step would process
ollama stop qwen2.5:7b-instruct
uv run kb parse                                   # new or changed documents; already parsed ones load from the cache
uv run kb status                                  # failures, documents with few texts or many empty pages
uv run kb inspect <doc_id> --details              # spot checks: section titles of decks, Word files
uv run kb ingest                                  # parse is done, so this chunks and indexes
uv run kb status                                  # all documents indexed; the latest run's failures, if any
```

Check after `kb parse`: documents marked FAILED (re-run with `uv run kb parse --doc <doc_id>`, or leave them to `kb ingest`, which retries anything not yet parsed); image-only PDFs (many pages, almost no texts: candidates for OCR); decks whose sections are all "Slide N". `kb parse` prints its failures but does not write the failure log; `kb ingest` does, so `uv run kb ingest --retry-failed` covers failures from the second half. For a few documents, `kb ingest` alone is simpler.

**After changing the structure or chunking code, or `config/domain.yaml`:** `uv run kb chunk --force` (all, or the affected `--doc`s), then `uv run kb index`, `uv run kb coverage` and `uv run kb eval --configs dense,hybrid+rr10`.

**Only manifest values changed** (groups, external_ok, release range, version): `uv run kb manifest validate`, then `uv run kb index` updates the payloads without re-embedding.

Either way the corpus changes, so earlier cached answers are no longer served; remove them with `uv run kb cache clear --stale` (storing the next answer also does it).

**After pulling new code**

```bash
uv sync                                    # new or updated dependencies
uv run python scripts/init_db.py           # apply new schema migrations (e.g. v6: feedback reasons)
uv run pytest -q
uv run python scripts/check_env.py
```

**Changing the language models** (`config/models.yaml`)

```bash
uv run kb models list                      # validate: every problem listed at once
uv run kb models check --model claude-opus # reachable, speed, JSON (costs a few tokens)
uv run kb ask "..." --model claude-opus    # try it on one question
```

A running `kb serve` picks up the changed file without a restart; cached answers of the old settings are no longer served.

**Measuring a change in answers** (prompt, retrieval settings, answer model)

```bash
uv run kb eval-answers --types compare     # the affected type first
uv run kb eval-answers                     # the whole set before closing a phase
uv run kb calibrate                        # is the "not found" threshold still right?
```

Compare the summary with the previous report in `data/eval/`; every report records the models, settings and a hash of the prompt.

**After editing the golden set** (`eval/golden.json`)

```bash
uv run python eval/check_golden.py         # must_include in the expected answers, pages and sources exist
uv run kb coverage                         # the key facts are in the stored chunks
```

Mark reviewed questions with `"status": "reviewed"`: `kb calibrate` counts only those by default.

### Development and checks

```bash
uv run pytest -q                          # test suite, quiet output
uv run pytest tests/ingest/test_structure.py -q   # one test file (tests mirror src/kb)
uv run pytest -k cache -q                 # tests whose name contains "cache"
uv run ruff check src tests scripts eval  # lint
uv run ruff check --fix src tests scripts eval   # lint and apply safe fixes
uv run python eval/check_golden.py        # golden set consistency, after editing eval/golden.json
```

### Inspecting the data

- **Qdrant:** dashboard at <http://127.0.0.1:6444/dashboard> → Collections → `kb_chunks`.
- **SQLite:** open `data/kb.db` read-only in DB Browser for SQLite. Useful queries:

```sql
SELECT doc_id, status, page_count, chunk_count, parse_seconds, embed_seconds FROM documents;
SELECT section_id, heading_path, page_start, page_end, token_count FROM sections WHERE doc_id = 'install-guide';
SELECT trace_id, route, query, top_rerank_score, gate_decision, llm_model, total_ms FROM traces ORDER BY created_at DESC LIMIT 20;
SELECT query, answer, sources FROM traces WHERE route = 'answer' ORDER BY created_at DESC LIMIT 5;
SELECT stage, duration_ms, data FROM trace_stages WHERE trace_id = '<trace_id>' ORDER BY seq;
SELECT f.rating, f.reason, f.comment, t.query, t.top_rerank_score, t.gate_decision FROM feedback f JOIN traces t USING (trace_id);
SELECT query_norm, user_groups, release_filter, hit_count, created_at, last_hit_at FROM answer_cache ORDER BY hit_count DESC;
SELECT query, cache_hit, total_ms FROM traces WHERE cache_hit = 1 ORDER BY created_at DESC LIMIT 10;
```

### Local project notes (not in the repository)

The design document, plan, trade-offs, technical debt and code-flow pages live in `docs/local/`, which is git-ignored because they use real document names and questions. The design document is written in Markdown (`docs/local/DESIGN.md`) and also built as one offline HTML page with the diagrams drawn in (`docs/local/DESIGN.html`):

```bash
uv run python docs/local/design_build/build_design_html.py   # rebuild DESIGN.html after editing DESIGN.md
start docs/local/DESIGN.html                                 # open it (Windows; macOS: open, Linux: xdg-open)
```

After changing or adding a diagram in `DESIGN.md`, render the diagrams again before rebuilding:

```bash
uv run python docs/local/design_build/build_design_html.py   # also writes the diagram sources (blocks.json)
python docs/local/design_build/save_server.py                # serves the build folder on http://127.0.0.1:8766
```

Open <http://127.0.0.1:8766/render.html>, click **Render all** (it loads Mermaid from a CDN, only at this step), stop the server with Ctrl+C, and run the build script once more. Never commit or publish `DESIGN.md` or `DESIGN.html`.

### Git workflow (one branch per phase)

```bash
git checkout -b phase-3-answers                        # start a phase
git add <files>                                        # stage explicit paths
git commit -m "Phase 3 step 1: ..."
git push -u origin phase-3-answers                     # first push of the branch; later just `git push`
git checkout master                                    # finish the phase
git pull
git merge --no-ff phase-3-answers -m "Merge Phase 3: answer generation"
git push
```
